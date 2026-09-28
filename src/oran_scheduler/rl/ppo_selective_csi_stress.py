from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
)


MultiCellPPOInputProvider = Callable[
    [int, int],
    PPOTrainingTTIInputs,
]


_VALID_MODES = {
    "all",
    "ppo_radio",
    "td_rate_only",
    "cqi_only",
    "rank_only",
    "precoder_only",
    "none",
}


def stale_fields_for_mode(
    mode: str,
) -> frozenset[str]:

    mapping = {
        "all": frozenset(
            {
                "td_rate",
                "rank",
                "cqi",
                "precoder",
            }
        ),

        #
        # Keep PF-TDS instantaneous-rate input fresh,
        # while PPO sees stale radio-state features.
        #
        "ppo_radio": frozenset(
            {
                "rank",
                "cqi",
                "precoder",
            }
        ),

        "td_rate_only": frozenset(
            {
                "td_rate",
            }
        ),

        "cqi_only": frozenset(
            {
                "cqi",
            }
        ),

        "rank_only": frozenset(
            {
                "rank",
            }
        ),

        "precoder_only": frozenset(
            {
                "precoder",
            }
        ),

        "none": frozenset(),
    }

    if mode not in mapping:
        raise ValueError(
            f"Unsupported CSI-staleness mode: {mode}. "
            f"Expected one of {sorted(_VALID_MODES)}."
        )

    return mapping[mode]


@dataclass(frozen=True)
class SelectiveCSIStalenessConfig:

    delay_ttis: int = 0
    mode: str = "all"

    def __post_init__(
        self,
    ) -> None:

        if self.delay_ttis < 0:
            raise ValueError(
                "delay_ttis must be non-negative."
            )

        stale_fields_for_mode(
            self.mode
        )


def _clone_observation(
    observation: OneLDSCellTTIObservation,
) -> OneLDSCellTTIObservation:

    return OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            observation
            .serving_global_ue_indices
            .detach()
            .clone()
        ),

        serving_ue_valid_mask=(
            observation
            .serving_ue_valid_mask
            .detach()
            .clone()
        ),

        td_instantaneous_rate_bps=(
            observation
            .td_instantaneous_rate_bps
            .detach()
            .clone()
        ),

        rank=(
            observation
            .rank
            .detach()
            .clone()
        ),

        dl_buffer=(
            observation
            .dl_buffer
            .detach()
            .clone()
        ),

        wideband_cqi=(
            observation
            .wideband_cqi
            .detach()
            .clone()
        ),

        subband_cqi=(
            observation
            .subband_cqi
            .detach()
            .clone()
        ),

        precoder_directions=(
            observation
            .precoder_directions
            .detach()
            .clone()
        ),
    )


def _choose(
    *,
    field: str,
    current,
    delayed,
    stale_fields: frozenset[str],
):
    if field in stale_fields:
        return delayed

    return current


class SelectiveDelayedCSIInputProvider:

    def __init__(
        self,
        *,
        base_provider: MultiCellPPOInputProvider,
        config: SelectiveCSIStalenessConfig,
    ) -> None:

        self.base_provider = (
            base_provider
        )

        self.config = config

        self.stale_fields = (
            stale_fields_for_mode(
                config.mode
            )
        )

        self._history_by_stream: dict[
            int,
            deque[
                OneLDSCellTTIObservation
            ],
        ] = {}

        self._last_tti_by_stream: dict[
            int,
            int,
        ] = {}

        self._last_output_by_stream: dict[
            int,
            PPOTrainingTTIInputs,
        ] = {}


    def __call__(
        self,
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:

        previous_tti = (
            self
            ._last_tti_by_stream
            .get(
                stream_index
            )
        )

        if (
            previous_tti is not None
            and tti_index < previous_tti
        ):
            raise ValueError(
                "Selective CSI provider cannot "
                "move backward in TTI time."
            )

        if (
            previous_tti is not None
            and tti_index == previous_tti
        ):
            return (
                self
                ._last_output_by_stream[
                    stream_index
                ]
            )

        current_inputs = (
            self.base_provider(
                tti_index,
                stream_index,
            )
        )

        current = (
            current_inputs.observation
        )

        history = (
            self
            ._history_by_stream
            .setdefault(
                stream_index,
                deque(
                    maxlen=(
                        self.config.delay_ttis
                        + 1
                    )
                ),
            )
        )

        history.append(
            _clone_observation(
                current
            )
        )

        if self.config.delay_ttis == 0:
            delayed = history[-1]

        elif len(history) <= (
            self.config.delay_ttis
        ):
            delayed = history[0]

        else:
            delayed = history[
                -(
                    self.config.delay_ttis
                    + 1
                )
            ]

        if not (
            current
            .serving_global_ue_indices
            .equal(
                delayed
                .serving_global_ue_indices
            )
        ):
            raise RuntimeError(
                "Serving UE identity changed "
                "across CSI history."
            )

        stressed = (
            OneLDSCellTTIObservation(
                serving_global_ue_indices=(
                    current
                    .serving_global_ue_indices
                ),

                serving_ue_valid_mask=(
                    current
                    .serving_ue_valid_mask
                ),

                td_instantaneous_rate_bps=(
                    _choose(
                        field="td_rate",
                        current=(
                            current
                            .td_instantaneous_rate_bps
                        ),
                        delayed=(
                            delayed
                            .td_instantaneous_rate_bps
                        ),
                        stale_fields=(
                            self.stale_fields
                        ),
                    )
                ),

                rank=(
                    _choose(
                        field="rank",
                        current=current.rank,
                        delayed=delayed.rank,
                        stale_fields=(
                            self.stale_fields
                        ),
                    )
                ),

                #
                # Traffic information remains current.
                #
                dl_buffer=(
                    current.dl_buffer
                ),

                wideband_cqi=(
                    _choose(
                        field="cqi",
                        current=(
                            current.wideband_cqi
                        ),
                        delayed=(
                            delayed.wideband_cqi
                        ),
                        stale_fields=(
                            self.stale_fields
                        ),
                    )
                ),

                subband_cqi=(
                    _choose(
                        field="cqi",
                        current=(
                            current.subband_cqi
                        ),
                        delayed=(
                            delayed.subband_cqi
                        ),
                        stale_fields=(
                            self.stale_fields
                        ),
                    )
                ),

                precoder_directions=(
                    _choose(
                        field="precoder",
                        current=(
                            current
                            .precoder_directions
                        ),
                        delayed=(
                            delayed
                            .precoder_directions
                        ),
                        stale_fields=(
                            self.stale_fields
                        ),
                    )
                ),
            )
        )

        output = PPOTrainingTTIInputs(
            observation=stressed,

            #
            # Always CURRENT physical truth.
            #
            physical_inputs_builder=(
                current_inputs
                .physical_inputs_builder
            ),

            packet_arrivals=(
                current_inputs
                .packet_arrivals
            ),
        )

        self._last_tti_by_stream[
            stream_index
        ] = tti_index

        self._last_output_by_stream[
            stream_index
        ] = output

        return output
