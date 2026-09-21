from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import torch

from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
)


MultiCellPPOInputProvider = Callable[
    [
        int,
        int,
    ],
    PPOTrainingTTIInputs,
]


@dataclass(frozen=True)
class CSIStalenessConfig:
    """
    Scheduler-side CSI reporting delay.

    delay_ttis = d means:

        scheduler at TTI t
            sees channel-derived reports from t-d

    while:

        actual PHY evaluation at TTI t
            still uses the current TTI-t channel.

    Only channel-derived scheduler reports are stale.

    Current traffic/buffer state is intentionally
    NOT delayed.
    """

    delay_ttis: int = 0

    def __post_init__(
        self,
    ) -> None:
        if self.delay_ttis < 0:
            raise ValueError(
                "delay_ttis must be non-negative."
            )


def _clone_observation(
    observation: OneLDSCellTTIObservation,
) -> OneLDSCellTTIObservation:
    """
    Detach one scheduler observation so it can safely
    survive in the delay history.
    """

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


def _build_stale_csi_observation(
    *,
    current: OneLDSCellTTIObservation,
    delayed: OneLDSCellTTIObservation,
) -> OneLDSCellTTIObservation:
    """
    Combine:

        CURRENT
            UE identity/layout
            traffic placeholder

        DELAYED
            PF-TDS instantaneous-rate report
            RI/rank
            wideband CQI
            sub-band CQI
            precoder directions / PMI surrogate

    This creates scheduler-side stale CSI while
    leaving the actual physical channel untouched.
    """

    if not torch.equal(
        current.serving_global_ue_indices,
        delayed.serving_global_ue_indices,
    ):
        raise RuntimeError(
            "Serving UE identity changed across "
            "the CSI-delay history."
        )

    if not torch.equal(
        current.serving_ue_valid_mask,
        delayed.serving_ue_valid_mask,
    ):
        raise RuntimeError(
            "Serving UE validity layout changed "
            "across the CSI-delay history."
        )

    return OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            current.serving_global_ue_indices
        ),

        serving_ue_valid_mask=(
            current.serving_ue_valid_mask
        ),

        #
        # This is channel-derived in the current
        # reproduction and therefore becomes stale.
        #
        td_instantaneous_rate_bps=(
            delayed
            .td_instantaneous_rate_bps
        ),

        #
        # CSI reports.
        #
        rank=delayed.rank,

        #
        # Traffic remains current.
        # The traffic-aware PPO step later replaces
        # this placeholder with the real current
        # buffer state.
        #
        dl_buffer=current.dl_buffer,

        wideband_cqi=(
            delayed.wideband_cqi
        ),

        subband_cqi=(
            delayed.subband_cqi
        ),

        precoder_directions=(
            delayed.precoder_directions
        ),
    )


class DelayedCSIInputProvider:
    """
    Wrap a real multi-cell PPO input provider with
    scheduler-side CSI delay.

    Crucial separation:

        observation
            can be stale

        physical_inputs_builder
            always comes from CURRENT TTI

    Therefore an allocation selected using stale
    reports is judged by the real current PHY.
    """

    def __init__(
        self,
        *,
        base_provider: MultiCellPPOInputProvider,
        config: CSIStalenessConfig,
    ) -> None:
        self.base_provider = (
            base_provider
        )

        self.config = config

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
                "Delayed CSI provider cannot move "
                "backward in TTI time."
            )

        #
        # Preserve provider idempotence if the same
        # stream is requested twice in one TTI.
        #
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
                current_inputs.observation
            )
        )

        if self.config.delay_ttis == 0:
            delayed_observation = (
                history[-1]
            )

        elif len(history) <= (
            self.config.delay_ttis
        ):
            #
            # Startup only.
            #
            # Until t-d exists, use the oldest
            # available report.
            #
            # Stress-test metrics should discard
            # this startup interval.
            #
            delayed_observation = (
                history[0]
            )

        else:
            delayed_observation = (
                history[
                    -(
                        self.config.delay_ttis
                        + 1
                    )
                ]
            )

        stressed_observation = (
            _build_stale_csi_observation(
                current=(
                    current_inputs
                    .observation
                ),
                delayed=(
                    delayed_observation
                ),
            )
        )

        output = PPOTrainingTTIInputs(
            observation=(
                stressed_observation
            ),

            #
            # IMPORTANT:
            # current TTI physical truth.
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
