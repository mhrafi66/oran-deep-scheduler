from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
)


CandidateInterventionMode = Literal[
    "native",
    "fresh_candidates",
    "fresh_features",
    "fresh_both",
]


InputProvider = Callable[
    [
        int,
        int,
    ],
    PPOTrainingTTIInputs,
]


@dataclass(frozen=True)
class CandidateInterventionConfig:
    """
    Evaluation-only causal intervention.

    The scheduler observation is decomposed into:

    1. PF/TDS candidate-selection information
       represented by td_instantaneous_rate_bps;

    2. downstream PPO radio features:
       - rank
       - wideband CQI
       - subband CQI
       - precoder directions

    Modes
    -----
    native:
        Stressed candidates + stressed PPO features.

    fresh_candidates:
        Fresh PF/TDS rate information +
        stressed PPO radio features.

        This is the main candidate-rescue experiment.

    fresh_features:
        Stressed PF/TDS rate information +
        fresh PPO radio features.

        This isolates downstream PPO-feature rescue.

    fresh_both:
        Fresh PF/TDS rate information +
        fresh PPO radio features.

        This is the full scheduler-observation rescue.

    Traffic buffers are not manipulated here.
    The real traffic manager supplies current queue state later.

    Physical evaluation is never modified.
    """

    mode: CandidateInterventionMode = "native"

    def __post_init__(
        self,
    ) -> None:
        if self.mode not in (
            "native",
            "fresh_candidates",
            "fresh_features",
            "fresh_both",
        ):
            raise ValueError(
                "Unsupported candidate intervention mode: "
                f"{self.mode}."
            )


def _validate_observation_pair(
    *,
    fresh: OneLDSCellTTIObservation,
    stressed: OneLDSCellTTIObservation,
) -> None:
    """
    Fresh and stressed observations must represent
    the same serving-UE layout.
    """

    if not torch.equal(
        fresh.serving_global_ue_indices,
        stressed.serving_global_ue_indices,
    ):
        raise ValueError(
            "Fresh and stressed observations have "
            "different serving UE identities."
        )

    if not torch.equal(
        fresh.serving_ue_valid_mask,
        stressed.serving_ue_valid_mask,
    ):
        raise ValueError(
            "Fresh and stressed observations have "
            "different serving UE validity masks."
        )

    fresh_device = (
        fresh
        .serving_global_ue_indices
        .device
    )

    tensors = (
        stressed.serving_global_ue_indices,
        fresh.td_instantaneous_rate_bps,
        stressed.td_instantaneous_rate_bps,
        fresh.rank,
        stressed.rank,
        fresh.wideband_cqi,
        stressed.wideband_cqi,
        fresh.subband_cqi,
        stressed.subband_cqi,
        fresh.precoder_directions,
        stressed.precoder_directions,
    )

    if any(
        tensor.device != fresh_device
        for tensor
        in tensors
    ):
        raise ValueError(
            "Fresh and stressed observations must "
            "be on the same device."
        )


def build_candidate_intervention_observation(
    *,
    fresh: OneLDSCellTTIObservation,
    stressed: OneLDSCellTTIObservation,
    mode: CandidateInterventionMode,
) -> OneLDSCellTTIObservation:
    """
    Construct the scheduler observation for one
    causal-rescue condition.

    Important decomposition:

        PF/TDS candidate source
            <- td_instantaneous_rate_bps

        PPO radio-feature source
            <- rank
            <- wideband_cqi
            <- subband_cqi
            <- precoder_directions

    The DL buffer is kept from the stressed/current scheduler
    input, but the traffic-aware preparation stage will later
    replace it using the real TrafficBufferManager anyway.
    """

    _validate_observation_pair(
        fresh=fresh,
        stressed=stressed,
    )

    if mode not in (
        "native",
        "fresh_candidates",
        "fresh_features",
        "fresh_both",
    ):
        raise ValueError(
            "Unsupported candidate intervention mode: "
            f"{mode}."
        )

    use_fresh_candidates = mode in (
        "fresh_candidates",
        "fresh_both",
    )

    use_fresh_features = mode in (
        "fresh_features",
        "fresh_both",
    )

    if use_fresh_candidates:
        td_rate = (
            fresh
            .td_instantaneous_rate_bps
        )
    else:
        td_rate = (
            stressed
            .td_instantaneous_rate_bps
        )

    if use_fresh_features:
        feature_source = fresh
    else:
        feature_source = stressed

    return OneLDSCellTTIObservation(
        serving_global_ue_indices=(
            stressed
            .serving_global_ue_indices
        ),

        serving_ue_valid_mask=(
            stressed
            .serving_ue_valid_mask
        ),

        td_instantaneous_rate_bps=(
            td_rate
        ),

        rank=(
            feature_source.rank
        ),

        #
        # Traffic state is not part of this intervention.
        #
        # prepare_traffic_aware_ppo_cell_tti() later replaces
        # this field using the real queue state.
        #
        dl_buffer=(
            stressed.dl_buffer
        ),

        wideband_cqi=(
            feature_source
            .wideband_cqi
        ),

        subband_cqi=(
            feature_source
            .subband_cqi
        ),

        precoder_directions=(
            feature_source
            .precoder_directions
        ),
    )


class CandidateInterventionInputProvider:
    """
    Compose fresh/current and stressed scheduler observations.

    The stressed provider is called FIRST because it may contain
    temporal wrappers whose internal history must advance in the
    normal experiment order.

    The fresh provider is then queried for the same TTI/stream.
    In the real Sionna evaluator this provider is cached, so this
    obtains current radio truth without rebuilding the channel.

    Physical truth and packet arrivals always remain those from
    the stressed experiment path.
    """

    def __init__(
        self,
        *,
        fresh_provider: InputProvider,
        stressed_provider: InputProvider,
        config: CandidateInterventionConfig,
    ) -> None:
        self.fresh_provider = (
            fresh_provider
        )

        self.stressed_provider = (
            stressed_provider
        )

        self.config = config

        self._last_key: (
            tuple[int, int] | None
        ) = None

        self._last_output: (
            PPOTrainingTTIInputs | None
        ) = None


    def __call__(
        self,
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:

        key = (
            tti_index,
            stream_index,
        )

        if (
            self._last_key == key
            and self._last_output
            is not None
        ):
            return self._last_output

        #
        # IMPORTANT:
        # advance the real stressed wrapper chain first.
        #
        stressed_inputs = (
            self.stressed_provider(
                tti_index,
                stream_index,
            )
        )

        #
        # Obtain the cached current/fresh observation.
        #
        fresh_inputs = (
            self.fresh_provider(
                tti_index,
                stream_index,
            )
        )

        observation = (
            build_candidate_intervention_observation(
                fresh=(
                    fresh_inputs
                    .observation
                ),
                stressed=(
                    stressed_inputs
                    .observation
                ),
                mode=(
                    self.config.mode
                ),
            )
        )

        #
        # Keep ACTUAL CURRENT physical truth and the exact
        # packet-arrival realization belonging to this run.
        #
        output = PPOTrainingTTIInputs(
            observation=observation,

            physical_inputs_builder=(
                stressed_inputs
                .physical_inputs_builder
            ),

            packet_arrivals=(
                stressed_inputs
                .packet_arrivals
            ),
        )

        self._last_key = key
        self._last_output = output

        return output
