from __future__ import annotations

from dataclasses import dataclass

import torch

from oran_scheduler.rl.ppo_physical_tti import (
    PPOPhysicalTTIOutcome,
)
from oran_scheduler.rl.ppo_traffic_cell_step import (
    PreparedTrafficAwarePPOCellTTI,
)
from oran_scheduler.schedulers.classical_runtime_backend import (
    ClassicalRuntimeScheduleResult,
    ClassicalSchedulerMode,
    run_classical_runtime_scheduler,
)
from oran_scheduler.schedulers.throughput_history import (
    CellThroughputHistoryUpdate,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
    OneLDSCellTTIStateManager,
)
from oran_scheduler.simulator.tds_eligibility import (
    TDSEligibilityData,
)
from oran_scheduler.simulator.traffic import (
    TrafficBufferManager,
    TrafficServiceResult,
    TrafficTTIStart,
)


@dataclass(frozen=True)
class TrafficAwareClassicalCellTTIResult:
    """
    Traffic-aware Baseline/PF-Greedy result.

    This intentionally exposes the same core attributes
    consumed by:

        PPOTrainingMetricsObserver
        RuntimeServiceMetrics

    so network metrics are scheduler-independent.
    """

    traffic_start: TrafficTTIStart

    scheduler_observation: (
        OneLDSCellTTIObservation
    )

    tds_eligibility: TDSEligibilityData

    prepared: object

    schedule: ClassicalRuntimeScheduleResult

    physical_outcome: PPOPhysicalTTIOutcome

    serving_offered_capacity_bps: torch.Tensor

    traffic_service: TrafficServiceResult

    candidate_delivered_rate_bps: torch.Tensor

    #
    # Classical schedulers do not have a PPO reward.
    #
    reward: None

    expert_labels: None

    num_expert_demonstrations_added: int

    history_update: CellThroughputHistoryUpdate


def scatter_candidate_values_to_serving(
    *,
    candidate_values: torch.Tensor,
    candidate_serving_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    num_serving_ues: int,
) -> torch.Tensor:
    """
    Candidate order -> persistent serving-UE order.

    Serving UEs not present in the current PF-TDS
    shortlist receive zero service.
    """

    if candidate_values.ndim != 1:
        raise ValueError(
            "candidate_values must have shape "
            "[candidate]."
        )

    if tuple(
        candidate_serving_indices.shape
    ) != tuple(
        candidate_values.shape
    ):
        raise ValueError(
            "candidate_serving_indices shape mismatch."
        )

    if tuple(
        candidate_valid_mask.shape
    ) != tuple(
        candidate_values.shape
    ):
        raise ValueError(
            "candidate_valid_mask shape mismatch."
        )

    if candidate_valid_mask.dtype != torch.bool:
        raise ValueError(
            "candidate_valid_mask must use bool."
        )

    device = candidate_values.device

    if (
        candidate_serving_indices.device != device
        or candidate_valid_mask.device != device
    ):
        raise ValueError(
            "Candidate tensors must share one device."
        )

    output = torch.zeros(
        num_serving_ues,
        dtype=candidate_values.dtype,
        device=device,
    )

    valid_indices = (
        candidate_serving_indices[
            candidate_valid_mask
        ]
    )

    valid_values = (
        candidate_values[
            candidate_valid_mask
        ]
    )

    if valid_indices.numel() > 0:

        if torch.any(
            valid_indices < 0
        ):
            raise ValueError(
                "Valid candidate serving indices "
                "cannot be negative."
            )

        if torch.any(
            valid_indices >= num_serving_ues
        ):
            raise ValueError(
                "Candidate serving index exceeds "
                "serving layout."
            )

        if (
            torch.unique(
                valid_indices
            ).numel()
            != valid_indices.numel()
        ):
            raise ValueError(
                "Candidate serving indices must be unique."
            )

        output[
            valid_indices
        ] = valid_values

    return output


def gather_serving_values_to_candidates(
    *,
    serving_values: torch.Tensor,
    candidate_serving_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> torch.Tensor:

    if serving_values.ndim != 1:
        raise ValueError(
            "serving_values must have shape "
            "[serving_ue]."
        )

    gathered = serving_values[
        candidate_serving_indices
    ]

    return torch.where(
        candidate_valid_mask,
        gathered,
        torch.zeros_like(
            gathered
        ),
    )


def run_prepared_classical_traffic_tti(
    *,
    mode: ClassicalSchedulerMode,
    preparation: PreparedTrafficAwarePPOCellTTI,
    traffic_manager: TrafficBufferManager,
    state_manager: OneLDSCellTTIStateManager,
    num_user_slots: int,
) -> TrafficAwareClassicalCellTTIResult:
    """
    Execute one already-prepared classical TTI.

    Preparation has already done:

        arrivals
        queue-visible state
        TDS eligibility
        PF-TDS
        candidate PHY input construction

    This function performs:

        Baseline/PF-Greedy SDS
            ->
        physical offered capacity
            ->
        actual queue-limited delivery
            ->
        PF-history update
    """

    prepared = (
        preparation.prepared
    )

    classical = (
        run_classical_runtime_scheduler(
            mode=mode,
            prepared=prepared,
            physical_inputs=(
                preparation
                .physical_inputs
            ),
            num_user_slots=(
                num_user_slots
            ),
        )
    )

    physical_outcome = (
        classical
        .physical_outcome
    )

    num_serving_ues = int(
        preparation
        .scheduler_observation
        .serving_global_ue_indices
        .numel()
    )

    serving_capacity = (
        scatter_candidate_values_to_serving(
            candidate_values=(
                physical_outcome
                .candidate_total_target_compliant_rate_bps
            ),

            candidate_serving_indices=(
                prepared
                .candidate_serving_indices
            ),

            candidate_valid_mask=(
                prepared
                .candidate_valid_mask
            ),

            num_serving_ues=(
                num_serving_ues
            ),
        )
    )

    traffic_service = (
        traffic_manager.apply_service(
            offered_service_capacity_bps=(
                serving_capacity
            )
        )
    )

    candidate_delivered = (
        gather_serving_values_to_candidates(
            serving_values=(
                traffic_service
                .delivered_rate_bps
            ),

            candidate_serving_indices=(
                prepared
                .candidate_serving_indices
            ),

            candidate_valid_mask=(
                prepared
                .candidate_valid_mask
            ),
        )
    )

    history_update = (
        state_manager.complete_tti(
            candidate_delivered_rate_bps=(
                candidate_delivered
            )
        )
    )

    return TrafficAwareClassicalCellTTIResult(
        traffic_start=(
            preparation
            .traffic_start
        ),

        scheduler_observation=(
            preparation
            .scheduler_observation
        ),

        tds_eligibility=(
            preparation
            .tds_eligibility
        ),

        prepared=prepared,

        schedule=classical,

        physical_outcome=(
            physical_outcome
        ),

        serving_offered_capacity_bps=(
            serving_capacity
        ),

        traffic_service=(
            traffic_service
        ),

        candidate_delivered_rate_bps=(
            candidate_delivered
        ),

        reward=None,

        expert_labels=None,

        num_expert_demonstrations_added=0,

        history_update=(
            history_update
        ),
    )
