from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)

from oran_scheduler.rl.ppo_multicell_rollout import (
    OneLDSPPOMultiCellRolloutController,
)

from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
)

from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)

from oran_scheduler.rl.ppo_traffic_cell_step import (
    PPOExecutionScheduleTransform,
    PPOTrafficRewardPopulation,
    TrafficAwarePPOCellTTIStepResult,
    run_traffic_aware_ppo_cell_tti_step,
)

from oran_scheduler.simulator.dynamic_handover_coordinator import (
    DynamicHandoverCoordinator,
    DynamicHandoverTransition,
)

from oran_scheduler.simulator.global_ue_state import (
    GlobalUEStateSnapshot,
)

from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)

from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)


DynamicRadioInputProvider = Callable[
    [
        int,
        int,
    ],
    PPOTrainingTTIInputs,
]


HandoverLinkPowerProvider = Callable[
    [
        int,
    ],
    torch.Tensor,
]


DynamicHandoverEvalObserver = Callable[
    [
        int,
        int,
        TrafficAwarePPOCellTTIStepResult,
        DynamicHandoverTransition,
    ],
    None,
]


@dataclass(frozen=True)
class DynamicHandoverEvalTTIResult:

    tti_index: int

    handover_transition: (
        DynamicHandoverTransition
    )

    cell_results: tuple[
        TrafficAwarePPOCellTTIStepResult,
        ...,
    ]


@dataclass(frozen=True)
class DynamicHandoverEvalRunResult:

    start_tti_index: int

    end_tti_index_exclusive: int

    num_ttis: int

    num_cells: int

    tti_results: tuple[
        DynamicHandoverEvalTTIResult,
        ...,
    ]

    final_global_state: GlobalUEStateSnapshot

    num_completed_handovers: int


def _validate_radio_membership(
    *,
    expected_global_ue_indices: torch.Tensor,
    inputs: PPOTrainingTTIInputs,
) -> None:

    actual = (
        inputs
        .observation
        .serving_global_ue_indices
    )

    if not torch.equal(
        actual,
        expected_global_ue_indices,
    ):
        raise RuntimeError(
            "Dynamic Sionna membership and "
            "cell-local scheduler membership "
            "disagree."
        )


def run_dynamic_multicell_ppo_evaluation(
    *,
    start_tti_index: int,

    num_ttis: int,

    radio_input_provider: (
        DynamicRadioInputProvider
    ),

    handover_link_power_provider: (
        HandoverLinkPowerProvider
    ),

    coordinator: DynamicHandoverCoordinator,

    rollout_controllers: tuple[
        OneLDSPPOMultiCellRolloutController,
        ...,
    ],

    tds_eligibility_config: (
        TDSBufferEligibilityConfig
    ),

    state_config: OneLDSStateConfig,

    greedy_config: PPOGreedySearchConfig,

    reward_config: PPORewardConfig,

    reward_population: (
        PPOTrafficRewardPopulation
    ),

    execution_schedule_transforms: (
        tuple[
            PPOExecutionScheduleTransform | None,
            ...,
        ]
        | None
    ) = None,

    observer: (
        DynamicHandoverEvalObserver | None
    ) = None,

    device: str | torch.device = "cuda:0",
) -> DynamicHandoverEvalRunResult:
    """
    Evaluation-only dynamic-association PPO runner.

    Per TTI:

        previous local state
            ->
        synchronize persistent UE state
            ->
        handover decision
            ->
        update serving association
            ->
        rematerialize local cell state
            ->
        dynamic Sionna population
            ->
        global identity-owned packet arrivals
            ->
        untracked PPO scheduling
            ->
        PHY service
            ->
        traffic service
            ->
        PF-history update

    No PPO rollout transition or optimizer update
    is created.
    """

    if start_tti_index < 0:
        raise ValueError(
            "start_tti_index must be "
            "non-negative."
        )

    if num_ttis <= 0:
        raise ValueError(
            "num_ttis must be positive."
        )

    num_cells = (
        coordinator.num_streams
    )

    if len(
        rollout_controllers
    ) != num_cells:
        raise ValueError(
            "rollout controller count does not "
            "match dynamic selected-cell count."
        )

    if (
        execution_schedule_transforms
        is None
    ):
        execution_schedule_transforms = (
            tuple(
                None
                for _ in range(
                    num_cells
                )
            )
        )

    if len(
        execution_schedule_transforms
    ) != num_cells:
        raise ValueError(
            "execution transform count does not "
            "match dynamic selected-cell count."
        )

    local_states = (
        coordinator
        .materialize_selected_cells()
    )

    tti_results: list[
        DynamicHandoverEvalTTIResult
    ] = []

    num_completed_handovers = 0

    for tti_index in range(
        start_tti_index,
        start_tti_index + num_ttis,
    ):

        link_power = (
            handover_link_power_provider(
                tti_index
            )
        )

        transition = (
            coordinator
            .advance_handover(
                tti_index=tti_index,

                link_power=link_power,

                local_states=(
                    local_states
                ),
            )
        )

        local_states = (
            transition.local_states
        )

        num_completed_handovers += int(
            transition
            .moved_global_ue_indices
            .numel()
        )

        cell_results = []

        for stream_index in range(
            num_cells
        ):

            inputs = (
                radio_input_provider(
                    tti_index,
                    stream_index,
                )
            )

            local_state = (
                local_states[
                    stream_index
                ]
            )

            _validate_radio_membership(
                expected_global_ue_indices=(
                    local_state
                    .global_ue_indices
                ),

                inputs=inputs,
            )

            packet_arrivals = (
                coordinator
                .packet_arrivals_for_stream(
                    tti_index=tti_index,

                    stream_index=(
                        stream_index
                    ),

                    local_states=(
                        local_states
                    ),
                )
            )

            result = (
                run_traffic_aware_ppo_cell_tti_step(
                    tti_index=tti_index,

                    collect_experience=False,

                    observation=(
                        inputs.observation
                    ),

                    traffic_manager=(
                        local_state
                        .traffic_manager
                    ),

                    tds_eligibility_config=(
                        tds_eligibility_config
                    ),

                    state_manager=(
                        local_state
                        .state_manager
                    ),

                    training_controller=(
                        rollout_controllers[
                            stream_index
                        ]
                    ),

                    state_config=(
                        state_config
                    ),

                    physical_inputs_builder=(
                        inputs
                        .physical_inputs_builder
                    ),

                    greedy_config=(
                        greedy_config
                    ),

                    reward_config=(
                        reward_config
                    ),

                    reward_population=(
                        reward_population
                    ),

                    execution_schedule_transform=(
                        execution_schedule_transforms[
                            stream_index
                        ]
                    ),

                    packet_arrivals=(
                        packet_arrivals
                    ),

                    device=device,
                )
            )

            cell_results.append(
                result
            )

            if observer is not None:
                observer(
                    tti_index,
                    stream_index,
                    result,
                    transition,
                )

        tti_results.append(
            DynamicHandoverEvalTTIResult(
                tti_index=tti_index,

                handover_transition=(
                    transition
                ),

                cell_results=(
                    tuple(
                        cell_results
                    )
                ),
            )
        )

    coordinator.synchronize_selected_cells(
        local_states
    )

    return DynamicHandoverEvalRunResult(
        start_tti_index=(
            start_tti_index
        ),

        end_tti_index_exclusive=(
            start_tti_index
            + num_ttis
        ),

        num_ttis=num_ttis,

        num_cells=num_cells,

        tti_results=(
            tuple(
                tti_results
            )
        ),

        final_global_state=(
            coordinator
            .registry
            .snapshot()
        ),

        num_completed_handovers=(
            num_completed_handovers
        ),
    )
