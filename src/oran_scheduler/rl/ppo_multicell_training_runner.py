from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
)
from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)
from oran_scheduler.rl.ppo_multicell_rollout import (
    OneLDSPPOMultiCellRolloutController,
)
from oran_scheduler.rl.ppo_multistream_training import (
    PPOMultiStreamTrainingCoordinator,
)
from oran_scheduler.rl.ppo_pf_expert import (
    PPOPFExpertConfig,
)
from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
    PPORewardReduction,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingRunnerConfig,
    PPOTrainingTTIInputs,
)
from oran_scheduler.rl.ppo_traffic_cell_step import (
    PPOTrafficRewardPopulation,
    TrafficAwarePPOCellTTIStepResult,
    prepare_traffic_aware_ppo_cell_tti,
    run_traffic_aware_ppo_cell_tti_step,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizers,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIStateManager,
)
from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)
from oran_scheduler.simulator.traffic import (
    TrafficBufferManager,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)


PPOMultiCellTrainingTTIInputProvider = Callable[
    [
        int,
        int,
    ],
    PPOTrainingTTIInputs,
]

PPOMultiCellResultObserver = Callable[
    [
        int,
        int,
        TrafficAwarePPOCellTTIStepResult,
    ],
    None,
]


@dataclass(frozen=True)
class PPOMultiCellTrainingRunResult:
    """
    Compact diagnostics for one synchronized
    multi-cell training invocation.
    """

    start_tti_index: int

    end_tti_index_exclusive: int

    num_ttis: int

    num_cells: int

    num_warmup_ttis: int

    num_collection_ttis: int

    first_collected_tti_index: (
        int | None
    )

    num_ppo_updates: int

    num_expert_guidance_updates: int

    num_expert_demonstrations_added: int

    final_transition_buffer_size: int

    unresolved_boundary_by_cell: tuple[
        bool,
        ...
    ]


def run_multicell_ppo_training(
    *,
    start_tti_index: int,
    num_ttis: int,
    input_provider: (
        PPOMultiCellTrainingTTIInputProvider
    ),
    traffic_managers: tuple[
        TrafficBufferManager,
        ...,
    ],
    state_managers: tuple[
        OneLDSCellTTIStateManager,
        ...,
    ],
    rollout_controllers: tuple[
        OneLDSPPOMultiCellRolloutController,
        ...,
    ],
    actor: OneLDSPPOActor,
    critic: nn.Module,
    optimizers: PPOOptimizers,
    centralized_training: (
        PPOMultiStreamTrainingCoordinator
    ),
    tds_eligibility_config: (
        TDSBufferEligibilityConfig
    ),
    state_config: OneLDSStateConfig,
    greedy_config: PPOGreedySearchConfig,
    reward_config: PPORewardConfig,
    reward_population: (
        PPOTrafficRewardPopulation
    ),
    runner_config: PPOTrainingRunnerConfig,
    expert_buffer: (
        PPOExpertDemonstrationBuffer | None
    ) = None,
    pf_expert_config: (
        PPOPFExpertConfig | None
    ) = None,
    expert_candidate_augmentation_config: (
        PPOCandidateAugmentationConfig | None
    ) = None,
    expert_candidate_augmentation_generator: (
        torch.Generator | None
    ) = None,
    reward_reduction: (
        PPORewardReduction
    ) = "mean",
    cell_result_observer: (
        PPOMultiCellResultObserver | None
    ) = None,
    device: str | torch.device = "cuda:0",
) -> PPOMultiCellTrainingRunResult:
    """
    Run synchronized centralized multi-cell PPO.

    CRITICAL TTI ORDER:

        1. prepare every cell through its slot-0 state

        2. resolve every previous-TTI boundary

        3. perform centralized PPO/JSD update if ready

        4. only then sample any current-TTI action

        5. execute PHY/traffic/reward for every cell
    """

    if start_tti_index < 0:
        raise ValueError(
            "start_tti_index must be non-negative."
        )

    if num_ttis <= 0:
        raise ValueError(
            "num_ttis must be positive."
        )

    num_cells = (
        centralized_training
        .transition_buffer
        .config
        .num_streams
    )

    if len(
        traffic_managers
    ) != num_cells:
        raise ValueError(
            "traffic_managers must contain one "
            "entry per centralized stream."
        )

    if len(
        state_managers
    ) != num_cells:
        raise ValueError(
            "state_managers must contain one entry "
            "per centralized stream."
        )

    if len(
        rollout_controllers
    ) != num_cells:
        raise ValueError(
            "rollout_controllers must contain one "
            "entry per centralized stream."
        )

    shared_transition_buffer = (
        centralized_training
        .transition_buffer
    )

    for cell_index, controller in enumerate(
        rollout_controllers
    ):
        if controller.stream_id != cell_index:
            raise ValueError(
                "rollout controller stream_id must "
                "equal its cell index."
            )

        if (
            controller.transition_buffer
            is not shared_transition_buffer
        ):
            raise ValueError(
                "Every cell rollout controller must "
                "use the centralized transition "
                "buffer."
            )

        if controller.actor is not actor:
            raise ValueError(
                "Every cell must use the same shared "
                "actor object."
            )

        if controller.critic is not critic:
            raise ValueError(
                "Every cell must use the same shared "
                "critic object."
            )

    expert_coordinator = (
        centralized_training
        .expert_guidance_coordinator
    )

    if expert_coordinator is not None:
        if expert_buffer is None:
            raise ValueError(
                "Centralized expert guidance is "
                "enabled but expert_buffer is None."
            )

        if (
            expert_coordinator.expert_buffer
            is not expert_buffer
        ):
            raise ValueError(
                "PF expert collection and JSD "
                "training must use the same expert "
                "buffer object."
            )

        if pf_expert_config is None:
            raise ValueError(
                "pf_expert_config is required when "
                "centralized expert guidance is "
                "enabled."
            )

    updates_before = (
        centralized_training.num_updates
    )

    expert_updates_before = (
        centralized_training
        .num_expert_guidance_updates
    )

    num_warmup_ttis = 0

    num_collection_ttis = 0

    first_collected_tti_index: (
        int | None
    ) = None

    num_expert_demonstrations_added = 0

    for offset in range(
        num_ttis
    ):
        tti_index = (
            start_tti_index
            + offset
        )

        collect_experience = (
            tti_index
            >= (
                runner_config
                .first_collection_tti_index
            )
        )

        if collect_experience:
            num_collection_ttis += 1

            if (
                first_collected_tti_index
                is None
            ):
                first_collected_tti_index = (
                    tti_index
                )

        else:
            num_warmup_ttis += 1

            if len(
                shared_transition_buffer
            ) != 0:
                raise RuntimeError(
                    "Cannot enter warm-up while "
                    "centralized on-policy samples "
                    "already exist."
                )

            if any(
                controller
                .has_unresolved_tti_boundary
                for controller
                in rollout_controllers
            ):
                raise RuntimeError(
                    "Cannot enter warm-up while a "
                    "tracked cell trajectory is "
                    "unresolved."
                )

        #
        # ------------------------------------------------------
        # PHASE A:
        #
        # Prepare ALL cells before any actor action.
        # ------------------------------------------------------
        #
        preparations = []

        for cell_index in range(
            num_cells
        ):
            tti_inputs = input_provider(
                tti_index,
                cell_index,
            )

            controller = (
                rollout_controllers[
                    cell_index
                ]
            )

            preparation = (
                prepare_traffic_aware_ppo_cell_tti(
                    tti_index=tti_index,
                    observation=(
                        tti_inputs
                        .observation
                    ),
                    traffic_manager=(
                        traffic_managers[
                            cell_index
                        ]
                    ),
                    tds_eligibility_config=(
                        tds_eligibility_config
                    ),
                    state_manager=(
                        state_managers[
                            cell_index
                        ]
                    ),
                    state_config=state_config,
                    physical_inputs_builder=(
                        tti_inputs
                        .physical_inputs_builder
                    ),
                    num_user_slots=(
                        controller
                        .config
                        .num_user_slots
                    ),
                    packet_arrivals=(
                        tti_inputs
                        .packet_arrivals
                    ),
                    device=device,
                )
            )

            preparations.append(
                (
                    preparation,
                    tti_inputs,
                )
            )

        #
        # ------------------------------------------------------
        # PHASE B:
        #
        # Resolve ALL previous-TTI cell boundaries.
        # ------------------------------------------------------
        #
        if collect_experience:
            for cell_index in range(
                num_cells
            ):
                preparation, _ = (
                    preparations[
                        cell_index
                    ]
                )

                rollout_controllers[
                    cell_index
                ].begin_tti(
                    tti_index=tti_index,
                    first_state=(
                        preparation
                        .first_decision
                        .state_data
                        .state
                    ),
                )

            #
            # Every cell's old boundary has now been
            # resolved.
            #
            # It is finally safe to update the shared
            # policy.
            #
            centralized_training.update_if_ready(
                actor=actor,
                critic=critic,
                optimizers=optimizers,
            )

        #
        # ------------------------------------------------------
        # PHASE C:
        #
        # All cells now sample from the SAME current
        # policy for this TTI.
        # ------------------------------------------------------
        #
        for cell_index in range(
            num_cells
        ):
            (
                preparation,
                tti_inputs,
            ) = preparations[
                cell_index
            ]

            cell_result = (
                run_traffic_aware_ppo_cell_tti_step(
                    tti_index=tti_index,
                    collect_experience=(
                        collect_experience
                    ),
                    observation=(
                        tti_inputs
                        .observation
                    ),
                    traffic_manager=(
                        traffic_managers[
                            cell_index
                        ]
                    ),
                    tds_eligibility_config=(
                        tds_eligibility_config
                    ),
                    state_manager=(
                        state_managers[
                            cell_index
                        ]
                    ),
                    training_controller=(
                        rollout_controllers[
                            cell_index
                        ]
                    ),
                    state_config=state_config,
                    physical_inputs_builder=(
                        tti_inputs
                        .physical_inputs_builder
                    ),
                    greedy_config=greedy_config,
                    reward_config=reward_config,
                    reward_population=(
                        reward_population
                    ),
                    expert_buffer=(
                        expert_buffer
                    ),
                    pf_expert_config=(
                        pf_expert_config
                    ),
                    candidate_augmentation_config=(
                        expert_candidate_augmentation_config
                    ),
                    candidate_augmentation_generator=(
                        expert_candidate_augmentation_generator
                    ),
                    preparation=(
                        preparation
                    ),
                    reward_reduction=(
                        reward_reduction
                    ),
                    packet_arrivals=(
                        tti_inputs
                        .packet_arrivals
                    ),
                    device=device,
                )
            )

            num_expert_demonstrations_added += (
                cell_result
                .num_expert_demonstrations_added
            )

            if cell_result_observer is not None:
                cell_result_observer(
                    tti_index,
                    cell_index,
                    cell_result,
                )

    updates_after = (
        centralized_training.num_updates
    )

    expert_updates_after = (
        centralized_training
        .num_expert_guidance_updates
    )

    return PPOMultiCellTrainingRunResult(
        start_tti_index=(
            start_tti_index
        ),
        end_tti_index_exclusive=(
            start_tti_index
            + num_ttis
        ),
        num_ttis=num_ttis,
        num_cells=num_cells,
        num_warmup_ttis=(
            num_warmup_ttis
        ),
        num_collection_ttis=(
            num_collection_ttis
        ),
        first_collected_tti_index=(
            first_collected_tti_index
        ),
        num_ppo_updates=(
            updates_after
            - updates_before
        ),
        num_expert_guidance_updates=(
            expert_updates_after
            - expert_updates_before
        ),
        num_expert_demonstrations_added=(
            num_expert_demonstrations_added
        ),
        final_transition_buffer_size=len(
            shared_transition_buffer
        ),
        unresolved_boundary_by_cell=tuple(
            controller
            .has_unresolved_tti_boundary
            for controller
            in rollout_controllers
        ),
    )


