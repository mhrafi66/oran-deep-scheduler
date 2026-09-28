from collections.abc import Callable
from dataclasses import dataclass

import torch

from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)
from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
    PPORewardReduction,
)
from oran_scheduler.rl.ppo_single_cell_step import (
    PPOPhysicalInputsBuilder,
)
from oran_scheduler.rl.ppo_traffic_cell_step import (
    PPOTrafficRewardPopulation,
    run_traffic_aware_ppo_cell_tti_step,
)
from oran_scheduler.rl.ppo_training_controller import (
    OneLDSPPOTrainingController,
)

from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
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
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_pf_expert import (
    PPOPFExpertConfig,
)

@dataclass(frozen=True)
class PPOTrainingTTIInputs:
    """
    Environment inputs for one training TTI.

    observation:
        Current scheduler-facing radio information.

    physical_inputs_builder:
        Builds candidate-specific physical inputs
        after PF-TDS selects the candidate set.

    packet_arrivals:
        Optional deterministic traffic trace.

        None means TrafficBufferManager samples its
        normal stochastic arrivals.
    """

    observation: OneLDSCellTTIObservation

    physical_inputs_builder: (
        PPOPhysicalInputsBuilder
    )

    packet_arrivals: (
        torch.Tensor | None
    ) = None

    #
    # Optional per-TTI scheduling gate.
    #
    # This is intentionally separate from
    # observation.serving_ue_valid_mask:
    #
    #     serving_ue_valid_mask
    #         = static association / padded layout
    #
    #     tds_eligibility_override_mask
    #         = temporary schedulability
    #
    # The override may only REMOVE UEs from the
    # normal PF-TDS eligible population.
    #
    tds_eligibility_override_mask: (
        torch.Tensor | None
    ) = None


PPOTrainingTTIInputProvider = Callable[
    [
        int,
    ],
    PPOTrainingTTIInputs,
]


@dataclass(frozen=True)
class PPOTrainingRunnerConfig:
    """
    Temporal collection configuration.

    PAPER-SPECIFIED:
        Training samples begin only after the first
        100 TTIs.

    ZERO-BASED SIMULATOR CONVENTION:
        TTI indices:

            0 ... 99
                warm-up

            100
                first TTI where experience is
                collected

        Therefore:
            first_collection_tti_index = 100
    """

    first_collection_tti_index: int = 100

    def __post_init__(self) -> None:
        if self.first_collection_tti_index < 0:
            raise ValueError(
                "first_collection_tti_index cannot "
                "be negative."
            )


@dataclass(frozen=True)
class PPOTrainingRunResult:
    """
    Compact diagnostics for one runner invocation.

    No complete TTI tensors are retained here, so a
    long run does not accidentally keep every GPU
    computation/result in memory.
    """

    start_tti_index: int

    end_tti_index_exclusive: int

    num_ttis: int

    num_warmup_ttis: int

    num_collection_ttis: int

    first_collected_tti_index: (
        int | None
    )

    num_ppo_updates: int

    num_expert_guidance_updates: int

    final_agent_buffer_size: int

    has_unresolved_tti_boundary: bool


def run_single_cell_ppo_training(
    *,
    start_tti_index: int,
    num_ttis: int,
    input_provider: PPOTrainingTTIInputProvider,
    traffic_manager: TrafficBufferManager,
    tds_eligibility_config: (
        TDSBufferEligibilityConfig
    ),
    state_manager: OneLDSCellTTIStateManager,
    training_controller: (
        OneLDSPPOTrainingController
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
    candidate_augmentation_config: (
        PPOCandidateAugmentationConfig | None
    ) = None,
    candidate_augmentation_generator: (
        torch.Generator | None
    ) = None,
    reward_reduction: (
        PPORewardReduction
    ) = "mean",
    device: str | torch.device = "cuda:0",
) -> PPOTrainingRunResult:
    """
    Run temporally contiguous single-cell PPO
    training TTIs.

    Before first_collection_tti_index:
        environment evolves normally,
        but PPO experience is NOT collected.

    Starting at first_collection_tti_index:
        on-policy PPO trajectory collection begins.

    This runner intentionally does NOT mark the
    final TTI as terminal.

    Stopping the Python loop is NOT the same as an
    environment terminal state.

    Therefore the final collected layer may remain
    unresolved until training continues with the
    next TTI.
    """

    if start_tti_index < 0:
        raise ValueError(
            "start_tti_index must be non-negative."
        )

    if num_ttis <= 0:
        raise ValueError(
            "num_ttis must be positive."
        )

    expert_coordinator = (
        training_controller
        .expert_guidance_coordinator
    )

    if expert_coordinator is not None:
        if expert_buffer is None:
            raise ValueError(
                "The PPO controller has expert "
                "guidance enabled, but no expert "
                "buffer was supplied to the "
                "training runner."
            )

        if (
            expert_buffer
            is not
            expert_coordinator
            .expert_buffer
        ):
            raise ValueError(
                "The expert buffer used for PF "
                "label collection must be the same "
                "buffer used for expert-guidance "
                "updates."
            )

        if pf_expert_config is None:
            raise ValueError(
                "Expert guidance is enabled but "
                "pf_expert_config was not supplied."
            )


    updates_before = (
        training_controller.num_updates
    )

    expert_updates_before = (
        training_controller
        .num_expert_guidance_updates
    )

    num_warmup_ttis = 0

    num_collection_ttis = 0

    first_collected_tti_index: (
        int | None
    ) = None

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

        tti_inputs = input_provider(
            tti_index
        )

        run_traffic_aware_ppo_cell_tti_step(
            tti_index=tti_index,
            collect_experience=(
                collect_experience
            ),
            observation=(
                tti_inputs.observation
            ),
            traffic_manager=(
                traffic_manager
            ),
            tds_eligibility_config=(
                tds_eligibility_config
            ),
            state_manager=(
                state_manager
            ),
            training_controller=(
                training_controller
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
                candidate_augmentation_config
            ),
            candidate_augmentation_generator=(
                candidate_augmentation_generator
            ),
            reward_reduction=(
                reward_reduction
            ),
            packet_arrivals=(
                tti_inputs.packet_arrivals
            ),

            tds_eligibility_override_mask=(
                tti_inputs
                .tds_eligibility_override_mask
            ),

            device=device,
        )

    updates_after = (
        training_controller.num_updates
    )

    expert_updates_after = (
        training_controller
        .num_expert_guidance_updates
    )

    return PPOTrainingRunResult(
        start_tti_index=(
            start_tti_index
        ),
        end_tti_index_exclusive=(
            start_tti_index
            + num_ttis
        ),
        num_ttis=num_ttis,
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
        final_agent_buffer_size=(
            training_controller
            .agent_buffer_size
        ),
        has_unresolved_tti_boundary=(
            training_controller
            .has_unresolved_tti_boundary
        ),
        num_expert_guidance_updates=(
            expert_updates_after
            - expert_updates_before
        ),
    )


