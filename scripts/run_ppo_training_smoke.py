from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import torch

from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertBufferConfig,
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_expert_guidance import (
    PPOExpertGuidanceConfig,
)
from oran_scheduler.rl.ppo_expert_training import (
    PPOExpertGuidanceCoordinator,
)
from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
)
from oran_scheduler.rl.ppo_multicell_rollout import (
    OneLDSPPOMultiCellRolloutConfig,
    OneLDSPPOMultiCellRolloutController,
)
from oran_scheduler.rl.ppo_multicell_training_runner import (
    run_multicell_ppo_training,
)
from oran_scheduler.rl.ppo_multistream_buffer import (
    PPOMultiStreamBufferConfig,
    PPOMultiStreamTransitionBuffer,
)
from oran_scheduler.rl.ppo_multistream_training import (
    PPOMultiStreamTrainingCoordinator,
)
from oran_scheduler.rl.ppo_multistream_update import (
    PPOMultiStreamUpdateConfig,
)
from oran_scheduler.rl.ppo_pf_expert import (
    PPOPFExpertConfig,
)
from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingRunnerConfig,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizerConfig,
    create_ppo_optimizers,
)
from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)



NUM_CELLS = 2

NUM_USER_SLOTS = 2

NUM_TTIS = 8

UPDATE_SIZE = (
    NUM_CELLS
    * NUM_USER_SLOTS
)

DEVICE = torch.device(
    "cuda:0"
)

SEED = 1234


def load_smoke_fixture_module():
    """
    TEMPORARY SCAFFOLDING.

    Reuse the already-tested small physical training
    fixture without duplicating its environment setup
    inside this script.

    The next simulator-integration milestone removes
    this dependency and supplies real Sionna-generated
    observations/physical inputs.
    """

    repository_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    fixture_path = (
        repository_root
        / "tests"
        / "rl"
        / "test_ppo_training_runner.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "_ppo_training_smoke_fixture",
            fixture_path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not load PPO training smoke "
            "fixture."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def bytes_to_mib(
    value: int,
) -> float:
    return (
        float(value)
        / (
            1024.0
            * 1024.0
        )
    )


def module_parameter_norm(
    module: torch.nn.Module,
) -> float:
    """
    L2 norm over all trainable parameters.

    This gives us one simple diagnostic showing that
    the actor/critic parameters evolve over updates.
    """

    with torch.no_grad():
        squared_norm = torch.zeros(
            (),
            dtype=torch.float64,
            device=DEVICE,
        )

        for parameter in module.parameters():
            squared_norm += (
                parameter
                .detach()
                .to(
                    dtype=torch.float64
                )
                .square()
                .sum()
            )

        return float(
            torch.sqrt(
                squared_norm
            ).item()
        )

    
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This smoke run requires a GPU. "
            "Request a CHPC GPU allocation and use "
            "cuda:0."
        )

    torch.manual_seed(
        SEED
    )

    torch.cuda.manual_seed_all(
        SEED
    )

    fixture = (
        load_smoke_fixture_module()
    )

    build_small_training_system = (
        fixture
        .build_small_training_system
    )

    build_input_provider = (
        fixture
        .build_input_provider
    )


    (
        state_config_0,
        unused_controller_0,
        state_manager_0,
        traffic_manager_0,
    ) = build_small_training_system(
        DEVICE
    )

    (
        state_config_1,
        unused_controller_1,
        state_manager_1,
        traffic_manager_1,
    ) = build_small_training_system(
        DEVICE
    )

    if (
        state_config_0
        != state_config_1
    ):
        raise RuntimeError(
            "The two smoke-test cells expose "
            "different RL state configurations."
        )

    state_config = (
        state_config_0
    )


    actor = (
        unused_controller_0
        .actor
    )

    critic = (
        unused_controller_0
        .critic
    )

    actor.train()

    critic.train()

    optimizers = (
        create_ppo_optimizers(
            actor=actor,
            critic=critic,
            config=(
                PPOOptimizerConfig()
            ),
        )
    )


    transition_buffer = (
        PPOMultiStreamTransitionBuffer(
            config=(
                PPOMultiStreamBufferConfig(
                    num_streams=(
                        NUM_CELLS
                    ),
                    update_size=(
                        UPDATE_SIZE
                    ),
                )
            )
        )
    )


    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                #
                # Paper buffer capacity.
                #
                capacity=4000,

                #
                # PAPER-UNSPECIFIED replacement
                # behavior.
                #
                replacement_mode="fifo",

                sampling_with_replacement=False,
            )
        )
    )

    expert_guidance = (
        PPOExpertGuidanceCoordinator(
            expert_buffer=(
                expert_buffer
            ),
            guidance_config=(
                PPOExpertGuidanceConfig(
                    #
                    # TEST-SCALE b'.
                    #
                    batch_size=2,

                    #
                    # OPEN-REPRODUCTION parameter.
                    #
                    guidance_weight=1.0,

                    #
                    # Primary interpretation of
                    # the paper's textual JSD
                    # description.
                    #
                    divergence_mode="true_jsd",
                )
            ),
        )
    )


    augmentation_config = (
        PPOCandidateAugmentationConfig(
            #
            # TEST-SCALE N_Pi.
            #
            num_permutations=1,

            include_original=True,
        )
    )

    ppo_augmentation_generator = (
        torch.Generator(
            device="cpu"
        )
    )

    ppo_augmentation_generator.manual_seed(
        1111
    )

    expert_augmentation_generator = (
        torch.Generator(
            device="cpu"
        )
    )

    expert_augmentation_generator.manual_seed(
        2222
    )


    rollout_controllers = tuple(
        OneLDSPPOMultiCellRolloutController(
            stream_id=cell_index,
            actor=actor,
            critic=critic,
            transition_buffer=(
                transition_buffer
            ),
            config=(
                OneLDSPPOMultiCellRolloutConfig(
                    num_user_slots=(
                        NUM_USER_SLOTS
                    ),
                )
            ),
        )
        for cell_index
        in range(
            NUM_CELLS
        )
    )


    centralized_training = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=(
                transition_buffer
            ),

            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode=(
                        "require_exact"
                    ),
                )
            ),

            gae_config=(
                PPOGAEConfig(
                    #
                    # OPEN-REPRODUCTION value.
                    #
                    gae_lambda=0.9,
                )
            ),

            loss_config=(
                PPOLossConfig(
                    #
                    # OPEN-REPRODUCTION value.
                    #
                    entropy_coefficient=0.0,
                )
            ),

            candidate_augmentation_config=(
                augmentation_config
            ),

            candidate_augmentation_generator=(
                ppo_augmentation_generator
            ),

            expert_guidance_coordinator=(
                expert_guidance
            ),
        )
    )


    provider_0 = (
        build_input_provider(
            DEVICE
        )
    )

    provider_1 = (
        build_input_provider(
            DEVICE
        )
    )


    def multicell_input_provider(
        tti_index: int,
        cell_index: int,
    ):
        if cell_index == 0:
            return provider_0(
                tti_index
            )

        if cell_index == 1:
            return provider_1(
                tti_index
            )

        raise ValueError(
            "Unexpected cell index."
        )


    print()
    print(
        "=" * 72
    )
    print(
        "CENTRALIZED PPO TRAINING SMOKE RUN"
    )
    print(
        "=" * 72
    )

    print(
        f"Device:             {DEVICE}"
    )

    print(
        f"GPU:                "
        f"{torch.cuda.get_device_name(DEVICE)}"
    )

    print(
        f"Cells:              {NUM_CELLS}"
    )

    print(
        f"User slots/cell:    {NUM_USER_SLOTS}"
    )

    print(
        f"TTIs:               {NUM_TTIS}"
    )

    print(
        f"Real update size:   {UPDATE_SIZE}"
    )

    print(
        "Candidate copies:   "
        "original + 1 permutation"
    )

    print(
        "Expert guidance:    enabled"
    )

    print(
        "Warm-up:            disabled "
        "(smoke-test only)"
    )

    print(
        "-" * 72
    )

    print(
        "IMPORTANT: this is a TEST-SCALE physical "
        "integration run, not yet the paper's "
        "21-cell / 420-UE Sionna experiment."
    )

    print(
        "=" * 72
    )
    print()


    initial_actor_norm = (
        module_parameter_norm(
            actor
        )
    )

    initial_critic_norm = (
        module_parameter_norm(
            critic
        )
    )

    total_start_time = (
        time.perf_counter()
    )


    for tti_index in range(
        NUM_TTIS
    ):
        updates_before = (
            centralized_training
            .num_updates
        )

        expert_updates_before = (
            centralized_training
            .num_expert_guidance_updates
        )

        expert_buffer_before = len(
            expert_buffer
        )

        torch.cuda.reset_peak_memory_stats(
            DEVICE
        )

        tti_start_time = (
            time.perf_counter()
        )

        result = (
            run_multicell_ppo_training(
                start_tti_index=(
                    tti_index
                ),

                #
                # Intentionally execute ONE TTI at
                # a time so diagnostics are visible.
                #
                num_ttis=1,

                input_provider=(
                    multicell_input_provider
                ),

                traffic_managers=(
                    traffic_manager_0,
                    traffic_manager_1,
                ),

                state_managers=(
                    state_manager_0,
                    state_manager_1,
                ),

                rollout_controllers=(
                    rollout_controllers
                ),

                actor=actor,

                critic=critic,

                optimizers=optimizers,

                centralized_training=(
                    centralized_training
                ),

                tds_eligibility_config=(
                    TDSBufferEligibilityConfig(
                        mode=(
                            "data_available_only"
                        ),
                    )
                ),

                state_config=(
                    state_config
                ),

                greedy_config=(
                    PPOGreedySearchConfig()
                ),

                reward_config=(
                    PPORewardConfig(
                        geometric_mean_normalizer_bps=(
                            100.0e6
                        ),
                    )
                ),

                reward_population=(
                    "candidates"
                ),

                #
                # TEST-SCALE:
                # collect from TTI 0.
                #
                # Paper uses the first 100 TTIs as
                # non-collection warm-up.
                #
                runner_config=(
                    PPOTrainingRunnerConfig(
                        first_collection_tti_index=0,
                    )
                ),

                expert_buffer=(
                    expert_buffer
                ),

                pf_expert_config=(
                    PPOPFExpertConfig()
                ),

                expert_candidate_augmentation_config=(
                    augmentation_config
                ),

                expert_candidate_augmentation_generator=(
                    expert_augmentation_generator
                ),

                device=DEVICE,
            )
        )

        torch.cuda.synchronize(
            DEVICE
        )

        elapsed = (
            time.perf_counter()
            - tti_start_time
        )

        updates_after = (
            centralized_training
            .num_updates
        )

        expert_updates_after = (
            centralized_training
            .num_expert_guidance_updates
        )

        update_happened = (
            updates_after
            > updates_before
        )

        expert_update_happened = (
            expert_updates_after
            > expert_updates_before
        )

        allocated_memory = (
            torch.cuda.memory_allocated(
                DEVICE
            )
        )

        peak_memory = (
            torch.cuda.max_memory_allocated(
                DEVICE
            )
        )

        print(
            f"TTI {tti_index:03d}"
        )

        print(
            "  centralized buffer: "
            f"{result.final_transition_buffer_size}"
            f"/{UPDATE_SIZE}"
        )

        print(
            "  expert demos added: "
            f"{len(expert_buffer) - expert_buffer_before}"
        )

        print(
            "  expert buffer:      "
            f"{len(expert_buffer)}/4000"
        )

        print(
            "  unresolved cells:   "
            f"{sum(result.unresolved_boundary_by_cell)}"
            f"/{NUM_CELLS}"
        )

        if update_happened:
            update = (
                centralized_training
                .last_update
            )

            if update is None:
                raise RuntimeError(
                    "Centralized update count changed "
                    "without update diagnostics."
                )

            print(
                "  >>> PPO UPDATE "
                f"#{update.ppo_update_index}"
            )

            print(
                "      real transitions: "
                f"{update.num_real_transitions}"
            )

            print(
                "      optimizer samples: "
                f"{update.num_optimizer_samples}"
            )

            print(
                "      actor loss:       "
                f"{float(update.ppo_update.actor_loss.item()): .6f}"
            )

            print(
                "      critic loss:      "
                f"{float(update.ppo_update.critic_loss.item()): .6f}"
            )

            print(
                "      mean PPO ratio:   "
                f"{float(update.ppo_update.mean_probability_ratio.item()): .6f}"
            )

            print(
                "      mean advantage:   "
                f"{float(update.ppo_update.mean_advantage.item()): .6f}"
            )

            if (
                update.expert_update
                is not None
            ):
                expert_loss_data = (
                    update
                    .expert_update
                    .guidance_update
                    .loss_data
                )

                print(
                    "  >>> EXPERT UPDATE "
                    f"#{update.expert_update.ppo_update_index}"
                )

                print(
                    "      raw JSD loss:    "
                    f"{float(expert_loss_data.raw_loss.item()): .6f}"
                )

                print(
                    "      weighted loss:   "
                    f"{float(expert_loss_data.weighted_loss.item()): .6f}"
                )

        elif expert_update_happened:
            raise RuntimeError(
                "Expert update occurred without a "
                "matching centralized PPO update."
            )


        print(
            "  actor norm:          "
            f"{module_parameter_norm(actor):.6f}"
        )

        print(
            "  critic norm:         "
            f"{module_parameter_norm(critic):.6f}"
        )

        print(
            "  GPU allocated:       "
            f"{bytes_to_mib(allocated_memory):.1f} MiB"
        )

        print(
            "  GPU peak this TTI:   "
            f"{bytes_to_mib(peak_memory):.1f} MiB"
        )

        print(
            "  elapsed:             "
            f"{elapsed:.3f} s"
        )

        print()


    torch.cuda.synchronize(
        DEVICE
    )

    total_elapsed = (
        time.perf_counter()
        - total_start_time
    )

    final_actor_norm = (
        module_parameter_norm(
            actor
        )
    )

    final_critic_norm = (
        module_parameter_norm(
            critic
        )
    )

    print(
        "=" * 72
    )

    print(
        "SMOKE RUN COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "Centralized PPO updates: "
        f"{centralized_training.num_updates}"
    )

    print(
        "Expert-guidance updates: "
        f"{centralized_training.num_expert_guidance_updates}"
    )

    print(
        "Expert buffer size:       "
        f"{len(expert_buffer)}"
    )

    print(
        "Final rollout buffer:     "
        f"{len(transition_buffer)}"
    )

    print(
        "Actor parameter norm:     "
        f"{initial_actor_norm:.6f}"
        " -> "
        f"{final_actor_norm:.6f}"
    )

    print(
        "Critic parameter norm:    "
        f"{initial_critic_norm:.6f}"
        " -> "
        f"{final_critic_norm:.6f}"
    )

    print(
        "Total elapsed:            "
        f"{total_elapsed:.3f} s"
    )

    print(
        "=" * 72
    )

if __name__ == "__main__":
    main()