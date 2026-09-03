from types import SimpleNamespace

import torch

import oran_scheduler.rl.ppo_multicell_training_runner as runner_module
from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)
from oran_scheduler.rl.ppo_critic import (
    OneLDSPPOCritic,
    OneLDSPPOCriticConfig,
)
from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
)
from oran_scheduler.rl.ppo_multicell_rollout import (
    OneLDSPPOMultiCellRolloutConfig,
    OneLDSPPOMultiCellRolloutController,
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
from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingRunnerConfig,
    PPOTrainingTTIInputs,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizerConfig,
    create_ppo_optimizers,
)
from oran_scheduler.simulator.tds_eligibility import (
    TDSBufferEligibilityConfig,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)

def test_multicell_runner_updates_before_next_tti_actions(
    monkeypatch,
):
    device = (
        torch.device(
            "cuda:0"
        )
        if torch.cuda.is_available()
        else torch.device(
            "cpu"
        )
    )

    state_config = OneLDSStateConfig(
        throughput_normalization_bps=(
            100.0e6
        ),
        buffer_normalization=8000.0,
        subband_cqi_normalization=15.0,
        num_candidates=2,
        num_rbgs=1,
        max_rank=2,
    )

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=(
                state_config.state_size
            ),
            hidden_size=8,
            num_rbgs=1,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=(
                state_config.state_size
            ),
            hidden_size=8,
        )
    ).to(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    transition_buffer = (
        PPOMultiStreamTransitionBuffer(
            config=(
                PPOMultiStreamBufferConfig(
                    num_streams=2,
                    update_size=4,
                )
            )
        )
    )

    centralized = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=(
                transition_buffer
            ),
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode="require_exact",
                )
            ),
            gae_config=PPOGAEConfig(
                gae_lambda=0.9,
            ),
            loss_config=PPOLossConfig(
                entropy_coefficient=0.0,
            ),
        )
    )

    controllers = tuple(
        OneLDSPPOMultiCellRolloutController(
            stream_id=cell_index,
            actor=actor,
            critic=critic,
            transition_buffer=(
                transition_buffer
            ),
            config=(
                OneLDSPPOMultiCellRolloutConfig(
                    num_user_slots=2,
                )
            ),
        )
        for cell_index
        in range(
            2
        )
    )

    class DummyManager:
        def __init__(
            self,
            cell_index: int,
        ) -> None:
            self.cell_index = cell_index

    traffic_managers = (
        DummyManager(0),
        DummyManager(1),
    )

    state_managers = (
        DummyManager(0),
        DummyManager(1),
    )

    execution_events = []

    def build_state(
        *,
        tti_index: int,
        cell_index: int,
        user_slot_index: int,
    ) -> torch.Tensor:
        value = float(
            100 * cell_index
            + 10 * tti_index
            + user_slot_index
        )

        return torch.full(
            (
                state_config.state_size,
            ),
            fill_value=value,
            dtype=torch.float32,
            device=device,
        )

    def build_decision(
        *,
        tti_index: int,
        cell_index: int,
        user_slot_index: int,
    ):
        return SimpleNamespace(
            state_data=SimpleNamespace(
                state=build_state(
                    tti_index=tti_index,
                    cell_index=cell_index,
                    user_slot_index=(
                        user_slot_index
                    ),
                )
            ),
            action_mask=torch.ones(
                (
                    1,
                    3,
                ),
                dtype=torch.bool,
                device=device,
            ),
        )

    def fake_prepare(
        **kwargs,
    ):
        tti_index = kwargs[
            "tti_index"
        ]

        state_manager = kwargs[
            "state_manager"
        ]

        cell_index = (
            state_manager.cell_index
        )

        return SimpleNamespace(
            tti_index=tti_index,
            num_user_slots=2,
            first_decision=build_decision(
                tti_index=tti_index,
                cell_index=cell_index,
                user_slot_index=0,
            ),
        )

    def fake_run_cell_step(
        **kwargs,
    ):
        tti_index = kwargs[
            "tti_index"
        ]

        controller = kwargs[
            "training_controller"
        ]

        collect_experience = kwargs[
            "collect_experience"
        ]

        cell_index = (
            controller.stream_id
        )

        execution_events.append(
            (
                tti_index,
                cell_index,
                centralized.num_updates,
            )
        )

        if collect_experience:
            policy = (
                controller
                .make_action_policy(
                    tti_index=tti_index,
                )
            )

        else:
            policy = (
                controller
                .make_untracked_action_policy()
            )

        for user_slot_index in range(
            2
        ):
            decision = build_decision(
                tti_index=tti_index,
                cell_index=cell_index,
                user_slot_index=(
                    user_slot_index
                ),
            )

            policy(
                user_slot_index,
                decision,
            )

        if collect_experience:
            controller.finish_tti(
                reward_by_rbg=torch.ones(
                    (
                        2,
                        1,
                    ),
                    dtype=torch.float32,
                    device=device,
                ),
                reduced_reward=torch.ones(
                    2,
                    dtype=torch.float32,
                    device=device,
                ),
            )

        return SimpleNamespace(
            num_expert_demonstrations_added=0,
        )

    monkeypatch.setattr(
        runner_module,
        "prepare_traffic_aware_ppo_cell_tti",
        fake_prepare,
    )

    monkeypatch.setattr(
        runner_module,
        "run_traffic_aware_ppo_cell_tti_step",
        fake_run_cell_step,
    )

    def input_provider(
        tti_index: int,
        cell_index: int,
    ) -> PPOTrainingTTIInputs:
        del tti_index
        del cell_index

        return PPOTrainingTTIInputs(
            observation=None,
            physical_inputs_builder=(
                lambda prepared: prepared
            ),
        )

    result = (
        runner_module
        .run_multicell_ppo_training(
            start_tti_index=0,
            num_ttis=2,
            input_provider=(
                input_provider
            ),
            traffic_managers=(
                traffic_managers
            ),
            state_managers=(
                state_managers
            ),
            rollout_controllers=(
                controllers
            ),
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            centralized_training=(
                centralized
            ),
            tds_eligibility_config=(
                TDSBufferEligibilityConfig(
                    mode="all_serving_ues",
                )
            ),
            state_config=state_config,
            greedy_config=(
                runner_module
                .PPOGreedySearchConfig()
            ),
            reward_config=PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    100.0e6
                ),
            ),
            reward_population="candidates",
            runner_config=(
                PPOTrainingRunnerConfig(
                    first_collection_tti_index=0,
                )
            ),
            device=device,
        )
    )

    #
    # TTI 0:
    #
    # Both cells used policy update count 0.
    #
    assert execution_events[
        0
    ][
        2
    ] == 0

    assert execution_events[
        1
    ][
        2
    ] == 0

    #
    # End TTI 0:
    #
    # Each 2-slot cell has:
    #
    #     slot 0 resolved
    #     slot 1 pending
    #
    # central buffer = 2.
    #
    #
    # Beginning TTI 1 resolves BOTH pending slot-1
    # transitions:
    #
    #     total = 4
    #
    # PPO update happens BEFORE either TTI-1 cell
    # executes.
    #
    assert execution_events[
        2
    ][
        2
    ] == 1

    assert execution_events[
        3
    ][
        2
    ] == 1

    assert result.num_ppo_updates == 1

    #
    # TTI 1 finishes with each new slot-1 boundary
    # still unresolved.
    #
    assert (
        result
        .unresolved_boundary_by_cell
        == (
            True,
            True,
        )
    )

    #
    # Each cell's slot 0 from TTI 1 is finalized at
    # finish_tti().
    #
    assert (
        result
        .final_transition_buffer_size
        == 2
    )


