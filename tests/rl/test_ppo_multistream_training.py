import pytest
import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
    compute_joint_action_log_prob,
)
from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.rl.ppo_critic import (
    OneLDSPPOCritic,
    OneLDSPPOCriticConfig,
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
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
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
from oran_scheduler.rl.ppo_rollout import (
    build_pending_ppo_transition,
    finalize_ppo_transition,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizerConfig,
    create_ppo_optimizers,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def build_transition(
    *,
    actor: OneLDSPPOActor,
    critic: OneLDSPPOCritic,
    state: torch.Tensor,
    next_state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    reward: float,
    stream_tti_index: int = 0,
):
    with torch.no_grad():
        (
            log_prob_by_rbg,
            _,
        ) = actor.evaluate_actions(
            state=state.unsqueeze(
                0
            ),
            action_mask=(
                action_mask.unsqueeze(
                    0
                )
            ),
            actions=(
                actions.unsqueeze(
                    0
                )
            ),
        )

        joint_log_prob = (
            compute_joint_action_log_prob(
                log_prob_by_rbg
            )[
                0
            ]
        )

        old_value = critic(
            state.unsqueeze(
                0
            )
        )[
            0
        ]

    pending = build_pending_ppo_transition(
        tti_index=stream_tti_index,
        user_slot_index=0,
        state=state,
        actions=actions,
        action_mask=action_mask,
        old_log_prob_by_rbg=(
            log_prob_by_rbg[
                0
            ]
        ),
        old_joint_log_prob=(
            joint_log_prob
        ),
        old_value=old_value,
    )

    return finalize_ppo_transition(
        pending,
        reward_by_rbg=torch.full(
            (
                actions.shape[0],
            ),
            reward,
            dtype=torch.float32,
            device=state.device,
        ),
        reduced_reward=torch.tensor(
            reward,
            dtype=torch.float32,
            device=state.device,
        ),
        next_state=next_state,
        terminated=torch.tensor(
            False,
            dtype=torch.bool,
            device=state.device,
        ),
    )


def build_ready_training_system(
    device: torch.device,
):
    #
    # K = 2 candidates
    #
    # A = K + 1 = 3 actions
    #
    # state_size = 4:
    #     2 features per candidate
    #
    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=4,
            hidden_size=8,
            num_rbgs=1,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=4,
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

    buffer = PPOMultiStreamTransitionBuffer(
        config=PPOMultiStreamBufferConfig(
            num_streams=2,
            update_size=2,
        )
    )

    action_mask = torch.ones(
        (
            1,
            3,
        ),
        dtype=torch.bool,
        device=device,
    )

    #
    # Independent Cell 0.
    #
    buffer.add_transition(
        stream_id=0,
        transition=build_transition(
            actor=actor,
            critic=critic,
            state=torch.tensor(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
                device=device,
            ),
            next_state=torch.tensor(
                [
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                ],
                device=device,
            ),
            actions=torch.tensor(
                [
                    0,
                ],
                dtype=torch.long,
                device=device,
            ),
            action_mask=action_mask,
            reward=1.0,
        ),
    )

    #
    # Independent Cell 1.
    #
    buffer.add_transition(
        stream_id=1,
        transition=build_transition(
            actor=actor,
            critic=critic,
            state=torch.tensor(
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                ],
                device=device,
            ),
            next_state=torch.tensor(
                [
                    11.0,
                    21.0,
                    31.0,
                    41.0,
                ],
                device=device,
            ),
            actions=torch.tensor(
                [
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            action_mask=action_mask,
            reward=2.0,
        ),
    )

    return (
        actor,
        critic,
        optimizers,
        buffer,
    )



def maximum_adam_step(
    optimizer: torch.optim.Adam,
) -> int:
    steps = []

    for state in optimizer.state.values():
        if "step" not in state:
            continue

        step = state[
            "step"
        ]

        if isinstance(
            step,
            torch.Tensor,
        ):
            steps.append(
                int(
                    step.item()
                )
            )

        else:
            steps.append(
                int(
                    step
                )
            )

    if len(
        steps
    ) == 0:
        return 0

    return max(
        steps
    )


def test_multistream_training_performs_ppo_and_clears_buffer():
    device = preferred_device()

    (
        actor,
        critic,
        optimizers,
        buffer,
    ) = build_ready_training_system(
        device
    )

    coordinator = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=buffer,
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

    assert len(
        buffer
    ) == 2

    result = coordinator.update_if_ready(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
    )

    assert result is not None

    assert (
        result.ppo_update_index
        == 1
    )

    assert (
        result.num_real_transitions
        == 2
    )

    assert (
        result.num_optimizer_samples
        == 2
    )

    assert (
        result.num_transitions_by_stream
        == (
            1,
            1,
        )
    )

    assert coordinator.num_updates == 1

    #
    # Successfully consumed.
    #
    assert len(
        buffer
    ) == 0

    assert (
        maximum_adam_step(
            optimizers.actor
        )
        == 1
    )

    assert (
        maximum_adam_step(
            optimizers.critic
        )
        == 1
    )


def test_multistream_training_uses_augmented_optimizer_batch():
    device = preferred_device()

    (
        actor,
        critic,
        optimizers,
        buffer,
    ) = build_ready_training_system(
        device
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        1234
    )

    coordinator = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=buffer,
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
            candidate_augmentation_config=(
                PPOCandidateAugmentationConfig(
                    num_permutations=1,
                    include_original=True,
                )
            ),
            candidate_augmentation_generator=(
                generator
            ),
        )
    )

    result = coordinator.update_if_ready(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
    )

    assert result is not None

    #
    # Real environment experiences:
    #
    assert (
        result.num_real_transitions
        == 2
    )

    #
    # Each:
    #     original + one permutation
    #
    assert (
        result.num_optimizer_samples
        == 4
    )

    assert len(
        buffer
    ) == 0


def test_multistream_training_runs_ppo_then_expert_update():
    device = preferred_device()

    torch.manual_seed(
        1234
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            1234
        )

    (
        actor,
        critic,
        optimizers,
        buffer,
    ) = build_ready_training_system(
        device
    )

    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=100,
            )
        )
    )

    #
    # One valid Teacher-2 demonstration.
    #
    expert_buffer.add(
        state=torch.tensor(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
        expert_actions=torch.tensor(
            [
                0,
            ],
            dtype=torch.long,
            device=device,
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

    expert_coordinator = (
        PPOExpertGuidanceCoordinator(
            expert_buffer=expert_buffer,
            guidance_config=(
                PPOExpertGuidanceConfig(
                    batch_size=1,
                    guidance_weight=1.0,
                    divergence_mode="true_jsd",
                )
            ),
        )
    )

    coordinator = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=buffer,
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
            expert_guidance_coordinator=(
                expert_coordinator
            ),
        )
    )

    result = coordinator.update_if_ready(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
    )

    assert result is not None

    #
    # One centralized PPO update.
    #
    assert coordinator.num_updates == 1

    #
    # Exactly one matching Teacher-2 update.
    #
    assert (
        coordinator
        .num_expert_guidance_updates
        == 1
    )

    assert (
        expert_coordinator
        .num_guidance_updates
        == 1
    )

    assert result.expert_update is not None

    assert (
        result
        .expert_update
        .ppo_update_index
        == 1
    )

    #
    # Same actor optimizer:
    #
    #     step 1 = PPO
    #     step 2 = expert guidance
    #
    assert (
        maximum_adam_step(
            optimizers.actor
        )
        == 2
    )

    #
    # Critic participates only in PPO.
    #
    assert (
        maximum_adam_step(
            optimizers.critic
        )
        == 1
    )

    #
    # Rollout is consumed only after both.
    #
    assert len(
        buffer
    ) == 0


def test_expert_preflight_happens_before_ppo_update():
    device = preferred_device()

    (
        actor,
        critic,
        optimizers,
        buffer,
    ) = build_ready_training_system(
        device
    )

    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=100,
                sampling_with_replacement=False,
            )
        )
    )

    #
    # Only ONE demonstration...
    #
    expert_buffer.add(
        state=torch.zeros(
            4,
            device=device,
        ),
        expert_actions=torch.tensor(
            [
                0,
            ],
            dtype=torch.long,
            device=device,
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

    expert_coordinator = (
        PPOExpertGuidanceCoordinator(
            expert_buffer=expert_buffer,
            guidance_config=(
                PPOExpertGuidanceConfig(
                    #
                    # ...but Teacher 2 requires TWO.
                    #
                    batch_size=2,
                    guidance_weight=1.0,
                    divergence_mode="true_jsd",
                )
            ),
        )
    )

    coordinator = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=buffer,
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
            expert_guidance_coordinator=(
                expert_coordinator
            ),
        )
    )

    actor_before = tuple(
        parameter
        .detach()
        .clone()
        for parameter
        in actor.parameters()
    )

    critic_before = tuple(
        parameter
        .detach()
        .clone()
        for parameter
        in critic.parameters()
    )

    with pytest.raises(
        RuntimeError,
        match="enough demonstrations",
    ):
        coordinator.update_if_ready(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
        )

    #
    # PPO must NOT have happened.
    #
    assert coordinator.num_updates == 0

    assert (
        maximum_adam_step(
            optimizers.actor
        )
        == 0
    )

    assert (
        maximum_adam_step(
            optimizers.critic
        )
        == 0
    )

    for before, after in zip(
        actor_before,
        actor.parameters(),
    ):
        torch.testing.assert_close(
            before,
            after,
        )

    for before, after in zip(
        critic_before,
        critic.parameters(),
    ):
        torch.testing.assert_close(
            before,
            after,
        )

    #
    # Data remain available because nothing was
    # consumed.
    #
    assert len(
        buffer
    ) == 2


def test_use_all_when_reached_updates_all_current_samples():
    device = preferred_device()

    (
        actor,
        critic,
        optimizers,
        buffer,
    ) = build_ready_training_system(
        device
    )

    #
    # M = 2, but add a third current-policy sample.
    #
    action_mask = torch.ones(
        (
            1,
            3,
        ),
        dtype=torch.bool,
        device=device,
    )

    buffer.add_transition(
        stream_id=0,
        transition=build_transition(
            actor=actor,
            critic=critic,
            #
            # Continue Cell 0:
            #
            # previous next_state:
            #     [2,3,4,5]
            #
            state=torch.tensor(
                [
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                ],
                device=device,
            ),
            next_state=torch.tensor(
                [
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ],
                device=device,
            ),
            actions=torch.tensor(
                [
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            action_mask=action_mask,
            reward=1.5,
            stream_tti_index=1,
        ),
    )

    assert len(
        buffer
    ) == 3

    coordinator = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=buffer,
            update_config=(
                PPOMultiStreamUpdateConfig(
                    boundary_mode=(
                        "use_all_when_reached"
                    ),
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

    result = coordinator.update_if_ready(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
    )

    assert result is not None

    assert (
        result.num_real_transitions
        == 3
    )

    assert (
        result.num_transitions_by_stream
        == (
            2,
            1,
        )
    )

    assert len(
        buffer
    ) == 0


def test_multistream_training_waits_when_not_ready():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=4,
            hidden_size=8,
            num_rbgs=1,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=4,
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

    buffer = PPOMultiStreamTransitionBuffer(
        config=PPOMultiStreamBufferConfig(
            num_streams=2,
            update_size=2,
        )
    )

    coordinator = (
        PPOMultiStreamTrainingCoordinator(
            transition_buffer=buffer,
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

    result = coordinator.update_if_ready(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
    )

    assert result is None

    assert coordinator.num_updates == 0

    assert (
        maximum_adam_step(
            optimizers.actor
        )
        == 0
    )

    assert (
        maximum_adam_step(
            optimizers.critic
        )
        == 0
    )

