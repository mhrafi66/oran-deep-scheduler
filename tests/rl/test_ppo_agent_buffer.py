import pytest
import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
    compute_joint_action_log_prob,
)
from oran_scheduler.rl.ppo_agent_buffer import (
    PPOAgentBufferConfig,
    PPOAgentUpdateCoordinator,
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
from oran_scheduler.rl.ppo_rollout import (
    PPORolloutTransition,
    build_pending_ppo_transition,
    finalize_ppo_transition,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizerConfig,
    create_ppo_optimizers,
)

from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )



def build_paper_shaped_transitions(
    *,
    actor: OneLDSPPOActor,
    critic: OneLDSPPOCritic,
    device: torch.device,
    num_transitions: int = 128,
) -> list[
    PPORolloutTransition
]:
    """
    Synthetic but paper-shaped contiguous PPO
    trajectory.

    Shapes:

        state:
            410

        RBGs:
            18

        actions/RBG:
            11
    """

    state_size = 410

    num_rbgs = 18

    num_actions = 11

    #
    # Need N+1 states because every transition has
    # a real successor state.
    #
    states = torch.rand(
        (
            num_transitions + 1,
            state_size,
        ),
        dtype=torch.float32,
        device=device,
    )

    action_masks = torch.ones(
        (
            num_transitions,
            num_rbgs,
            num_actions,
        ),
        dtype=torch.bool,
        device=device,
    )

    #
    # Exercise masking too.
    #
    action_masks[
        :,
        0,
        3,
    ] = False

    action_masks[
        :,
        7,
        5,
    ] = False

    with torch.no_grad():
        sampled = actor.sample_actions(
            state=states[:-1],
            action_mask=action_masks,
        )

        old_log_prob_by_rbg = (
            sampled
            .log_prob_by_rbg
            .detach()
            .clone()
        )

        old_joint_log_prob = (
            compute_joint_action_log_prob(
                old_log_prob_by_rbg
            )
            .detach()
            .clone()
        )

        old_value = (
            critic(
                states[:-1]
            )
            .detach()
            .clone()
        )

    transitions: list[
        PPORolloutTransition
    ] = []

    for index in range(
        num_transitions
    ):
        sign = (
            1.0
            if index % 2 == 0
            else -1.0
        )

        reward_by_rbg = torch.full(
            (
                num_rbgs,
            ),
            0.2 * sign,
            dtype=torch.float32,
            device=device,
        )

        reduced_reward = (
            reward_by_rbg.mean()
        )

        transitions.append(
            PPORolloutTransition(
                tti_index=(
                    index // 4
                ),
                user_slot_index=(
                    index % 4
                ),
                state=(
                    states[index]
                    .detach()
                    .clone()
                ),
                actions=(
                    sampled
                    .actions[index]
                    .detach()
                    .clone()
                ),
                action_mask=(
                    action_masks[index]
                    .detach()
                    .clone()
                ),
                old_log_prob_by_rbg=(
                    old_log_prob_by_rbg[
                        index
                    ]
                    .detach()
                    .clone()
                ),
                old_joint_log_prob=(
                    old_joint_log_prob[
                        index
                    ]
                    .detach()
                    .clone()
                ),
                reward_by_rbg=(
                    reward_by_rbg
                    .detach()
                    .clone()
                ),
                reduced_reward=(
                    reduced_reward
                    .detach()
                    .clone()
                ),
                old_value=(
                    old_value[index]
                    .detach()
                    .clone()
                ),
                next_state=(
                    states[index + 1]
                    .detach()
                    .clone()
                ),
                terminated=torch.tensor(
                    False,
                    dtype=torch.bool,
                    device=device,
                ),
            )
        )

    return transitions



def snapshot_parameters(
    module: torch.nn.Module,
) -> list[
    torch.Tensor
]:
    return [
        parameter
        .detach()
        .clone()
        for parameter
        in module.parameters()
    ]


def any_parameter_changed(
    before: list[
        torch.Tensor
    ],
    module: torch.nn.Module,
) -> bool:
    return any(
        not torch.equal(
            old,
            new.detach(),
        )
        for old, new in zip(
            before,
            module.parameters(),
        )
    )


def test_agent_buffer_uses_paper_update_size():
    coordinator = (
        PPOAgentUpdateCoordinator()
    )

    assert (
        coordinator
        .config
        .update_size
        == 128
    )

    assert len(
        coordinator
    ) == 0

    assert coordinator.is_empty

    assert not coordinator.is_full

    assert (
        coordinator.remaining_capacity
        == 128
    )


def test_agent_buffer_does_not_update_before_full():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig()
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig()
    ).to(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    transitions = (
        build_paper_shaped_transitions(
            actor=actor,
            critic=critic,
            device=device,
            num_transitions=127,
        )
    )

    coordinator = (
        PPOAgentUpdateCoordinator()
    )

    coordinator.add_transitions(
        transitions
    )

    result = coordinator.update_if_ready(
        actor=actor,
        critic=critic,
        optimizers=optimizers,

        #
        # PAPER-UNSPECIFIED lambda.
        # Explicit unit-test choice only.
        #
        gae_config=PPOGAEConfig(
            gae_lambda=0.9,
        ),

        #
        # PAPER-UNSPECIFIED entropy coefficient.
        # Zero isolates PPO behavior in this test.
        #
        loss_config=PPOLossConfig(
            entropy_coefficient=0.0,
        ),
    )

    assert result is None

    assert len(
        coordinator
    ) == 127

def test_full_agent_buffer_runs_gae_updates_models_and_clears():
    device = preferred_device()

    torch.manual_seed(
        1234
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            1234
        )

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig()
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig()
    ).to(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    transitions = (
        build_paper_shaped_transitions(
            actor=actor,
            critic=critic,
            device=device,
        )
    )

    coordinator = (
        PPOAgentUpdateCoordinator()
    )

    coordinator.add_transitions(
        transitions
    )

    assert len(
        coordinator
    ) == 128

    assert coordinator.is_full

    actor_before = snapshot_parameters(
        actor
    )

    critic_before = snapshot_parameters(
        critic
    )

    result = coordinator.update_if_ready(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        gae_config=PPOGAEConfig(
            gae_lambda=0.9,
        ),
        loss_config=PPOLossConfig(
            entropy_coefficient=0.0,
        ),
    )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    assert result is not None

    assert (
        result.num_transitions
        == 128
    )

    assert any_parameter_changed(
        actor_before,
        actor,
    )

    assert any_parameter_changed(
        critic_before,
        critic,
    )

    assert torch.isfinite(
        result.bootstrap_value
    )

    assert torch.isfinite(
        result.mean_td_residual
    )

    assert torch.isfinite(
        result.mean_advantage
    )

    assert torch.isfinite(
        result.mean_target_return
    )

    assert coordinator.is_empty

    assert len(
        coordinator
    ) == 0

    assert (
        coordinator.remaining_capacity
        == 128
    )

def test_agent_buffer_refuses_overflow():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig()
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig()
    ).to(
        device
    )

    transitions = (
        build_paper_shaped_transitions(
            actor=actor,
            critic=critic,
            device=device,
            num_transitions=4,
        )
    )

    coordinator = (
        PPOAgentUpdateCoordinator(
            config=PPOAgentBufferConfig(
                update_size=3,
            )
        )
    )

    with pytest.raises(
        RuntimeError
    ):
        coordinator.add_transitions(
            transitions
        )

    #
    # Atomic operation:
    # none of the four should have been added.
    #
    assert len(
        coordinator
    ) == 0


def test_agent_buffer_rejects_noncontiguous_transitions():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig()
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig()
    ).to(
        device
    )

    transitions = (
        build_paper_shaped_transitions(
            actor=actor,
            critic=critic,
            device=device,
            num_transitions=2,
        )
    )

    second = transitions[1]

    broken_second = (
        PPORolloutTransition(
            tti_index=second.tti_index,
            user_slot_index=(
                second.user_slot_index
            ),
            state=(
                second.state
                + 1.0
            ),
            actions=second.actions,
            action_mask=(
                second.action_mask
            ),
            old_log_prob_by_rbg=(
                second
                .old_log_prob_by_rbg
            ),
            old_joint_log_prob=(
                second
                .old_joint_log_prob
            ),
            reward_by_rbg=(
                second.reward_by_rbg
            ),
            reduced_reward=(
                second.reduced_reward
            ),
            old_value=(
                second.old_value
            ),
            next_state=(
                second.next_state
            ),
            terminated=(
                second.terminated
            ),
        )
    )

    coordinator = (
        PPOAgentUpdateCoordinator()
    )

    coordinator.add_transition(
        transitions[0]
    )

    with pytest.raises(
        ValueError
    ):
        coordinator.add_transition(
            broken_second
        )

    assert len(
        coordinator
    ) == 1


def test_agent_buffer_update_stays_on_model_device():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig()
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig()
    ).to(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    transitions = (
        build_paper_shaped_transitions(
            actor=actor,
            critic=critic,
            device=device,
        )
    )

    coordinator = (
        PPOAgentUpdateCoordinator()
    )

    coordinator.add_transitions(
        transitions
    )

    result = coordinator.update_if_ready(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        gae_config=PPOGAEConfig(
            gae_lambda=0.9,
        ),
        loss_config=PPOLossConfig(
            entropy_coefficient=0.0,
        ),
    )

    assert result is not None

    assert (
        result.bootstrap_value.device
        == device
    )

    assert (
        result.optimizer.actor_loss.device
        == device
    )

    assert (
        result.optimizer.critic_loss.device
        == device
    )

    for parameter in actor.parameters():
        assert (
            parameter.device
            == device
        )

    for parameter in critic.parameters():
        assert (
            parameter.device
            == device
        )

def test_agent_buffer_augments_only_at_update_time():
    device = (
        torch.device(
            "cuda:0"
        )
        if torch.cuda.is_available()
        else torch.device(
            "cpu"
        )
    )

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=6,
            hidden_size=8,
            num_rbgs=2,
            num_actions_per_rbg=4,
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=6,
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

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        1234
    )

    coordinator = (
        PPOAgentUpdateCoordinator(
            config=PPOAgentBufferConfig(
                update_size=2,
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

    state_0 = torch.arange(
        6,
        dtype=torch.float32,
        device=device,
    )

    state_1 = state_0 + 10.0

    state_2 = state_0 + 20.0

    action_mask = torch.ones(
        (
            2,
            4,
        ),
        dtype=torch.bool,
        device=device,
    )

    def make_transition(
        *,
        state,
        next_state,
        actions,
        slot,
        reward,
    ):
        with torch.no_grad():
            log_prob, _ = (
                actor.evaluate_actions(
                    state=state.unsqueeze(
                        0
                    ),
                    action_mask=(
                        action_mask.unsqueeze(
                            0
                        )
                    ),
                    actions=actions.unsqueeze(
                        0
                    ),
                )
            )

            joint_log_prob = (
                compute_joint_action_log_prob(
                    log_prob
                )[
                    0
                ]
            )

            value = critic(
                state.unsqueeze(
                    0
                )
            )[
                0
            ]

        pending = (
            build_pending_ppo_transition(
                tti_index=0,
                user_slot_index=slot,
                state=state,
                actions=actions,
                action_mask=(
                    action_mask
                ),
                old_log_prob_by_rbg=(
                    log_prob[
                        0
                    ]
                ),
                old_joint_log_prob=(
                    joint_log_prob
                ),
                old_value=value,
            )
        )

        return finalize_ppo_transition(
            pending,
            reward_by_rbg=torch.full(
                (
                    2,
                ),
                reward,
                device=device,
            ),
            reduced_reward=torch.tensor(
                reward,
                device=device,
            ),
            next_state=next_state,
            terminated=torch.tensor(
                False,
                dtype=torch.bool,
                device=device,
            ),
        )

    coordinator.add_transition(
        make_transition(
            state=state_0,
            next_state=state_1,
            actions=torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            ),
            slot=0,
            reward=0.1,
        )
    )

    #
    # Still exactly ONE real temporal transition.
    #
    assert len(
        coordinator
    ) == 1

    coordinator.add_transition(
        make_transition(
            state=state_1,
            next_state=state_2,
            actions=torch.tensor(
                [
                    2,
                    3,
                ],
                dtype=torch.long,
                device=device,
            ),
            slot=1,
            reward=0.2,
        )
    )

    #
    # Buffer contains 2 REAL transitions.
    #
    assert len(
        coordinator
    ) == 2

    result = coordinator.update_if_ready(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        gae_config=PPOGAEConfig(
            gae_lambda=0.9,
        ),
        loss_config=PPOLossConfig(
            entropy_coefficient=0.0,
        ),
    )

    assert result is not None

    #
    # Real trajectory:
    #     2
    #
    # Optimizer representations:
    #     original + 1 permutation
    #     for each transition
    #
    #     2 * 2 = 4
    #
    assert result.num_transitions == 2

    assert (
        result.num_optimizer_samples
        == 4
    )

    #
    # Successful PPO update clears only the real
    # temporal rollout buffer.
    #
    assert len(
        coordinator
    ) == 0

    #
    # Before optimizer.step(), new and old policies
    # are identical, so the mean PPO ratio used by
    # the update should begin at approximately 1.
    #
    torch.testing.assert_close(
        result
        .optimizer
        .mean_probability_ratio,
        torch.tensor(
            1.0,
            dtype=torch.float32,
            device=device,
        ),
        atol=1.0e-5,
        rtol=1.0e-5,
    )



