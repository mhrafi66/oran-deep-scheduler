import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
    compute_joint_action_log_prob,
)
from oran_scheduler.rl.ppo_critic import (
    OneLDSPPOCritic,
    OneLDSPPOCriticConfig,
)
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
)
from oran_scheduler.rl.ppo_rollout import (
    PPORolloutBatch,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizerConfig,
    create_ppo_optimizers,
    perform_ppo_update,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def build_paper_shaped_update_inputs(
    device: torch.device,
):
    """
    Build one synthetic PPO update batch using the
    actual paper-shaped 1LDS dimensions:

        batch = 128
        state = 410
        RBGs = 18
        actions/RBG = 11
    """

    torch.manual_seed(
        1234
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            1234
        )

    num_transitions = 128

    state_size = 410

    num_rbgs = 18

    num_actions = 11

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=state_size,
            hidden_size=32,
            num_rbgs=num_rbgs,
            num_actions_per_rbg=(
                num_actions
            ),
        )
    ).to(
        device
    )

    critic = OneLDSPPOCritic(
        OneLDSPPOCriticConfig(
            state_size=state_size,
            hidden_size=32,
        )
    ).to(
        device
    )

    states = torch.rand(
        (
            num_transitions,
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
    # Make a few actions invalid so that this tests
    # the real masked-policy path rather than only
    # the trivial all-valid case.
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
            state=states,
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
                states
            )
            .detach()
            .clone()
        )

    #
    # The optimizer does not need rewards or
    # next_states directly, but PPORolloutBatch
    # preserves the complete rollout structure.
    #

    reward_by_rbg = torch.zeros(
        (
            num_transitions,
            num_rbgs,
        ),
        dtype=torch.float32,
        device=device,
    )

    reduced_reward = torch.zeros(
        num_transitions,
        dtype=torch.float32,
        device=device,
    )

    next_states = torch.rand(
        (
            num_transitions,
            state_size,
        ),
        dtype=torch.float32,
        device=device,
    )

    terminated = torch.zeros(
        num_transitions,
        dtype=torch.bool,
        device=device,
    )

    batch = PPORolloutBatch(
        states=states,
        actions=sampled.actions.detach().clone(),
        action_masks=action_masks,
        old_log_prob_by_rbg=(
            old_log_prob_by_rbg
        ),
        old_joint_log_prob=(
            old_joint_log_prob
        ),
        reward_by_rbg=(
            reward_by_rbg
        ),
        reduced_reward=(
            reduced_reward
        ),
        old_value=old_value,
        next_states=next_states,
        terminated=terminated,
    )

    #
    # Synthetic GAE outputs for the optimizer unit
    # test.
    #
    # GAE itself is already tested separately.
    #

    advantage = torch.linspace(
        -1.0,
        1.0,
        steps=num_transitions,
        dtype=torch.float32,
        device=device,
    )

    target_return = (
        old_value
        + torch.linspace(
            -0.5,
            0.5,
            steps=num_transitions,
            dtype=torch.float32,
            device=device,
        )
    )

    return (
        actor,
        critic,
        batch,
        advantage,
        target_return,
    )


def snapshot_parameters(
    module: torch.nn.Module,
) -> list[torch.Tensor]:
    return [
        parameter
        .detach()
        .clone()
        for parameter
        in module.parameters()
    ]


def any_parameter_changed(
    before: list[torch.Tensor],
    module: torch.nn.Module,
) -> bool:
    after = list(
        module.parameters()
    )

    return any(
        not torch.equal(
            old,
            new.detach(),
        )
        for old, new in zip(
            before,
            after,
        )
    )


def test_create_ppo_optimizers_uses_paper_learning_rates():
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

    assert isinstance(
        optimizers.actor,
        torch.optim.Adam,
    )

    assert isinstance(
        optimizers.critic,
        torch.optim.Adam,
    )

    assert (
        optimizers
        .actor
        .param_groups[0]["lr"]
        == 1.0e-4
    )

    assert (
        optimizers
        .critic
        .param_groups[0]["lr"]
        == 2.0e-4
    )

def test_ppo_update_changes_actor_and_critic_parameters():
    device = preferred_device()

    (
        actor,
        critic,
        batch,
        advantage,
        target_return,
    ) = build_paper_shaped_update_inputs(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    actor_before = snapshot_parameters(
        actor
    )

    critic_before = snapshot_parameters(
        critic
    )

    result = perform_ppo_update(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        batch=batch,
        advantage=advantage,
        target_return=target_return,
        loss_config=PPOLossConfig(
            #
            # PAPER-UNSPECIFIED.
            #
            # Zero isolates the PPO policy objective
            # in this optimizer unit test.
            #
            entropy_coefficient=0.0,
        ),
    )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
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
        result.actor_loss
    )

    assert torch.isfinite(
        result.critic_loss
    )

def test_first_ppo_update_starts_from_old_policy_ratio_one():
    device = preferred_device()

    (
        actor,
        critic,
        batch,
        advantage,
        target_return,
    ) = build_paper_shaped_update_inputs(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    result = perform_ppo_update(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        batch=batch,
        advantage=advantage,
        target_return=target_return,
        loss_config=PPOLossConfig(
            entropy_coefficient=0.0,
        ),
    )

    torch.testing.assert_close(
        result.mean_probability_ratio,
        torch.tensor(
            1.0,
            dtype=torch.float32,
            device=device,
        ),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_ppo_update_does_not_modify_rollout_targets():
    device = preferred_device()

    (
        actor,
        critic,
        batch,
        advantage,
        target_return,
    ) = build_paper_shaped_update_inputs(
        device
    )

    old_log_prob_before = (
        batch
        .old_joint_log_prob
        .clone()
    )

    advantage_before = (
        advantage.clone()
    )

    target_return_before = (
        target_return.clone()
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    perform_ppo_update(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        batch=batch,
        advantage=advantage,
        target_return=target_return,
        loss_config=PPOLossConfig(
            entropy_coefficient=0.0,
        ),
    )

    torch.testing.assert_close(
        batch.old_joint_log_prob,
        old_log_prob_before,
    )

    torch.testing.assert_close(
        advantage,
        advantage_before,
    )

    torch.testing.assert_close(
        target_return,
        target_return_before,
    )


def test_adam_moment_state_follows_model_device_after_update():
    device = preferred_device()

    (
        actor,
        critic,
        batch,
        advantage,
        target_return,
    ) = build_paper_shaped_update_inputs(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    perform_ppo_update(
        actor=actor,
        critic=critic,
        optimizers=optimizers,
        batch=batch,
        advantage=advantage,
        target_return=target_return,
        loss_config=PPOLossConfig(
            entropy_coefficient=0.0,
        ),
    )

    for optimizer in (
        optimizers.actor,
        optimizers.critic,
    ):
        assert len(
            optimizer.state
        ) > 0

        for state in optimizer.state.values():
            assert "step" in state
            assert "exp_avg" in state
            assert "exp_avg_sq" in state

            #
            # Adam's moment tensors participate in
            # parameter updates and must follow the
            # model onto cuda:0.
            #
            assert (
                state["exp_avg"].device
                == device
            )

            assert (
                state["exp_avg_sq"].device
                == device
            )

            #
            # AMSGrad is disabled in our current
            # reproduction, but keep this check
            # correct if it is enabled later.
            #
            if "max_exp_avg_sq" in state:
                assert (
                    state[
                        "max_exp_avg_sq"
                    ].device
                    == device
                )

            #
            # PyTorch Adam may intentionally keep
            # its scalar step counter on CPU when
            # capturable=False. Its device is not
            # evidence that optimization happened
            # on CPU.
            #
            step = state["step"]

            assert isinstance(
                step,
                torch.Tensor,
            )

            assert step.ndim == 0

            assert torch.isfinite(
                step
            )

def test_ppo_update_rejects_device_mismatch():
    if not torch.cuda.is_available():
        return

    device = torch.device(
        "cuda:0"
    )

    (
        actor,
        critic,
        batch,
        advantage,
        target_return,
    ) = build_paper_shaped_update_inputs(
        device
    )

    optimizers = create_ppo_optimizers(
        actor=actor,
        critic=critic,
        config=PPOOptimizerConfig(),
    )

    with torch.no_grad():
        cpu_states = (
            batch
            .states
            .cpu()
        )

    broken_batch = PPORolloutBatch(
        states=cpu_states,
        actions=batch.actions,
        action_masks=batch.action_masks,
        old_log_prob_by_rbg=(
            batch.old_log_prob_by_rbg
        ),
        old_joint_log_prob=(
            batch.old_joint_log_prob
        ),
        reward_by_rbg=(
            batch.reward_by_rbg
        ),
        reduced_reward=(
            batch.reduced_reward
        ),
        old_value=batch.old_value,
        next_states=batch.next_states,
        terminated=batch.terminated,
    )

    try:
        perform_ppo_update(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            batch=broken_batch,
            advantage=advantage,
            target_return=target_return,
            loss_config=PPOLossConfig(
                entropy_coefficient=0.0,
            ),
        )

    except ValueError:
        return

    raise AssertionError(
        "Device mismatch must raise ValueError."
    )

