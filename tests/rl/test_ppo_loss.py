import math

import pytest
import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
    compute_joint_action_log_prob,
)
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
    compute_ppo_actor_loss,
    compute_ppo_clipped_policy_objective,
    compute_ppo_critic_loss,
    reduce_ppo_entropy_by_rbg,
)


def test_clipped_policy_objective_matches_numerical_example():
    ratios = torch.tensor(
        [
            1.5,
            0.5,
            1.5,
            0.5,
        ],
        dtype=torch.float32,
    )

    current_log_prob = torch.log(
        ratios
    )

    old_log_prob = torch.zeros(
        4,
        dtype=torch.float32,
    )

    advantage = torch.tensor(
        [
            1.0,
            1.0,
            -1.0,
            -1.0,
        ],
        dtype=torch.float32,
    )

    result = (
        compute_ppo_clipped_policy_objective(
            current_joint_log_prob=(
                current_log_prob
            ),
            old_joint_log_prob=(
                old_log_prob
            ),
            advantage=advantage,
            clip_epsilon=0.2,
        )
    )

    expected_ratio = ratios

    expected_clipped_ratio = torch.tensor(
        [
            1.2,
            0.8,
            1.2,
            0.8,
        ],
        dtype=torch.float32,
    )

    expected_surrogate = torch.tensor(
        [
            1.2,
            0.5,
            -1.5,
            -0.8,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        result.ratio,
        expected_ratio,
    )

    torch.testing.assert_close(
        result.clipped_ratio,
        expected_clipped_ratio,
    )

    torch.testing.assert_close(
        result.surrogate,
        expected_surrogate,
    )

    torch.testing.assert_close(
        result.objective,
        expected_surrogate.mean(),
    )


def test_entropy_reduction_supports_joint_sum_and_mean():
    entropy_by_rbg = torch.tensor(
        [
            [
                0.5,
                1.0,
                1.5,
            ],
            [
                0.2,
                0.4,
                0.6,
            ],
        ],
        dtype=torch.float32,
    )

    summed = reduce_ppo_entropy_by_rbg(
        entropy_by_rbg,
        reduction="sum",
    )

    averaged = reduce_ppo_entropy_by_rbg(
        entropy_by_rbg,
        reduction="mean",
    )

    torch.testing.assert_close(
        summed,
        torch.tensor(
            [
                3.0,
                1.2,
            ]
        ),
    )

    torch.testing.assert_close(
        averaged,
        torch.tensor(
            [
                1.0,
                0.4,
            ]
        ),
    )


def test_actor_ratio_is_one_before_policy_changes():
    torch.manual_seed(
        123
    )

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=4,
            hidden_size=3,
            num_rbgs=2,
            num_actions_per_rbg=3,
        )
    )

    state = torch.tensor(
        [
            [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            [
                0.4,
                0.3,
                0.2,
                0.1,
            ],
        ],
        dtype=torch.float32,
    )

    action_mask = torch.tensor(
        [
            [
                [
                    True,
                    True,
                    True,
                ],
                [
                    True,
                    False,
                    True,
                ],
            ],
            [
                [
                    True,
                    True,
                    False,
                ],
                [
                    True,
                    True,
                    True,
                ],
            ],
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
        sampled = actor.sample_actions(
            state=state,
            action_mask=action_mask,
        )

        old_log_prob_by_rbg = (
            sampled
            .log_prob_by_rbg
            .clone()
        )

        old_joint_log_prob = (
            compute_joint_action_log_prob(
                old_log_prob_by_rbg
            )
        )

    result = compute_ppo_actor_loss(
        actor=actor,
        state=state,
        action_mask=action_mask,
        actions=sampled.actions,
        old_log_prob_by_rbg=(
            old_log_prob_by_rbg
        ),
        old_joint_log_prob=(
            old_joint_log_prob
        ),
        advantage=torch.tensor(
            [
                1.0,
                -0.5,
            ],
            dtype=torch.float32,
        ),
        config=PPOLossConfig(
            entropy_coefficient=0.0,
        ),
    )

    torch.testing.assert_close(
        result.policy.ratio,
        torch.ones(
            2,
            dtype=torch.float32,
        ),
    )


def test_actor_loss_detaches_old_policy_and_advantage():
    torch.manual_seed(
        7
    )

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=4,
            hidden_size=4,
            num_rbgs=2,
            num_actions_per_rbg=3,
        )
    )

    state = torch.tensor(
        [
            [
                1.0,
                0.5,
                0.25,
                0.75,
            ],
        ],
        dtype=torch.float32,
    )

    action_mask = torch.ones(
        (
            1,
            2,
            3,
        ),
        dtype=torch.bool,
    )

    with torch.no_grad():
        sampled = actor.sample_actions(
            state=state,
            action_mask=action_mask,
        )

    old_log_prob_by_rbg = (
        sampled
        .log_prob_by_rbg
        .detach()
        .clone()
        .requires_grad_(
            True
        )
    )

    old_joint_log_prob = (
        old_log_prob_by_rbg.sum(
            dim=-1
        )
    )

    advantage = torch.tensor(
        [
            1.0,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    result = compute_ppo_actor_loss(
        actor=actor,
        state=state,
        action_mask=action_mask,
        actions=sampled.actions,
        old_log_prob_by_rbg=(
            old_log_prob_by_rbg
        ),
        old_joint_log_prob=(
            old_joint_log_prob
        ),
        advantage=advantage,
        config=PPOLossConfig(
            entropy_coefficient=0.0,
        ),
    )

    result.loss.backward()

    assert (
        old_log_prob_by_rbg.grad
        is None
    )

    assert advantage.grad is None

    actor_gradients = [
        parameter.grad
        for parameter
        in actor.parameters()
        if parameter.requires_grad
    ]

    assert any(
        gradient is not None
        and torch.any(
            gradient != 0.0
        )
        for gradient in actor_gradients
    )

class LinearCritic(
    nn.Module
):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            1,
            1,
            bias=False,
        )

        with torch.no_grad():
            self.linear.weight.fill_(
                1.0
            )

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.linear(
                state
            )
            .squeeze(
                -1
            )
        )


def test_critic_loss_matches_value_equation_and_detaches_target():
    critic = LinearCritic()

    state = torch.tensor(
        [
            [
                0.5,
            ],
            [
                -0.5,
            ],
        ],
        dtype=torch.float32,
    )

    target_return = torch.tensor(
        [
            1.5,
            -2.5,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    result = compute_ppo_critic_loss(
        critic=critic,
        state=state,
        target_return=target_return,
    )

    expected_squared_error = torch.tensor(
        [
            1.0,
            4.0,
        ],
        dtype=torch.float32,
    )

    expected_loss = torch.tensor(
        1.25,
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        result.squared_error,
        expected_squared_error,
    )

    torch.testing.assert_close(
        result.loss,
        expected_loss,
    )

    result.loss.backward()

    assert target_return.grad is None

    assert (
        critic.linear.weight.grad
        is not None
    )

    assert torch.any(
        critic.linear.weight.grad
        != 0.0
    )

def test_actor_loss_rejects_inconsistent_old_joint_log_prob():
    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=4,
            hidden_size=3,
            num_rbgs=2,
            num_actions_per_rbg=3,
        )
    )

    state = torch.zeros(
        (
            1,
            4,
        ),
        dtype=torch.float32,
    )

    action_mask = torch.ones(
        (
            1,
            2,
            3,
        ),
        dtype=torch.bool,
    )

    actions = torch.zeros(
        (
            1,
            2,
        ),
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError
    ):
        compute_ppo_actor_loss(
            actor=actor,
            state=state,
            action_mask=action_mask,
            actions=actions,
            old_log_prob_by_rbg=torch.tensor(
                [
                    [
                        -0.2,
                        -0.3,
                    ]
                ],
                dtype=torch.float32,
            ),
            old_joint_log_prob=torch.tensor(
                [
                    -99.0,
                ],
                dtype=torch.float32,
            ),
            advantage=torch.tensor(
                [
                    1.0,
                ],
                dtype=torch.float32,
            ),
            config=PPOLossConfig(
                entropy_coefficient=0.0,
            ),
        )

def test_ppo_loss_config_rejects_invalid_values():
    with pytest.raises(
        ValueError
    ):
        PPOLossConfig(
            entropy_coefficient=-0.1,
        )

    with pytest.raises(
        ValueError
    ):
        PPOLossConfig(
            entropy_coefficient=0.0,
            clip_epsilon=1.2,
        )

    with pytest.raises(
        ValueError
    ):
        PPOLossConfig(
            entropy_coefficient=0.0,
            entropy_reduction="invalid",
        )


