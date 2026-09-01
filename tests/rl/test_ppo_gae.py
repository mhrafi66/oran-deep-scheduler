import pytest
import torch

from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
    compute_ppo_gae,
)

def test_compute_ppo_gae_matches_numerical_example():
    config = PPOGAEConfig(
        gamma=0.95,
        gae_lambda=0.9,
    )

    rewards = torch.tensor(
        [
            0.1,
            -0.2,
            0.2,
        ],
        dtype=torch.float32,
    )

    values = torch.tensor(
        [
            0.05,
            0.02,
            0.10,
        ],
        dtype=torch.float32,
    )

    next_value = torch.tensor(
        0.08,
        dtype=torch.float32,
    )

    terminated = torch.tensor(
        [
            False,
            False,
            False,
        ],
        dtype=torch.bool,
    )

    result = compute_ppo_gae(
        rewards=rewards,
        values=values,
        next_value=next_value,
        terminated=terminated,
        config=config,
    )

    expected_td_residual = torch.tensor(
        [
            0.069,
            -0.125,
            0.176,
        ],
        dtype=torch.float32,
    )

    expected_advantage = torch.tensor(
        [
            0.0907854,
            0.02548,
            0.176,
        ],
        dtype=torch.float32,
    )

    expected_target_return = torch.tensor(
        [
            0.1407854,
            0.04548,
            0.276,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        result.td_residual,
        expected_td_residual,
    )

    torch.testing.assert_close(
        result.advantage,
        expected_advantage,
    )

    torch.testing.assert_close(
        result.target_return,
        expected_target_return,
    )


def test_terminal_transition_does_not_bootstrap():
    config = PPOGAEConfig(
        gamma=0.95,
        gae_lambda=0.9,
    )

    rewards = torch.tensor(
        [
            1.0,
            2.0,
        ],
        dtype=torch.float32,
    )

    values = torch.tensor(
        [
            0.5,
            0.7,
        ],
        dtype=torch.float32,
    )

    # Deliberately huge.
    #
    # It MUST NOT affect the final transition because
    # that transition is terminal.
    next_value = torch.tensor(
        100.0,
        dtype=torch.float32,
    )

    terminated = torch.tensor(
        [
            False,
            True,
        ],
        dtype=torch.bool,
    )

    result = compute_ppo_gae(
        rewards=rewards,
        values=values,
        next_value=next_value,
        terminated=terminated,
        config=config,
    )

    expected_td_residual = torch.tensor(
        [
            1.165,
            1.3,
        ],
        dtype=torch.float32,
    )

    expected_advantage = torch.tensor(
        [
            2.2765,
            1.3,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        result.td_residual,
        expected_td_residual,
    )

    torch.testing.assert_close(
        result.advantage,
        expected_advantage,
    )

def test_lambda_zero_reduces_gae_to_one_step_td_error():
    config = PPOGAEConfig(
        gamma=0.95,
        gae_lambda=0.0,
    )

    rewards = torch.tensor(
        [
            0.1,
            0.2,
        ],
        dtype=torch.float32,
    )

    values = torch.tensor(
        [
            0.05,
            0.07,
        ],
        dtype=torch.float32,
    )

    next_value = torch.tensor(
        0.08,
        dtype=torch.float32,
    )

    terminated = torch.tensor(
        [
            False,
            False,
        ],
        dtype=torch.bool,
    )

    result = compute_ppo_gae(
        rewards=rewards,
        values=values,
        next_value=next_value,
        terminated=terminated,
        config=config,
    )

    torch.testing.assert_close(
        result.advantage,
        result.td_residual,
    )


def test_target_return_equals_advantage_plus_value():
    config = PPOGAEConfig(
        gamma=0.95,
        gae_lambda=0.8,
    )

    rewards = torch.tensor(
        [
            0.2,
            -0.1,
            0.3,
        ],
        dtype=torch.float32,
    )

    values = torch.tensor(
        [
            0.1,
            0.15,
            0.12,
        ],
        dtype=torch.float32,
    )

    next_value = torch.tensor(
        0.09,
        dtype=torch.float32,
    )

    terminated = torch.zeros(
        3,
        dtype=torch.bool,
    )

    result = compute_ppo_gae(
        rewards=rewards,
        values=values,
        next_value=next_value,
        terminated=terminated,
        config=config,
    )

    torch.testing.assert_close(
        result.target_return,
        result.advantage + values,
    )

def test_gae_config_rejects_invalid_parameters():
    with pytest.raises(
        ValueError
    ):
        PPOGAEConfig(
            gae_lambda=-0.1
        )

    with pytest.raises(
        ValueError
    ):
        PPOGAEConfig(
            gae_lambda=0.5,
            gamma=1.1,
        )


def test_gae_rejects_mismatched_transition_shapes():
    config = PPOGAEConfig(
        gae_lambda=0.9
    )

    rewards = torch.zeros(
        3,
        dtype=torch.float32,
    )

    values = torch.zeros(
        3,
        dtype=torch.float32,
    )

    next_value = torch.tensor(
        0.0,
        dtype=torch.float32,
    )

    terminated = torch.zeros(
        2,
        dtype=torch.bool,
    )

    with pytest.raises(
        ValueError
    ):
        compute_ppo_gae(
            rewards=rewards,
            values=values,
            next_value=next_value,
            terminated=terminated,
            config=config,
        )

