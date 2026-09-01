import torch

from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
    build_greedy_indicator,
    compute_geometric_mean_throughput,
    compute_ppo_reward,
)

from oran_scheduler.rl.ppo_reward import (
    reduce_ppo_rbg_rewards,
)


def test_geometric_mean_throughput():
    throughput = torch.tensor(
        [
            8.0,
            8.0,
            2.0,
        ],
        dtype=torch.float32,
    )

    valid_mask = torch.tensor(
        [
            True,
            True,
            True,
        ],
        dtype=torch.bool,
    )

    geometric_mean = (
        compute_geometric_mean_throughput(
            throughput_bps=throughput,
            valid_ue_mask=valid_mask,
        )
    )

    expected = torch.tensor(
        128.0 ** (1.0 / 3.0),
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        geometric_mean,
        expected,
    )


def test_geometric_mean_ignores_padding():
    throughput = torch.tensor(
        [
            8.0,
            8.0,
            2.0,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    valid_mask = torch.tensor(
        [
            True,
            True,
            True,
            False,
            False,
        ],
        dtype=torch.bool,
    )

    geometric_mean = (
        compute_geometric_mean_throughput(
            throughput_bps=throughput,
            valid_ue_mask=valid_mask,
        )
    )

    expected = torch.tensor(
        128.0 ** (1.0 / 3.0),
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        geometric_mean,
        expected,
    )

def test_real_zero_throughput_gives_zero_geometric_mean():
    throughput = torch.tensor(
        [
            8.0,
            0.0,
            2.0,
        ],
        dtype=torch.float32,
    )

    valid_mask = torch.tensor(
        [
            True,
            True,
            True,
        ],
        dtype=torch.bool,
    )

    geometric_mean = (
        compute_geometric_mean_throughput(
            throughput_bps=throughput,
            valid_ue_mask=valid_mask,
        )
    )

    assert float(
        geometric_mean.item()
    ) == 0.0


def test_build_greedy_indicator():
    better_exists = torch.tensor(
        [
            [
                False,
                True,
                False,
            ],
            [
                True,
                False,
                False,
            ],
        ],
        dtype=torch.bool,
    )

    indicator = build_greedy_indicator(
        better_allocation_exists=(
            better_exists
        ),
        dtype=torch.float32,
    )

    expected = torch.tensor(
        [
            [
                1.0,
                -1.0,
                1.0,
            ],
            [
                -1.0,
                1.0,
                1.0,
            ],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        indicator,
        expected,
    )


def test_compute_complete_ppo_reward_matrix():
    throughput = torch.tensor(
        [
            6.0,
            6.0,
            6.0,
        ],
        dtype=torch.float32,
    )

    valid_mask = torch.tensor(
        [
            True,
            True,
            True,
        ],
        dtype=torch.bool,
    )

    better_exists = torch.tensor(
        [
            [
                False,
                True,
                False,
                False,
            ],
            [
                False,
                False,
                True,
                False,
            ],
            [
                True,
                False,
                False,
                False,
            ],
        ],
        dtype=torch.bool,
    )

    result = compute_ppo_reward(
        throughput_bps=throughput,
        valid_ue_mask=valid_mask,
        better_allocation_exists=(
            better_exists
        ),
        config=PPORewardConfig(
            geometric_mean_normalizer_bps=10.0,
        ),
    )

    expected_indicator = torch.tensor(
        [
            [
                1.0,
                -1.0,
                1.0,
                1.0,
            ],
            [
                1.0,
                1.0,
                -1.0,
                1.0,
            ],
            [
                -1.0,
                1.0,
                1.0,
                1.0,
            ],
        ],
        dtype=torch.float32,
    )

    expected_reward = torch.tensor(
        [
            [
                0.6,
                -0.6,
                0.6,
                0.6,
            ],
            [
                0.2,
                0.2,
                -0.2,
                0.2,
            ],
            [
                -0.2,
                0.2,
                0.2,
                0.2,
            ],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        result.greedy_indicator,
        expected_indicator,
    )

    torch.testing.assert_close(
        result.geometric_mean_throughput_bps,
        torch.tensor(
            6.0,
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        result.normalized_geometric_mean,
        torch.tensor(
            0.6,
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        result.reward_by_layer_rbg,
        expected_reward,
    )

    assert tuple(
        result.reward_by_layer_rbg.shape
    ) == (
        3,
        4,
    )


def test_reward_rejects_negative_throughput():
    throughput = torch.tensor(
        [
            5.0,
            -1.0,
        ],
        dtype=torch.float32,
    )

    valid_mask = torch.tensor(
        [
            True,
            True,
        ],
        dtype=torch.bool,
    )

    better_exists = torch.zeros(
        (
            2,
            3,
        ),
        dtype=torch.bool,
    )

    try:
        compute_ppo_reward(
            throughput_bps=throughput,
            valid_ue_mask=valid_mask,
            better_allocation_exists=(
                better_exists
            ),
            config=PPORewardConfig(
                geometric_mean_normalizer_bps=10.0,
            ),
        )

    except ValueError:
        return

    raise AssertionError(
        "Negative throughput should raise ValueError."
    )



def test_reduce_ppo_rbg_rewards_with_mean():
    reward_by_rbg = torch.tensor(
        [
            [
                0.2,
                0.2,
                -0.2,
                0.2,
            ],
            [
                -0.2,
                -0.2,
                0.2,
                0.2,
            ],
        ],
        dtype=torch.float32,
    )

    reward_by_layer = (
        reduce_ppo_rbg_rewards(
            reward_by_rbg,
            reduction="mean",
        )
    )

    expected = torch.tensor(
        [
            0.1,
            0.0,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        reward_by_layer,
        expected,
    )

def test_reduce_ppo_rbg_rewards_with_sum():
    reward_by_rbg = torch.tensor(
        [
            [
                0.2,
                0.2,
                -0.2,
                0.2,
            ]
        ],
        dtype=torch.float32,
    )

    reward_by_layer = (
        reduce_ppo_rbg_rewards(
            reward_by_rbg,
            reduction="sum",
        )
    )

    expected = torch.tensor(
        [
            0.4,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        reward_by_layer,
        expected,
    )

def test_reduce_ppo_rbg_rewards_rejects_unknown_reduction():
    reward_by_rbg = torch.zeros(
        (
            2,
            18,
        ),
        dtype=torch.float32,
    )

    try:
        reduce_ppo_rbg_rewards(
            reward_by_rbg,
            reduction="median",
        )

    except ValueError:
        return

    raise AssertionError(
        "Unknown PPO reward reduction must "
        "raise ValueError."
    )







