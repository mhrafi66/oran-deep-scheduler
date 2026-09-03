import pytest
import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)
from oran_scheduler.rl.ppo_physical_score import (
    PPOPhysicalScoreInputs,
)
from oran_scheduler.rl.ppo_physical_tti import (
    evaluate_ppo_physical_tti,
    resolve_ppo_tti_reward,
)
from oran_scheduler.rl.ppo_reward import (
    PPORewardConfig,
)
from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)


def build_physical_case(
    device: str = "cuda:0",
):
    num_rbgs = 2

    h_freq = torch.zeros(
        (
            1,   # batch
            2,   # global UEs
            1,   # RX antenna
            1,   # BS
            2,   # TX antennas
            1,   # OFDM symbol
            24,  # 2 RBG x 12 SC
        ),
        dtype=torch.complex64,
        device=device,
    )

    #
    # UE 0 primarily occupies TX direction 0.
    #
    h_freq[
        0,
        0,
        0,
        0,
        0,
        0,
        :,
    ] = 1.0

    #
    # UE 1 primarily occupies TX direction 1.
    #
    h_freq[
        0,
        1,
        0,
        0,
        1,
        0,
        :,
    ] = 1.0

    candidate_global_ue_indices = (
        torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
            device=device,
        )
    )

    recommended_rank = torch.tensor(
        [
            [
                1,
                1,
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    rx_combiners = torch.ones(
        (
            1,
            2,
            num_rbgs,
            2,
            1,
        ),
        dtype=torch.complex64,
        device=device,
    )

    #
    # K = 2 candidates.
    #
    # Actor action 2 therefore means NO ALLOCATION.
    #
    actions = torch.tensor(
        [
            [
                0,
                0,
            ],
            [
                2,
                1,
            ],
        ],
        dtype=torch.long,
        device=device,
    )

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.tensor(
                [
                    [
                        0,
                        0,
                    ],
                    [
                        -1,
                        1,
                    ],
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    physical_inputs = (
        PPOPhysicalScoreInputs(
            candidate_global_ue_indices=(
                candidate_global_ue_indices
            ),
            h_freq=h_freq,
            serving_cell_index=0,
            recommended_rank=(
                recommended_rank
            ),
            rx_combiners=(
                rx_combiners
            ),
            csi_subcarrier_index=6,
            subcarriers_per_rbg=12,
            tx_power_per_subcarrier_w=2.0,
            noise_power_per_subcarrier_w=(
                1.0e-3
            ),
            link_adaptation_config=(
                LinkAdaptationConfig(
                    num_rbgs=num_rbgs,
                    subcarriers_per_rbg=12,
                    device=device,
                )
            ),
            rate_config=RateConfig(
                subcarriers_per_rbg=12,
                device=device,
            ),
        )
    )

    past_average_throughput = (
        torch.tensor(
            [
                1.0e6,
                1.0e6,
            ],
            dtype=torch.float32,
            device=device,
        )
    )

    candidate_valid_mask = torch.ones(
        2,
        dtype=torch.bool,
        device=device,
    )

    return (
        allocation,
        actions,
        past_average_throughput,
        candidate_valid_mask,
        physical_inputs,
    )


def test_real_physical_tti_outcome():
    (
        allocation,
        actions,
        past_average,
        candidate_valid_mask,
        physical_inputs,
    ) = build_physical_case()

    outcome = evaluate_ppo_physical_tti(
        allocation=allocation,
        actions=actions,
        past_average_throughput=(
            past_average
        ),
        candidate_valid_mask=(
            candidate_valid_mask
        ),
        physical_inputs=physical_inputs,
        greedy_config=(
            PPOGreedySearchConfig()
        ),
    )

    assert tuple(
        outcome
        .candidate_total_target_compliant_rate_bps
        .shape
    ) == (
        2,
    )

    assert torch.isfinite(
        outcome
        .candidate_total_target_compliant_rate_bps
    ).all()

    assert torch.all(
        outcome
        .candidate_total_target_compliant_rate_bps
        >= 0.0
    )

    torch.testing.assert_close(
        outcome
        .candidate_total_target_compliant_rate_bps
        .sum(),
        outcome
        .allocation_evaluation
        .total_cell_target_compliant_rate_bps,
    )

    assert tuple(
        outcome
        .greedy_search
        .better_allocation_exists
        .shape
    ) == (
        2,
        2,
    )

    torch.testing.assert_close(
        outcome
        .greedy_search
        .allocation
        .candidate_by_user_slot,
        allocation
        .candidate_by_user_slot,
    )

    assert (
        outcome.num_score_requests
        > 0
    )

    assert (
        outcome
        .num_unique_phy_evaluations
        > 0
    )

    assert (
        outcome
        .num_unique_phy_evaluations
        <= outcome.num_score_requests
    )


def test_actor_actions_must_match_final_allocation():
    (
        allocation,
        actions,
        past_average,
        candidate_valid_mask,
        physical_inputs,
    ) = build_physical_case()

    wrong_actions = actions.clone()

    wrong_actions[
        1,
        1,
    ] = 2

    with pytest.raises(
        ValueError,
        match="reconstruct",
    ):
        evaluate_ppo_physical_tti(
            allocation=allocation,
            actions=wrong_actions,
            past_average_throughput=(
                past_average
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
            physical_inputs=(
                physical_inputs
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
        )


def test_actor_cannot_select_invalid_candidate():
    (
        _,
        _,
        past_average,
        _,
        physical_inputs,
    ) = build_physical_case()

    candidate_valid_mask = torch.tensor(
        [
            True,
            False,
        ],
        dtype=torch.bool,
        device="cuda:0",
    )

    actions = torch.tensor(
        [
            [
                0,
                1,
            ]
        ],
        dtype=torch.long,
        device="cuda:0",
    )

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.tensor(
                [
                    [
                        0,
                        1,
                    ]
                ],
                dtype=torch.long,
                device="cuda:0",
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="invalid",
    ):
        evaluate_ppo_physical_tti(
            allocation=allocation,
            actions=actions,
            past_average_throughput=(
                past_average
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
            physical_inputs=(
                physical_inputs
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
        )


def test_reward_population_is_explicit():
    (
        allocation,
        actions,
        past_average,
        candidate_valid_mask,
        physical_inputs,
    ) = build_physical_case()

    outcome = evaluate_ppo_physical_tti(
        allocation=allocation,
        actions=actions,
        past_average_throughput=(
            past_average
        ),
        candidate_valid_mask=(
            candidate_valid_mask
        ),
        physical_inputs=physical_inputs,
        greedy_config=(
            PPOGreedySearchConfig()
        ),
    )

    #
    # Interpretation A:
    # only these two UEs belong to the reward
    # population.
    #
    candidate_reward = (
        resolve_ppo_tti_reward(
            physical_outcome=outcome,
            reward_throughput_bps=(
                torch.tensor(
                    [
                        8.0,
                        2.0,
                    ],
                    dtype=torch.float32,
                    device="cuda:0",
                )
            ),
            reward_valid_ue_mask=(
                torch.tensor(
                    [
                        True,
                        True,
                    ],
                    dtype=torch.bool,
                    device="cuda:0",
                )
            ),
            reward_population_name=(
                "candidate_ues"
            ),
            reward_config=PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    10.0
                ),
            ),
            reward_reduction="mean",
        )
    )

    #
    # G = sqrt(8 * 2) = 4.
    #
    torch.testing.assert_close(
        candidate_reward
        .reward_data
        .geometric_mean_throughput_bps,
        torch.tensor(
            4.0,
            dtype=torch.float32,
            device="cuda:0",
        ),
    )

    torch.testing.assert_close(
        candidate_reward
        .reward_data
        .normalized_geometric_mean,
        torch.tensor(
            0.4,
            dtype=torch.float32,
            device="cuda:0",
        ),
    )

    assert tuple(
        candidate_reward
        .reward_data
        .reward_by_layer_rbg
        .shape
    ) == (
        2,
        2,
    )

    assert tuple(
        candidate_reward
        .reduced_reward
        .shape
    ) == (
        2,
    )

    torch.testing.assert_close(
        torch.abs(
            candidate_reward
            .reward_data
            .reward_by_layer_rbg[
                0,
                :,
            ]
        ),
        torch.full(
            (
                2,
            ),
            0.4,
            dtype=torch.float32,
            device="cuda:0",
        ),
    )

    torch.testing.assert_close(
        torch.abs(
            candidate_reward
            .reward_data
            .reward_by_layer_rbg[
                1,
                :,
            ]
        ),
        torch.full(
            (
                2,
            ),
            0.2,
            dtype=torch.float32,
            device="cuda:0",
        ),
    )

    assert (
        candidate_reward
        .reward_population_name
        == "candidate_ues"
    )

    #
    # Interpretation B:
    # suppose a third valid cell UE also belongs to G,
    # but received zero throughput.
    #
    serving_reward = (
        resolve_ppo_tti_reward(
            physical_outcome=outcome,
            reward_throughput_bps=(
                torch.tensor(
                    [
                        8.0,
                        2.0,
                        0.0,
                    ],
                    dtype=torch.float32,
                    device="cuda:0",
                )
            ),
            reward_valid_ue_mask=(
                torch.tensor(
                    [
                        True,
                        True,
                        True,
                    ],
                    dtype=torch.bool,
                    device="cuda:0",
                )
            ),
            reward_population_name=(
                "serving_cell_ues"
            ),
            reward_config=PPORewardConfig(
                geometric_mean_normalizer_bps=(
                    10.0
                ),
            ),
            reward_reduction="mean",
        )
    )

    torch.testing.assert_close(
        serving_reward
        .reward_data
        .geometric_mean_throughput_bps,
        torch.tensor(
            0.0,
            dtype=torch.float32,
            device="cuda:0",
        ),
    )

    torch.testing.assert_close(
        serving_reward
        .reward_data
        .reward_by_layer_rbg[
            0,
            :,
        ],
        torch.zeros(
            2,
            dtype=torch.float32,
            device="cuda:0",
        ),
    )


