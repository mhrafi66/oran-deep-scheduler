import pytest
import torch
from dataclasses import replace

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
    CachedPPOPhysicalRBGScorer,
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


def test_ppo_physical_tti_supports_local_phy_ue_indices():
    (
        allocation,
        actions,
        past_average,
        candidate_valid_mask,
        physical_inputs,
    ) = build_physical_case()

    device = (
        physical_inputs
        .candidate_global_ue_indices
        .device
    )

    num_candidates = int(
        physical_inputs
        .candidate_global_ue_indices
        .shape[0]
    )

    # ==========================================================
    # BASELINE
    #
    # Existing/full-tensor convention:
    #
    #     global UE identity
    #         ==
    #     physical UE storage position
    # ==========================================================

    baseline_outcome = (
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
    )


    # ==========================================================
    # MEMORY-SAFE MAPPING
    #
    # The SAME physical channels are locally stored at:
    #
    #     0, 1, 2, ...
    #
    # but the scheduler now knows those UEs as:
    #
    #     100, 101, 102, ...
    #
    # Physical results must therefore remain identical.
    # ==========================================================

    global_ids = (
        torch.arange(
            num_candidates,
            dtype=torch.long,
            device=device,
        )
        + 100
    )

    local_phy_ids = torch.arange(
        num_candidates,
        dtype=torch.long,
        device=device,
    )

    mapped_physical_inputs = replace(
        physical_inputs,

        candidate_global_ue_indices=(
            global_ids
        ),

        candidate_physical_ue_indices=(
            local_phy_ids
        ),
    )

    mapped_outcome = (
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
                mapped_physical_inputs
            ),
            greedy_config=(
                PPOGreedySearchConfig()
            ),
        )
    )


    # ==========================================================
    # BASIC VALIDATION
    # ==========================================================

    assert tuple(
        mapped_outcome
        .candidate_total_target_compliant_rate_bps
        .shape
    ) == (
        num_candidates,
    )

    assert torch.isfinite(
        mapped_outcome
        .candidate_total_target_compliant_rate_bps
    ).all()


    # ==========================================================
    # FINAL ACTOR ALLOCATION PHY
    #
    # Changing persistent UE IDs must NOT change any
    # physical result when the local physical mapping
    # still points to exactly the same channel entries.
    # ==========================================================

    torch.testing.assert_close(
        mapped_outcome
        .allocation_evaluation
        .candidate_nominal_rate_bps,

        baseline_outcome
        .allocation_evaluation
        .candidate_nominal_rate_bps,
    )

    torch.testing.assert_close(
        mapped_outcome
        .allocation_evaluation
        .candidate_expected_goodput_bps,

        baseline_outcome
        .allocation_evaluation
        .candidate_expected_goodput_bps,
    )

    torch.testing.assert_close(
        mapped_outcome
        .allocation_evaluation
        .candidate_target_compliant_rate_bps,

        baseline_outcome
        .allocation_evaluation
        .candidate_target_compliant_rate_bps,
    )

    torch.testing.assert_close(
        mapped_outcome
        .allocation_evaluation
        .total_rbg_target_compliant_rate_bps,

        baseline_outcome
        .allocation_evaluation
        .total_rbg_target_compliant_rate_bps,
    )

    torch.testing.assert_close(
        mapped_outcome
        .allocation_evaluation
        .total_cell_target_compliant_rate_bps,

        baseline_outcome
        .allocation_evaluation
        .total_cell_target_compliant_rate_bps,
    )

    torch.testing.assert_close(
        mapped_outcome
        .allocation_evaluation
        .num_scheduled_ues_per_rbg,

        baseline_outcome
        .allocation_evaluation
        .num_scheduled_ues_per_rbg,
    )

    torch.testing.assert_close(
        mapped_outcome
        .allocation_evaluation
        .num_physical_layers_per_rbg,

        baseline_outcome
        .allocation_evaluation
        .num_physical_layers_per_rbg,
    )

    torch.testing.assert_close(
        mapped_outcome
        .allocation_evaluation
        .rzf_alpha_by_rbg,

        baseline_outcome
        .allocation_evaluation
        .rzf_alpha_by_rbg,
    )


    # ==========================================================
    # PPO COUNTERFACTUAL SEARCH
    #
    # This is especially important.
    #
    # The local mapping must propagate not only through
    # the actor's final allocation evaluation, but also
    # through every counterfactual PHY call used by the
    # PPO greedy reward judge.
    # ==========================================================

    torch.testing.assert_close(
        mapped_outcome
        .greedy_search
        .chosen_pf_sum,

        baseline_outcome
        .greedy_search
        .chosen_pf_sum,
    )

    torch.testing.assert_close(
        mapped_outcome
        .greedy_search
        .best_pf_sum,

        baseline_outcome
        .greedy_search
        .best_pf_sum,
    )

    torch.testing.assert_close(
        mapped_outcome
        .greedy_search
        .better_allocation_exists,

        baseline_outcome
        .greedy_search
        .better_allocation_exists,
    )


    # Both paths should require the same counterfactual
    # physical work.
    assert (
        mapped_outcome.num_score_requests
        ==
        baseline_outcome.num_score_requests
    )

    assert (
        mapped_outcome
        .num_unique_phy_evaluations
        ==
        baseline_outcome
        .num_unique_phy_evaluations
    )

    assert (
        mapped_outcome.num_score_requests
        > 0
    )

    assert (
        mapped_outcome
        .num_unique_phy_evaluations
        > 0
    )




def test_ppo_physical_tti_can_skip_counterfactual_greedy():
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
        physical_inputs=(
            physical_inputs
        ),
        greedy_config=(
            PPOGreedySearchConfig()
        ),
        run_counterfactual_greedy=False,
    )

    assert (
        outcome.greedy_search
        is None
    )

    assert (
        outcome.num_score_requests
        == 0
    )

    assert (
        outcome
        .num_unique_phy_evaluations
        == 0
    )

    assert torch.isfinite(
        outcome
        .candidate_total_target_compliant_rate_bps
    ).all()

    with pytest.raises(
        ValueError,
        match="counterfactual greedy search",
    ):
        resolve_ppo_tti_reward(
            physical_outcome=outcome,
            reward_throughput_bps=(
                torch.ones_like(
                    past_average
                )
            ),
            reward_valid_ue_mask=(
                candidate_valid_mask
            ),
            reward_population_name=(
                "test"
            ),
            reward_config=(
                PPORewardConfig(
                    geometric_mean_normalizer_bps=(
                        10.0
                    ),
                )
            ),
        )


def test_ppo_physical_tti_reuses_shared_scorer():
    (
        allocation,
        actions,
        past_average,
        candidate_valid_mask,
        physical_inputs,
    ) = build_physical_case()

    scorer = (
        CachedPPOPhysicalRBGScorer(
            physical_inputs
        )
    )

    #
    # Pre-populate one cache entry as Teacher 2
    # would do before the PPO reward judge.
    #
    scorer(
        torch.tensor(
            [
                0,
            ],
            dtype=torch.long,
            device=(
                past_average.device
            ),
        ),
        0,
    )

    requests_before = (
        scorer.num_score_requests
    )

    unique_before = (
        scorer
        .num_unique_phy_evaluations
    )

    outcome = evaluate_ppo_physical_tti(
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
        physical_scorer=scorer,
    )

    assert (
        outcome.greedy_search
        is not None
    )

    assert (
        outcome.num_score_requests
        ==
        (
            scorer.num_score_requests
            - requests_before
        )
    )

    assert (
        outcome
        .num_unique_phy_evaluations
        ==
        (
            scorer
            .num_unique_phy_evaluations
            - unique_before
        )
    )


