import torch

from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
    evaluate_ppo_layer_greedy_search,
    evaluate_ppo_schedule_greedy_search,
)
from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    PFGreedyRBGScore,
)


def build_score_function(
    scores,
    device,
):
    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:
        assert rbg_index == 0

        key = tuple(
            int(value)
            for value
            in selected_candidates.tolist()
        )

        total_rate, candidate_rates = (
            scores[key]
        )

        return PFGreedyRBGScore(
            total_rate_bps=torch.tensor(
                total_rate,
                dtype=torch.float32,
                device=device,
            ),
            selected_candidate_rate_bps=(
                torch.tensor(
                    candidate_rates,
                    dtype=torch.float32,
                    device=device,
                )
            ),
        )

    return score_rbg


def build_multi_rbg_score_function(
    scores,
    device,
):
    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:
        candidate_key = tuple(
            int(value)
            for value
            in selected_candidates.tolist()
        )

        key = (
            rbg_index,
            candidate_key,
        )

        total_rate, candidate_rates = (
            scores[key]
        )

        return PFGreedyRBGScore(
            total_rate_bps=torch.tensor(
                total_rate,
                dtype=torch.float32,
                device=device,
            ),
            selected_candidate_rate_bps=(
                torch.tensor(
                    candidate_rates,
                    dtype=torch.float32,
                    device=device,
                )
            ),
        )

    return score_rbg


def test_complete_schedule_greedy_search_replays_layers():
    device = "cuda:0"

    actions = torch.tensor(
        [
            [
                0,
                1,
            ],
            [
                1,
                0,
            ],
        ],
        dtype=torch.long,
        device=device,
    )

    past_average = torch.ones(
        3,
        dtype=torch.float32,
        device=device,
    )

    valid_mask = torch.ones(
        3,
        dtype=torch.bool,
        device=device,
    )

    scores = {
        # --------------------------------------------------
        # RBG 0, layer 0 alternatives
        # --------------------------------------------------
        (
            0,
            (),
        ): (
            0.0,
            [],
        ),

        (
            0,
            (0,),
        ): (
            5.0,
            [5.0],
        ),

        (
            0,
            (1,),
        ): (
            7.0,
            [7.0],
        ),

        (
            0,
            (2,),
        ): (
            4.0,
            [4.0],
        ),

        # --------------------------------------------------
        # RBG 0, layer 1 alternatives.
        #
        # Previous layer already selected candidate 0.
        # --------------------------------------------------
        (
            0,
            (0, 1),
        ): (
            12.0,
            [
                6.0,
                6.0,
            ],
        ),

        (
            0,
            (0, 2),
        ): (
            10.0,
            [
                9.0,
                1.0,
            ],
        ),

        # --------------------------------------------------
        # RBG 1, layer 0 alternatives
        # --------------------------------------------------
        (
            1,
            (),
        ): (
            0.0,
            [],
        ),

        (
            1,
            (0,),
        ): (
            6.0,
            [6.0],
        ),

        (
            1,
            (1,),
        ): (
            8.0,
            [8.0],
        ),

        (
            1,
            (2,),
        ): (
            5.0,
            [5.0],
        ),

        # --------------------------------------------------
        # RBG 1, layer 1 alternatives.
        #
        # Previous layer already selected candidate 1.
        # --------------------------------------------------
        (
            1,
            (1, 0),
        ): (
            8.0,
            [
                4.0,
                4.0,
            ],
        ),

        (
            1,
            (1, 2),
        ): (
            10.0,
            [
                7.0,
                3.0,
            ],
        ),
    }

    result = (
        evaluate_ppo_schedule_greedy_search(
            actions=actions,
            num_candidates=3,
            past_average_throughput=(
                past_average
            ),
            candidate_valid_mask=(
                valid_mask
            ),
            score_rbg=(
                build_multi_rbg_score_function(
                    scores,
                    device,
                )
            ),
            config=PPOGreedySearchConfig(),
        )
    )

    expected_better = torch.tensor(
        [
            [
                True,
                False,
            ],
            [
                False,
                True,
            ],
        ],
        dtype=torch.bool,
        device=device,
    )

    torch.testing.assert_close(
        result.better_allocation_exists,
        expected_better,
    )


    expected_chosen_pf = torch.tensor(
        [
            [
                5.0,
                8.0,
            ],
            [
                12.0,
                8.0,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    torch.testing.assert_close(
        result.chosen_pf_sum,
        expected_chosen_pf,
    )

    expected_best_pf = torch.tensor(
        [
            [
                7.0,
                8.0,
            ],
            [
                12.0,
                10.0,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )

    torch.testing.assert_close(
        result.best_pf_sum,
        expected_best_pf,
    )

    expected_allocation = torch.tensor(
        [
            [
                0,
                1,
            ],
            [
                1,
                0,
            ],
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result
        .allocation
        .candidate_by_user_slot,
        expected_allocation,
    )

    assert (
        result.num_phy_evaluations
        == 14
    )

def test_complete_schedule_requires_layer_rbg_matrix():
    device = "cuda:0"

    invalid_actions = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.long,
        device=device,
    )

    try:
        evaluate_ppo_schedule_greedy_search(
            actions=invalid_actions,
            num_candidates=2,
            past_average_throughput=(
                torch.ones(
                    2,
                    device=device,
                )
            ),
            candidate_valid_mask=(
                torch.ones(
                    2,
                    dtype=torch.bool,
                    device=device,
                )
            ),
            score_rbg=(
                build_multi_rbg_score_function(
                    {},
                    device,
                )
            ),
            config=PPOGreedySearchConfig(),
        )

    except ValueError:
        return

    raise AssertionError(
        "Complete actions must have shape "
        "[user_layer, RBG]."
    )






def test_first_layer_detects_better_pf_action():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [-1],
                [-1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    chosen_actions = torch.tensor(
        [0],
        dtype=torch.long,
        device=device,
    )

    past_average = torch.tensor(
        [
            10.0,
            1.0,
            10.0,
        ],
        dtype=torch.float32,
        device=device,
    )

    valid_mask = torch.ones(
        3,
        dtype=torch.bool,
        device=device,
    )

    scores = {
        (): (
            0.0,
            [],
        ),

        (0,): (
            10.0,
            [10.0],
        ),

        (1,): (
            4.0,
            [4.0],
        ),

        (2,): (
            2.0,
            [2.0],
        ),
    }

    result = evaluate_ppo_layer_greedy_search(
        allocation=allocation,
        user_slot_index=0,
        chosen_actions=chosen_actions,
        num_candidates=3,
        past_average_throughput=(
            past_average
        ),
        candidate_valid_mask=valid_mask,
        score_rbg=build_score_function(
            scores,
            device,
        ),
        config=PPOGreedySearchConfig(),
    )

    assert bool(
        result
        .better_allocation_exists[
            0
        ].item()
    )

    torch.testing.assert_close(
        result.chosen_pf_sum,
        torch.tensor(
            [1.0],
            device=device,
        ),
    )

    torch.testing.assert_close(
        result.best_pf_sum,
        torch.tensor(
            [4.0],
            device=device,
        ),
    )


def test_first_layer_reports_no_better_action():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [-1],
                [-1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    past_average = torch.tensor(
        [
            10.0,
            1.0,
            10.0,
        ],
        dtype=torch.float32,
        device=device,
    )

    scores = {
        (): (
            0.0,
            [],
        ),

        (0,): (
            10.0,
            [10.0],
        ),

        (1,): (
            4.0,
            [4.0],
        ),

        (2,): (
            2.0,
            [2.0],
        ),
    }

    result = evaluate_ppo_layer_greedy_search(
        allocation=allocation,
        user_slot_index=0,
        chosen_actions=torch.tensor(
            [1],
            dtype=torch.long,
            device=device,
        ),
        num_candidates=3,
        past_average_throughput=(
            past_average
        ),
        candidate_valid_mask=torch.ones(
            3,
            dtype=torch.bool,
            device=device,
        ),
        score_rbg=build_score_function(
            scores,
            device,
        ),
        config=PPOGreedySearchConfig(),
    )

    assert not bool(
        result
        .better_allocation_exists[
            0
        ].item()
    )

    torch.testing.assert_close(
        result.chosen_pf_sum,
        torch.tensor(
            [4.0],
            device=device,
        ),
    )

    torch.testing.assert_close(
        result.best_pf_sum,
        torch.tensor(
            [4.0],
            device=device,
        ),
    )


def test_no_allocation_can_be_better_than_bad_pairing():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0],
                [-1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    past_average = torch.tensor(
        [
            10.0,
            1.0,
            10.0,
        ],
        dtype=torch.float32,
        device=device,
    )

    scores = {
        (0,): (
            10.0,
            [10.0],
        ),

        (0, 1): (
            5.1,
            [5.0, 0.1],
        ),

        (0, 2): (
            8.0,
            [7.0, 1.0],
        ),
    }

    result = evaluate_ppo_layer_greedy_search(
        allocation=allocation,
        user_slot_index=1,
        chosen_actions=torch.tensor(
            [1],
            dtype=torch.long,
            device=device,
        ),
        num_candidates=3,
        past_average_throughput=(
            past_average
        ),
        candidate_valid_mask=torch.ones(
            3,
            dtype=torch.bool,
            device=device,
        ),
        score_rbg=build_score_function(
            scores,
            device,
        ),
        config=PPOGreedySearchConfig(),
    )

    assert bool(
        result
        .better_allocation_exists[
            0
        ].item()
    )


def test_higher_pf_counts_even_with_lower_total_throughput():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0],
                [-1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    past_average = torch.tensor(
        [
            10.0,
            1.0,
            0.5,
        ],
        dtype=torch.float32,
        device=device,
    )

    scores = {
        (0,): (
            10.0,
            [10.0],
        ),

        # RL choice:
        # total throughput = 12
        # PF = 8/10 + 4/1 = 4.8
        (0, 1): (
            12.0,
            [8.0, 4.0],
        ),

        # Alternative:
        # total throughput = 11, which is LOWER,
        # but PF = 5/10 + 6/0.5 = 12.5.
        (0, 2): (
            11.0,
            [5.0, 6.0],
        ),
    }

    result = evaluate_ppo_layer_greedy_search(
        allocation=allocation,
        user_slot_index=1,
        chosen_actions=torch.tensor(
            [1],
            dtype=torch.long,
            device=device,
        ),
        num_candidates=3,
        past_average_throughput=(
            past_average
        ),
        candidate_valid_mask=torch.ones(
            3,
            dtype=torch.bool,
            device=device,
        ),
        score_rbg=build_score_function(
            scores,
            device,
        ),
        config=PPOGreedySearchConfig(),
    )

    assert bool(
        result
        .better_allocation_exists[
            0
        ].item()
    )

    torch.testing.assert_close(
        result.chosen_pf_sum,
        torch.tensor(
            [4.8],
            device=device,
        ),
    )

    torch.testing.assert_close(
        result.best_pf_sum,
        torch.tensor(
            [12.5],
            device=device,
        ),
    )


def test_masked_candidate_action_is_rejected():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [-1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    valid_mask = torch.tensor(
        [
            True,
            False,
        ],
        dtype=torch.bool,
        device=device,
    )

    try:
        evaluate_ppo_layer_greedy_search(
            allocation=allocation,
            user_slot_index=0,
            chosen_actions=torch.tensor(
                [1],
                dtype=torch.long,
                device=device,
            ),
            num_candidates=2,
            past_average_throughput=(
                torch.ones(
                    2,
                    device=device,
                )
            ),
            candidate_valid_mask=valid_mask,
            score_rbg=build_score_function(
                {},
                device,
            ),
            config=PPOGreedySearchConfig(),
        )

    except ValueError:
        return

    raise AssertionError(
        "A masked candidate action must be rejected."
    )


