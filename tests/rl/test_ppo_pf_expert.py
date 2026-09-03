import torch

from oran_scheduler.rl.ppo_pf_expert import (
    PPOPFExpertConfig,
    generate_ppo_pf_expert_actions,
)
from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    PFGreedyRBGScore,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def build_score_function(
    *,
    scores,
    device: torch.device,
):
    calls = []

    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:
        candidate_tuple = tuple(
            int(value)
            for value
            in selected_candidates.tolist()
        )

        key = (
            rbg_index,
            candidate_tuple,
        )

        calls.append(
            key
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

    return (
        score_rbg,
        calls,
    )


def test_pf_expert_selects_maximum_pf_action():
    device = preferred_device()

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.full(
                (
                    2,
                    2,
                ),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
        )
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
        #
        # RBG 0
        #
        (0, ()): (
            0.0,
            [],
        ),
        (0, (0,)): (
            5.0,
            [5.0],
        ),
        (0, (1,)): (
            9.0,
            [9.0],
        ),
        (0, (2,)): (
            7.0,
            [7.0],
        ),

        #
        # RBG 1
        #
        (1, ()): (
            0.0,
            [],
        ),
        (1, (0,)): (
            4.0,
            [4.0],
        ),
        (1, (1,)): (
            6.0,
            [6.0],
        ),
        (1, (2,)): (
            10.0,
            [10.0],
        ),
    }

    score_rbg, _ = build_score_function(
        scores=scores,
        device=device,
    )

    result = (
        generate_ppo_pf_expert_actions(
            allocation=allocation,
            user_slot_index=0,
            num_candidates=3,
            past_average_throughput=(
                past_average
            ),
            candidate_valid_mask=(
                valid_mask
            ),
            score_rbg=score_rbg,
            config=PPOPFExpertConfig(),
        )
    )

    torch.testing.assert_close(
        result.expert_actions,
        torch.tensor(
            [
                1,
                2,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    torch.testing.assert_close(
        result.best_pf_sum,
        torch.tensor(
            [
                9.0,
                10.0,
            ],
            device=device,
        ),
    )

    assert tuple(
        result.action_mask.shape
    ) == (
        2,
        4,
    )


def test_pf_expert_uses_past_average_throughput():
    device = preferred_device()

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.full(
                (
                    1,
                    1,
                ),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
        )
    )

    #
    # UE0 has much larger past throughput.
    #
    past_average = torch.tensor(
        [
            10.0,
            1.0,
        ],
        dtype=torch.float32,
        device=device,
    )

    valid_mask = torch.ones(
        2,
        dtype=torch.bool,
        device=device,
    )

    scores = {
        (0, ()): (
            0.0,
            [],
        ),

        #
        # UE0 gets MORE raw rate:
        #
        (0, (0,)): (
            20.0,
            [
                20.0,
            ],
        ),

        #
        # UE1 gets LESS raw rate:
        #
        (0, (1,)): (
            8.0,
            [
                8.0,
            ],
        ),
    }

    score_rbg, _ = build_score_function(
        scores=scores,
        device=device,
    )

    result = (
        generate_ppo_pf_expert_actions(
            allocation=allocation,
            user_slot_index=0,
            num_candidates=2,
            past_average_throughput=(
                past_average
            ),
            candidate_valid_mask=(
                valid_mask
            ),
            score_rbg=score_rbg,
            config=PPOPFExpertConfig(),
        )
    )

    #
    # UE0:
    #     PF = 20 / 10 = 2
    #
    # UE1:
    #     PF = 8 / 1 = 8
    #
    # Therefore expert chooses UE1.
    #
    assert int(
        result.expert_actions[0].item()
    ) == 1


def test_pf_expert_conditions_on_previous_ppo_layers():
    device = preferred_device()

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.tensor(
                [
                    [
                        0,
                    ],
                    [
                        -1,
                    ],
                    [
                        -1,
                    ],
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    original_allocation = (
        allocation
        .candidate_by_user_slot
        .clone()
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
        #
        # Previous PPO layer already chose UE0.
        #
        # NO ALLOCATION:
        #
        (0, (0,)): (
            10.0,
            [
                10.0,
            ],
        ),

        #
        # Add UE1:
        #
        (0, (0, 1)): (
            14.0,
            [
                7.0,
                7.0,
            ],
        ),

        #
        # Add UE2:
        #
        (0, (0, 2)): (
            12.0,
            [
                10.0,
                2.0,
            ],
        ),
    }

    score_rbg, calls = (
        build_score_function(
            scores=scores,
            device=device,
        )
    )

    result = (
        generate_ppo_pf_expert_actions(
            allocation=allocation,
            user_slot_index=1,
            num_candidates=3,
            past_average_throughput=(
                past_average
            ),
            candidate_valid_mask=(
                valid_mask
            ),
            score_rbg=score_rbg,
            config=PPOPFExpertConfig(),
        )
    )

    assert int(
        result.expert_actions[0].item()
    ) == 1

    #
    # UE0 itself is illegal because PPO already used
    # it on this RBG.
    #
    assert not bool(
        result.action_mask[
            0,
            0,
        ].item()
    )

    assert (
        0,
        (
            0,
            0,
        ),
    ) not in calls

    #
    # Expert is supervision ONLY.
    #
    torch.testing.assert_close(
        allocation
        .candidate_by_user_slot,
        original_allocation,
    )


def test_pf_expert_never_selects_invalid_candidate():
    device = preferred_device()

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.full(
                (
                    1,
                    1,
                ),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
        )
    )

    valid_mask = torch.tensor(
        [
            True,
            False,
            True,
        ],
        dtype=torch.bool,
        device=device,
    )

    scores = {
        (0, ()): (
            0.0,
            [],
        ),
        (0, (0,)): (
            5.0,
            [5.0],
        ),
        (0, (2,)): (
            7.0,
            [7.0],
        ),
    }

    score_rbg, calls = (
        build_score_function(
            scores=scores,
            device=device,
        )
    )

    result = (
        generate_ppo_pf_expert_actions(
            allocation=allocation,
            user_slot_index=0,
            num_candidates=3,
            past_average_throughput=(
                torch.ones(
                    3,
                    device=device,
                )
            ),
            candidate_valid_mask=(
                valid_mask
            ),
            score_rbg=score_rbg,
            config=PPOPFExpertConfig(),
        )
    )

    assert int(
        result.expert_actions[0].item()
    ) == 2

    #
    # Candidate 1 is padded/invalid and should
    # never even reach the physical scorer.
    #
    assert (
        0,
        (
            1,
        ),
    ) not in calls


def test_pf_expert_can_choose_no_allocation():
    device = preferred_device()

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.tensor(
                [
                    [
                        0,
                    ],
                    [
                        -1,
                    ],
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    scores = {
        #
        # Keep only previously scheduled UE0:
        #
        (0, (0,)): (
            10.0,
            [
                10.0,
            ],
        ),

        #
        # Pair UE1, but interference destroys PF:
        #
        (0, (0, 1)): (
            6.0,
            [
                3.0,
                3.0,
            ],
        ),
    }

    score_rbg, _ = build_score_function(
        scores=scores,
        device=device,
    )

    result = (
        generate_ppo_pf_expert_actions(
            allocation=allocation,
            user_slot_index=1,
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
            score_rbg=score_rbg,
            config=PPOPFExpertConfig(),
        )
    )

    #
    # Actor convention:
    #
    #     num_candidates == NO ALLOCATION
    #
    assert int(
        result.expert_actions[0].item()
    ) == 2


def test_pf_expert_no_allocation_tie_breaking():
    device = preferred_device()

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.tensor(
                [
                    [
                        0,
                    ],
                    [
                        -1,
                    ],
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    scores = {
        (0, (0,)): (
            10.0,
            [
                10.0,
            ],
        ),

        #
        # Exact same PF sum after adding UE1.
        #
        (0, (0, 1)): (
            10.0,
            [
                5.0,
                5.0,
            ],
        ),
    }

    score_rbg, _ = build_score_function(
        scores=scores,
        device=device,
    )

    result = (
        generate_ppo_pf_expert_actions(
            allocation=allocation,
            user_slot_index=1,
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
            score_rbg=score_rbg,
            config=PPOPFExpertConfig(
                tie_breaking="no_allocation",
            ),
        )
    )

    assert int(
        result.expert_actions[0].item()
    ) == 2


def test_pf_expert_paper_shaped_output_on_gpu():
    device = preferred_device()

    num_candidates = 10
    num_rbgs = 18
    num_user_slots = 4

    allocation = CellAllocation(
        candidate_by_user_slot=(
            torch.full(
                (
                    num_user_slots,
                    num_rbgs,
                ),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
        )
    )

    past_average = torch.ones(
        num_candidates,
        dtype=torch.float32,
        device=device,
    )

    valid_mask = torch.ones(
        num_candidates,
        dtype=torch.bool,
        device=device,
    )

    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:
        del rbg_index

        #
        # Deterministic fake PHY/PF score:
        # candidate index + 1.
        #
        if selected_candidates.numel() == 0:
            rates = torch.empty(
                0,
                dtype=torch.float32,
                device=device,
            )

        else:
            rates = (
                selected_candidates.to(
                    torch.float32
                )
                + 1.0
            )

        return PFGreedyRBGScore(
            total_rate_bps=(
                rates.sum()
            ),
            selected_candidate_rate_bps=(
                rates
            ),
        )

    result = (
        generate_ppo_pf_expert_actions(
            allocation=allocation,
            user_slot_index=0,
            num_candidates=num_candidates,
            past_average_throughput=(
                past_average
            ),
            candidate_valid_mask=(
                valid_mask
            ),
            score_rbg=score_rbg,
            config=PPOPFExpertConfig(),
        )
    )

    assert tuple(
        result.expert_actions.shape
    ) == (
        18,
    )

    assert tuple(
        result.action_mask.shape
    ) == (
        18,
        11,
    )

    assert tuple(
        result.best_pf_sum.shape
    ) == (
        18,
    )

    assert (
        result.expert_actions.device
        == device
    )

    #
    # Candidate 9 has largest rate/PF score.
    #
    torch.testing.assert_close(
        result.expert_actions,
        torch.full(
            (
                18,
            ),
            fill_value=9,
            dtype=torch.long,
            device=device,
        ),
    )

    #
    # 11 legal actions per RBG:
    #
    # 10 candidates + NO ALLOCATION.
    #
    assert (
        result.num_phy_evaluations
        == 18 * 11
    )


