import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    PFGreedyRBGScore,
    PFGreedySDSConfig,
    run_pf_greedy_sds,
)


def test_pf_greedy_chooses_best_pf_candidate():
    device = "cuda:0"

    initial = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0],
                [-1],
            ],
            dtype=torch.long,
            device=device,
        )
    )



import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    PFGreedyRBGScore,
    PFGreedySDSConfig,
    run_pf_greedy_sds,
)


def test_pf_greedy_chooses_best_pf_candidate():
    device = "cuda:0"

    initial = CellAllocation(
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
        device=device,
    )

    scores = {
        (0,): (
            10.0,
            [10.0],
        ),

        (0, 1): (
            12.0,
            [8.0, 4.0],
        ),

        (0, 2): (
            20.0,
            [10.0, 10.0],
        ),
    }

    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:

        key = tuple(
            int(value)
            for value
            in selected_candidates.tolist()
        )

        total, rates = scores[key]

        return PFGreedyRBGScore(
            total_rate_bps=torch.tensor(
                total,
                device=device,
            ),
            selected_candidate_rate_bps=(
                torch.tensor(
                    rates,
                    device=device,
                )
            ),
        )
    

    result = run_pf_greedy_sds(
        initial_allocation=initial,
        num_candidates=3,
        past_average_throughput=(
            past_average
        ),
        score_rbg=score_rbg,
        config=PFGreedySDSConfig(),
    )

    selected = int(
        result
        .allocation
        .candidate_by_user_slot[
            1,
            0,
        ]
        .item()
    )

    assert selected == 1


def test_pf_greedy_requires_total_throughput_improvement():
    device = "cuda:0"

    initial = CellAllocation(
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
            0.1,
            10.0,
        ],
        device=device,
    )

    scores = {
        (0,): (
            10.0,
            [10.0],
        ),

        (0, 1): (
            9.0,
            [5.0, 4.0],
        ),

        (0, 2): (
            11.0,
            [9.0, 2.0],
        ),
    }

    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:

        key = tuple(
            int(value)
            for value
            in selected_candidates.tolist()
        )

        total, rates = scores[key]

        return PFGreedyRBGScore(
            total_rate_bps=torch.tensor(
                total,
                device=device,
            ),
            selected_candidate_rate_bps=(
                torch.tensor(
                    rates,
                    device=device,
                )
            ),
        )


    result = run_pf_greedy_sds(
        initial_allocation=initial,
        num_candidates=3,
        past_average_throughput=(
            past_average
        ),
        score_rbg=score_rbg,
        config=PFGreedySDSConfig(),
    )

    selected = int(
        result
        .allocation
        .candidate_by_user_slot[
            1,
            0,
        ]
        .item()
    )

    assert selected == 2

def test_pf_greedy_stops_when_nobody_improves():
    device = "cuda:0"

    initial = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0],
                [-1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    past_average = torch.ones(
        3,
        device=device,
    )

    scores = {
        (0,): (
            10.0,
            [10.0],
        ),

        (0, 1): (
            9.0,
            [5.0, 4.0],
        ),

        (0, 2): (
            8.0,
            [4.0, 4.0],
        ),
    }

    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> PFGreedyRBGScore:

        key = tuple(
            int(value)
            for value
            in selected_candidates.tolist()
        )

        total, rates = scores[key]

        return PFGreedyRBGScore(
            total_rate_bps=torch.tensor(
                total,
                device=device,
            ),
            selected_candidate_rate_bps=(
                torch.tensor(
                    rates,
                    device=device,
                )
            ),
        )

    result = run_pf_greedy_sds(
        initial_allocation=initial,
        num_candidates=3,
        past_average_throughput=(
            past_average
        ),
        score_rbg=score_rbg,
        config=PFGreedySDSConfig(),
    )


    assert int(
        result
        .allocation
        .candidate_by_user_slot[
            1,
            0,
        ]
        .item()
    ) == -1


    torch.testing.assert_close(
        result.rbg_rate_bps[
            0
        ],
        torch.tensor(
            10.0,
            device=device,
        ),
    )
    