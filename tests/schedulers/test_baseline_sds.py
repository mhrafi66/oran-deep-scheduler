import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.schedulers.baseline_sds import (
    BaselineSDSConfig,
    run_baseline_sds,
)


def test_baseline_sds_accepts_first_improvement():
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

    rates = {
        (0,): 10.0,
        (0, 1): 9.0,
        (0, 2): 12.0,
        (0, 3): 20.0,
    }

    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> torch.Tensor:

        assert rbg_index == 0

        key = tuple(
            int(value)
            for value
            in selected_candidates.tolist()
        )

        return torch.tensor(
            rates[key],
            device=device,
        )


    result = run_baseline_sds(
        initial_allocation=initial,
        num_candidates=4,
        score_rbg=score_rbg,
        config=BaselineSDSConfig(),
    )


    selected_second_slot = int(
        result
        .allocation
        .candidate_by_user_slot[
            1,
            0,
        ]
        .item()
    )

    assert selected_second_slot == 2

    torch.testing.assert_close(
        result.rbg_rate_bps[
            0
        ],
        torch.tensor(
            12.0,
            device=device,
        ),
    )

def test_baseline_sds_can_add_multiple_users():
    device = "cuda:0"

    initial = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0],
                [-1],
                [-1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    rates = {
        (0,): 10.0,

        (0, 1): 12.0,

        (0, 1, 2): 11.0,
        (0, 1, 3): 15.0,
    }


    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> torch.Tensor:

        key = tuple(
            int(value)
            for value
            in selected_candidates.tolist()
        )

        return torch.tensor(
            rates[key],
            device=device,
        )


    result = run_baseline_sds(
        initial_allocation=initial,
        num_candidates=4,
        score_rbg=score_rbg,
        config=BaselineSDSConfig(),
    )


    expected = torch.tensor(
        [
            [0],
            [1],
            [3],
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result
        .allocation
        .candidate_by_user_slot,
        expected,
    )


    torch.testing.assert_close(
        result.rbg_rate_bps[
            0
        ],
        torch.tensor(
            15.0,
            device=device,
        ),
    )


def test_baseline_sds_stops_when_no_candidate_improves():
    device = "cuda:0"

    initial = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0],
                [-1],
                [-1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    rates = {
        (0,): 10.0,
        (0, 1): 9.0,
        (0, 2): 8.0,
    }

    def score_rbg(
        selected_candidates: torch.Tensor,
        rbg_index: int,
    ) -> torch.Tensor:

        key = tuple(
            int(value)
            for value
            in selected_candidates.tolist()
        )

        return torch.tensor(
            rates[key],
            device=device,
        )
    

    result = run_baseline_sds(
        initial_allocation=initial,
        num_candidates=3,
        score_rbg=score_rbg,
        config=BaselineSDSConfig(),
    )

    expected = torch.tensor(
        [
            [0],
            [-1],
            [-1],
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result
        .allocation
        .candidate_by_user_slot,
        expected,
    )

