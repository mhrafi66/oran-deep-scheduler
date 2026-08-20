import torch

from oran_scheduler.schedulers.su_mimo_fds import (
    SUMIMOFDSConfig,
    run_su_mimo_fds,
)

def test_su_mimo_fds_selects_highest_pf_per_rbg():
    device = "cuda:0"

    candidate_rate = torch.tensor(
        [
            [
                [
                    [10.0, 1.0, 3.0],
                    [6.0, 8.0, 2.0],
                    [20.0, 1.0, 12.0],
                ]
            ]
        ],
        device=device,
    )

    candidate_history = torch.tensor(
        [
            [
                [2.0, 1.0, 4.0]
            ]
        ],
        device=device,
    )

    candidate_valid_mask = torch.tensor(
        [
            [
                [True, True, True]
            ]
        ],
        device=device,
    )

    result = run_su_mimo_fds(
        candidate_rate=candidate_rate,
        candidate_past_average_throughput=(
            candidate_history
        ),
        candidate_valid_mask=candidate_valid_mask,
        config=SUMIMOFDSConfig(),
    )

    expected = torch.tensor(
        [
            [
                [1, 1, 2]
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        result.selected_candidate_indices,
        expected,
    )

    assert torch.all(
        result.selected_valid_mask
    )

def test_su_mimo_fds_masks_invalid_candidates():
    device = "cuda:0"

    candidate_rate = torch.tensor(
        [
            [
                [
                    [5.0, 8.0],
                    [7.0, 4.0],
                    [9999.0, 9999.0],
                ]
            ]
        ],
        device=device,
    )

    candidate_history = torch.ones(
        1,
        1,
        3,
        device=device,
    )

    candidate_valid_mask = torch.tensor(
        [
            [
                [True, True, False]
            ]
        ],
        device=device,
    )

    result = run_su_mimo_fds(
        candidate_rate=candidate_rate,
        candidate_past_average_throughput=(
            candidate_history
        ),
        candidate_valid_mask=candidate_valid_mask,
        config=SUMIMOFDSConfig(),
    )

    expected = torch.tensor(
        [
            [
                [1, 0]
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        result.selected_candidate_indices,
        expected,
    )

def test_su_mimo_fds_handles_empty_cell():
    device = "cuda:0"

    candidate_rate = torch.ones(
        1,
        1,
        10,
        18,
        device=device,
    )

    candidate_history = torch.ones(
        1,
        1,
        10,
        device=device,
    )

    candidate_valid_mask = torch.zeros(
        1,
        1,
        10,
        dtype=torch.bool,
        device=device,
    )

    result = run_su_mimo_fds(
        candidate_rate=candidate_rate,
        candidate_past_average_throughput=(
            candidate_history
        ),
        candidate_valid_mask=candidate_valid_mask,
        config=SUMIMOFDSConfig(),
    )

    assert result.selected_candidate_indices.shape == (
        1,
        1,
        18,
    )

    assert torch.all(
        result.selected_candidate_indices == 0
    )

    assert not torch.any(
        result.selected_valid_mask
    )

    assert torch.all(
        result.selected_pf_metric == 0
    )

