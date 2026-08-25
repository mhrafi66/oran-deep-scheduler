import math

import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.state.spatial_features import (
    build_spatial_allocation_features,
    compute_allocated_rbg_count,
    compute_max_precoder_cross_correlation,
)


def test_allocated_rbg_count():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0, 1, 0],
                [2, 0, -1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    valid_mask = torch.tensor(
        [True, True, True],
        dtype=torch.bool,
        device=device,
    )

    result = compute_allocated_rbg_count(
        allocation=allocation,
        num_candidates=3,
        candidate_valid_mask=valid_mask,
    )


    expected = torch.tensor(
        [3, 1, 1],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result,
        expected,
    )

def test_rank1_precoder_cross_correlation():
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


    directions = torch.zeros(
        (
            3,  # candidates
            1,  # RBG
            2,  # maximum modes
            2,  # TX
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        0,
        0,
        0,
        :,
    ] = torch.tensor(
        [1.0, 0.0],
        dtype=torch.complex64,
        device=device,
    )

    directions[
        1,
        0,
        0,
        :,
    ] = torch.tensor(
        [1.0, 0.0],
        dtype=torch.complex64,
        device=device,
    )

    directions[
        2,
        0,
        0,
        :,
    ] = torch.tensor(
        [0.0, 1.0],
        dtype=torch.complex64,
        device=device,
    )

    ranks = torch.tensor(
        [1, 1, 1],
        dtype=torch.long,
        device=device,
    )

    valid_mask = torch.ones(
        3,
        dtype=torch.bool,
        device=device,
    )

    result = (
        compute_max_precoder_cross_correlation(
            allocation=allocation,
            candidate_rank=ranks,
            candidate_precoder_directions=(
                directions
            ),
            candidate_valid_mask=valid_mask,
        )
    )

    expected = torch.tensor(
        [
            [0.0],
            [1.0],
            [0.0],
        ],
        device=device,
    )

    torch.testing.assert_close(
        result,
        expected,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

def test_rank2_cross_correlation_can_exceed_one():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    directions = torch.zeros(
        (
            2,
            1,
            2,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        0,
        0,
        0,
        :,
    ] = torch.tensor(
        [1.0, 0.0],
        dtype=torch.complex64,
        device=device,
    )

    directions[
        0,
        0,
        1,
        :,
    ] = torch.tensor(
        [0.0, 1.0],
        dtype=torch.complex64,
        device=device,
    )

    value = 1.0 / math.sqrt(2.0)

    directions[
        1,
        0,
        0,
        :,
    ] = torch.tensor(
        [value, value],
        dtype=torch.complex64,
        device=device,
    )

    ranks = torch.tensor(
        [2, 1],
        dtype=torch.long,
        device=device,
    )

    valid_mask = torch.tensor(
        [True, True],
        dtype=torch.bool,
        device=device,
    )

    result = (
        compute_max_precoder_cross_correlation(
            allocation=allocation,
            candidate_rank=ranks,
            candidate_precoder_directions=(
                directions
            ),
            candidate_valid_mask=valid_mask,
        )
    )

    expected = torch.tensor(
        math.sqrt(2.0),
        device=device,
    )

    torch.testing.assert_close(
        result[
            0,
            0,
        ],
        expected,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert float(
        result[
            0,
            0,
        ].item()
    ) > 1.0

def test_build_spatial_allocation_features():
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

    directions = torch.zeros(
        (
            2,
            1,
            2,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        0,
        0,
        0,
        0,
    ] = 1.0

    directions[
        1,
        0,
        0,
        1,
    ] = 1.0

    ranks = torch.tensor(
        [1, 1],
        dtype=torch.long,
        device=device,
    )

    valid_mask = torch.tensor(
        [True, True],
        dtype=torch.bool,
        device=device,
    )

    result = build_spatial_allocation_features(
        allocation=allocation,
        candidate_rank=ranks,
        candidate_precoder_directions=(
            directions
        ),
        candidate_valid_mask=valid_mask,
    )

    expected_count = torch.tensor(
        [1, 0],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result.allocated_rbg_count,
        expected_count,
    )

    assert (
        result
        .max_precoder_cross_correlation
        .shape
        == (2, 1)
    )

