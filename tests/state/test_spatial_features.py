import math

import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
    NO_ALLOCATION,
    selected_candidates_for_rbg,
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

def _reference_max_precoder_cross_correlation(
    *,
    allocation,
    candidate_rank,
    candidate_precoder_directions,
    candidate_valid_mask,
):
    """
    Original scalar implementation retained only
    inside the test as an equivalence oracle.
    """

    num_candidates = int(
        candidate_rank.numel()
    )

    device = (
        candidate_precoder_directions.device
    )

    real_dtype = (
        candidate_precoder_directions
        .real
        .dtype
    )

    result = torch.zeros(
        (
            num_candidates,
            allocation.num_rbgs,
        ),
        dtype=real_dtype,
        device=device,
    )

    for rbg_index in range(
        allocation.num_rbgs
    ):
        scheduled_candidates = (
            selected_candidates_for_rbg(
                allocation=allocation,
                rbg_index=rbg_index,
            )
        )

        if scheduled_candidates.numel() == 0:
            continue

        for candidate_index in range(
            num_candidates
        ):
            if not bool(
                candidate_valid_mask[
                    candidate_index
                ].item()
            ):
                continue

            comparisons = (
                scheduled_candidates[
                    scheduled_candidates
                    != candidate_index
                ]
            )

            if comparisons.numel() == 0:
                continue

            candidate_rank_value = int(
                candidate_rank[
                    candidate_index
                ].item()
            )

            candidate_directions = (
                candidate_precoder_directions[
                    candidate_index,
                    rbg_index,
                    :candidate_rank_value,
                    :,
                ]
            )

            candidate_precoder = (
                candidate_directions.transpose(
                    0,
                    1,
                )
            )

            candidate_max = torch.zeros(
                (),
                dtype=real_dtype,
                device=device,
            )

            for scheduled_tensor in comparisons:
                scheduled_index = int(
                    scheduled_tensor.item()
                )

                scheduled_rank = int(
                    candidate_rank[
                        scheduled_index
                    ].item()
                )

                scheduled_directions = (
                    candidate_precoder_directions[
                        scheduled_index,
                        rbg_index,
                        :scheduled_rank,
                        :,
                    ]
                )

                scheduled_precoder = (
                    scheduled_directions.transpose(
                        0,
                        1,
                    )
                )

                overlap = (
                    candidate_precoder
                    .conj()
                    .transpose(
                        0,
                        1,
                    )
                    @ scheduled_precoder
                )

                pair = (
                    torch.abs(
                        overlap
                    )
                    .sum(
                        dim=0
                    )
                    .max()
                )

                candidate_max = torch.maximum(
                    candidate_max,
                    pair,
                )

            result[
                candidate_index,
                rbg_index,
            ] = candidate_max

    return result


def test_vectorized_precoder_correlation_matches_reference():
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    torch.manual_seed(
        20260914
    )

    num_candidates = 10
    num_rbgs = 18
    num_modes = 2
    num_tx = 16

    real = torch.randn(
        (
            num_candidates,
            num_rbgs,
            num_modes,
            num_tx,
        ),
        device=device,
    )

    imag = torch.randn_like(
        real
    )

    directions = torch.complex(
        real,
        imag,
    )

    #
    # Normalize each spatial direction so this
    # resembles actual CSI precoders.
    #
    directions = (
        directions
        / torch.linalg.vector_norm(
            directions,
            dim=-1,
            keepdim=True,
        )
    )

    ranks = torch.tensor(
        [
            1, 2, 1, 2, 1,
            2, 1, 2, 1, 2,
        ],
        dtype=torch.long,
        device=device,
    )

    valid_mask = torch.tensor(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ],
        dtype=torch.bool,
        device=device,
    )

    allocation_tensor = torch.full(
        (
            4,
            num_rbgs,
        ),
        fill_value=NO_ALLOCATION,
        dtype=torch.long,
        device=device,
    )

    #
    # Build different scheduled populations across
    # the 18 RBGs.
    #
    for rbg_index in range(
        num_rbgs
    ):
        allocation_tensor[
            0,
            rbg_index,
        ] = (
            rbg_index
            % 6
        )

        if rbg_index % 2 == 0:
            allocation_tensor[
                1,
                rbg_index,
            ] = (
                (rbg_index + 1)
                % 6
            )

        if rbg_index % 3 == 0:
            allocation_tensor[
                2,
                rbg_index,
            ] = (
                (rbg_index + 2)
                % 6
            )

    allocation = CellAllocation(
        candidate_by_user_slot=(
            allocation_tensor
        )
    )

    expected = (
        _reference_max_precoder_cross_correlation(
            allocation=allocation,
            candidate_rank=ranks,
            candidate_precoder_directions=(
                directions
            ),
            candidate_valid_mask=(
                valid_mask
            ),
        )
    )

    actual = (
        compute_max_precoder_cross_correlation(
            allocation=allocation,
            candidate_rank=ranks,
            candidate_precoder_directions=(
                directions
            ),
            candidate_valid_mask=(
                valid_mask
            ),
        )
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


