from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
    selected_candidates_for_rbg,
    validate_cell_allocation,
)


@dataclass
class SpatialAllocationFeatures:
    """
    Allocation-dependent features used by the deep scheduler.

    allocated_rbg_count:
        [candidate]

        Number of distinct RBGs currently allocated to each
        candidate UE.

    max_precoder_cross_correlation:
        [candidate, RBG]

        Maximum precoder cross-correlation between a candidate
        and the UEs already co-scheduled on that RBG.

        Zero when no comparison UE currently exists.
    """

    allocated_rbg_count: torch.Tensor

    max_precoder_cross_correlation: torch.Tensor

def compute_allocated_rbg_count(
    allocation: CellAllocation,
    num_candidates: int,
    candidate_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Count the number of distinct RBGs currently allocated
    to every candidate UE.

    Returns:
        [candidate]
    """

    validate_cell_allocation(
        allocation=allocation,
        num_candidates=num_candidates,
    )

    device = (
        allocation
        .candidate_by_user_slot
        .device
    )

    candidate_indices = torch.arange(
        num_candidates,
        dtype=torch.long,
        device=device,
    )

    candidate_is_on_rbg = (
        allocation
        .candidate_by_user_slot
        .unsqueeze(0)
        == candidate_indices[
            :,
            None,
            None,
        ]
    )

    candidate_is_on_rbg = (
        candidate_is_on_rbg.any(
            dim=1
        )
    )


    allocated_rbg_count = (
        candidate_is_on_rbg.sum(
            dim=-1
        )
    )

    if candidate_valid_mask is not None:

        if tuple(
            candidate_valid_mask.shape
        ) != (
            num_candidates,
        ):
            raise ValueError(
                "candidate_valid_mask must have "
                "shape [candidate]."
            )

        valid_mask = (
            candidate_valid_mask.to(
                device=device,
                dtype=torch.bool,
            )
        )

        allocated_rbg_count = torch.where(
            valid_mask,
            allocated_rbg_count,
            torch.zeros_like(
                allocated_rbg_count
            ),
        )

    return allocated_rbg_count

def compute_max_precoder_cross_correlation(
    allocation: CellAllocation,
    candidate_rank: torch.Tensor,
    candidate_precoder_directions: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the paper-style maximum precoder
    cross-correlation feature.

    candidate_rank:
        [candidate]

    candidate_precoder_directions:
        [candidate, RBG, mode, TX_ant]

    candidate_valid_mask:
        [candidate]

    Returns:
        [candidate, RBG]

    Reproduction substitution:
        The supplied precoder directions currently come from
        our ideal-SVD CSI surrogate rather than quantized PMI.
    """

    if candidate_rank.ndim != 1:
        raise ValueError(
            "candidate_rank must have shape [candidate]."
        )

    num_candidates = int(
        candidate_rank.numel()
    )

    validate_cell_allocation(
        allocation=allocation,
        num_candidates=num_candidates,
    )

    if candidate_precoder_directions.ndim != 4:
        raise ValueError(
            "candidate_precoder_directions must have shape "
            "[candidate, RBG, mode, TX_ant]."
        )

    if (
        candidate_precoder_directions.shape[0]
        != num_candidates
    ):
        raise ValueError(
            "Candidate dimension does not match."
        )

    if (
        candidate_precoder_directions.shape[1]
        != allocation.num_rbgs
    ):
        raise ValueError(
            "RBG dimension does not match allocation."
        )

    if tuple(
        candidate_valid_mask.shape
    ) != (
        num_candidates,
    ):
        raise ValueError(
            "candidate_valid_mask must have "
            "shape [candidate]."
        )

    if not torch.is_complex(
        candidate_precoder_directions
    ):
        raise ValueError(
            "Precoder directions must be complex."
        )

    if not torch.isfinite(
        candidate_precoder_directions
    ).all():
        raise ValueError(
            "Precoder directions contain "
            "non-finite values."
        )

    device = (
        candidate_precoder_directions.device
    )

    valid_mask = (
        candidate_valid_mask.to(
            device=device,
            dtype=torch.bool,
        )
    )

    max_available_modes = int(
        candidate_precoder_directions.shape[
            -2
        ]
    )

    valid_ranks = (
        candidate_rank[
            valid_mask
        ]
    )

    if torch.any(
        valid_ranks < 1
    ):
        raise ValueError(
            "Valid candidate ranks must be positive."
        )

    if torch.any(
        valid_ranks > max_available_modes
    ):
        raise ValueError(
            "Candidate rank exceeds the available "
            "precoder directions."
        )

    real_dtype = (
        candidate_precoder_directions
        .real
        .dtype
    )

    max_correlation = torch.zeros(
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

        if (
            scheduled_candidates.numel()
            == 0
        ):
            continue

        for candidate_index in range(
            num_candidates
        ):

            if not bool(
                valid_mask[
                    candidate_index
                ].item()
            ):
                continue

            comparison_candidates = (
                scheduled_candidates[
                    scheduled_candidates
                    != candidate_index
                ]
            )

            if (
                comparison_candidates.numel()
                == 0
            ):
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
                candidate_directions
                .transpose(
                    0,
                    1,
                )
            )

            candidate_max = torch.zeros(
                (),
                dtype=real_dtype,
                device=device,
            )

            for scheduled_index_tensor in (
                comparison_candidates
            ):

                scheduled_index = int(
                    scheduled_index_tensor.item()
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
                    scheduled_directions
                    .transpose(
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

                absolute_overlap = torch.abs(
                    overlap
                )

                correlation_by_scheduled_stream = (
                    absolute_overlap.sum(
                        dim=0
                    )
                )

                pair_correlation = (
                    correlation_by_scheduled_stream.max()
                )
                candidate_max = torch.maximum(
                    candidate_max,
                    pair_correlation,
                )

            max_correlation[
                candidate_index,
                rbg_index,
            ] = candidate_max

    return max_correlation


def build_spatial_allocation_features(
    allocation: CellAllocation,
    candidate_rank: torch.Tensor,
    candidate_precoder_directions: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> SpatialAllocationFeatures:
    """
    Build the allocation-dependent portion of the
    deep-scheduler state.
    """

    num_candidates = int(
        candidate_rank.numel()
    )

    allocated_rbg_count = (
        compute_allocated_rbg_count(
            allocation=allocation,
            num_candidates=num_candidates,
            candidate_valid_mask=(
                candidate_valid_mask
            ),
        )
    )

    max_cross_correlation = (
        compute_max_precoder_cross_correlation(
            allocation=allocation,
            candidate_rank=candidate_rank,
            candidate_precoder_directions=(
                candidate_precoder_directions
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
        )
    )

    return SpatialAllocationFeatures(
        allocated_rbg_count=(
            allocated_rbg_count
        ),
        max_precoder_cross_correlation=(
            max_cross_correlation
        ),
    )


