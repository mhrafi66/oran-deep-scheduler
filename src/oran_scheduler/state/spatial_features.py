from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
    NO_ALLOCATION,
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

    PERFORMANCE IMPLEMENTATION
    --------------------------
    The original implementation used nested Python
    loops:

        RBG
          -> candidate
              -> scheduled candidate

    and repeatedly called .item() on CUDA tensors.

    This implementation evaluates every:

        RBG
        x candidate
        x scheduled candidate
        x candidate stream
        x scheduled stream

    simultaneously on the GPU.

    The mathematical definition is unchanged.

    Reproduction substitution:
        The supplied precoder directions currently come
        from our ideal-SVD CSI surrogate rather than
        quantized PMI.
    """

    if candidate_rank.ndim != 1:
        raise ValueError(
            "candidate_rank must have shape "
            "[candidate]."
        )

    num_candidates = int(
        candidate_rank.numel()
    )

    validate_cell_allocation(
        allocation=allocation,
        num_candidates=num_candidates,
    )

    if (
        candidate_precoder_directions.ndim
        != 4
    ):
        raise ValueError(
            "candidate_precoder_directions must have "
            "shape "
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

    valid_mask = candidate_valid_mask.to(
        device=device,
        dtype=torch.bool,
    )

    ranks = candidate_rank.to(
        device=device,
        dtype=torch.long,
    )

    max_available_modes = int(
        candidate_precoder_directions.shape[
            -2
        ]
    )

    valid_ranks = ranks[
        valid_mask
    ]

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

    #
    # Fast path for the first SDS layer.
    #
    # There is nobody already scheduled, therefore
    # every cross-correlation is exactly zero.
    #
    has_scheduled_candidate = torch.any(
        allocation.candidate_by_user_slot
        != NO_ALLOCATION
    )

    if not bool(
        has_scheduled_candidate.item()
    ):
        return torch.zeros(
            (
                num_candidates,
                allocation.num_rbgs,
            ),
            dtype=real_dtype,
            device=device,
        )

    # ==========================================================
    # 1. Build active spatial-mode masks.
    # ==========================================================
    #
    # rank=1:
    #     [True, False]
    #
    # rank=2:
    #     [True, True]
    #
    mode_indices = torch.arange(
        max_available_modes,
        dtype=torch.long,
        device=device,
    )

    active_mode_mask = (
        mode_indices.unsqueeze(0)
        < ranks.unsqueeze(1)
    )

    active_mode_mask = (
        active_mode_mask
        & valid_mask.unsqueeze(1)
    )

    # ==========================================================
    # 2. Compute ALL pairwise precoder overlaps at once.
    # ==========================================================
    #
    # directions:
    #
    #     [candidate, RBG, mode, TX]
    #
    # overlap:
    #
    #     [RBG,
    #      candidate_i,
    #      candidate_j,
    #      mode_i,
    #      mode_j]
    #
    # overlap[..., i, j, a, b] =
    #
    #       v_(i,a)^H v_(j,b)
    #
    overlap = torch.einsum(
        "crmt,drnt->rcdmn",
        candidate_precoder_directions.conj(),
        candidate_precoder_directions,
    )

    absolute_overlap = torch.abs(
        overlap
    )

    # ==========================================================
    # 3. Apply candidate-i rank.
    # ==========================================================
    #
    # The scalar implementation performs:
    #
    #     absolute_overlap.sum(dim=0)
    #
    # over candidate i's active streams.
    #
    candidate_mode_mask = (
        active_mode_mask[
            None,
            :,
            None,
            :,
            None,
        ]
    )

    absolute_overlap = (
        absolute_overlap
        * candidate_mode_mask
    )

    #
    # Sum candidate-i streams.
    #
    # Result:
    #
    # [RBG, candidate_i, candidate_j, mode_j]
    #
    correlation_by_scheduled_stream = (
        absolute_overlap.sum(
            dim=-2
        )
    )

    # ==========================================================
    # 4. Apply scheduled-candidate rank.
    # ==========================================================

    scheduled_mode_mask = (
        active_mode_mask[
            None,
            None,
            :,
            :,
        ]
    )

    correlation_by_scheduled_stream = (
        correlation_by_scheduled_stream
        * scheduled_mode_mask
    )

    #
    # Original implementation:
    #
    # pair_correlation =
    #     correlation_by_scheduled_stream.max()
    #
    # for each candidate pair.
    #
    # Result:
    #
    # [RBG, candidate_i, candidate_j]
    #
    pair_correlation = (
        correlation_by_scheduled_stream.max(
            dim=-1
        ).values
    )

    # ==========================================================
    # 5. Determine which candidates are already scheduled
    #    on each RBG.
    # ==========================================================
    #
    # allocation:
    #
    #     [user_slot, RBG]
    #
    # scheduled_on_rbg:
    #
    #     [RBG, candidate]
    #
    candidate_indices = torch.arange(
        num_candidates,
        dtype=torch.long,
        device=device,
    )

    scheduled_on_rbg = (
        allocation
        .candidate_by_user_slot
        .transpose(
            0,
            1
        )
        .unsqueeze(-1)
        == candidate_indices[
            None,
            None,
            :,
        ]
    )

    scheduled_on_rbg = (
        scheduled_on_rbg.any(
            dim=1
        )
    )

    # ==========================================================
    # 6. Build candidate-pair comparison mask.
    # ==========================================================
    #
    # A candidate i may compare against j iff:
    #
    #   i is valid
    #   j is valid
    #   j is scheduled on this RBG
    #   i != j
    #
    comparison_mask = (
        scheduled_on_rbg[
            :,
            None,
            :,
        ]
        & valid_mask[
            None,
            :,
            None,
        ]
        & valid_mask[
            None,
            None,
            :,
        ]
    )

    not_self = ~torch.eye(
        num_candidates,
        dtype=torch.bool,
        device=device,
    )

    comparison_mask = (
        comparison_mask
        & not_self.unsqueeze(0)
    )

    # ==========================================================
    # 7. Maximum correlation against already scheduled UEs.
    # ==========================================================
    #
    # All correlations are non-negative.
    # Masked comparisons therefore safely become zero.
    #
    masked_pair_correlation = torch.where(
        comparison_mask,
        pair_correlation,
        torch.zeros_like(
            pair_correlation
        ),
    )

    #
    # Max over scheduled candidate j.
    #
    # [RBG, candidate_i]
    #
    max_correlation = (
        masked_pair_correlation.max(
            dim=-1
        ).values
    )

    #
    # Public API:
    #
    # [candidate, RBG]
    #
    max_correlation = (
        max_correlation
        .transpose(
            0,
            1
        )
        .contiguous()
    )

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


