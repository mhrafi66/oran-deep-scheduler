from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainResult,
)


@dataclass(frozen=True)
class CandidateSetComparison:
    """
    Compare one reference PF-TDS candidate set against another.

    All tensor outputs have shape:

        [batch, cell]

    unless stated otherwise.

    Terminology
    -----------
    fresh:
        Candidate set generated from the reference/current
        TD-rate information.

    stressed:
        Candidate set generated after applying some stress to
        the TD-rate information.

    Metrics
    -------
    fresh_valid_count:
        Number of valid candidates in the fresh set.

    stressed_valid_count:
        Number of valid candidates in the stressed set.

    overlap_count:
        Number of UE indices appearing in both candidate sets.

    union_count:
        Number of distinct UE indices appearing in either set.

    jaccard:
        |fresh intersect stressed|
        ---------------------------
        |fresh union stressed|

        If both sets are empty, this is defined as 1.

    fresh_recall:
        Fraction of fresh candidates that remain present in the
        stressed set.

        If the fresh set is empty, this is defined as 1.

    fresh_top1_retained:
        True if the highest-ranked valid fresh candidate is still
        present anywhere in the stressed candidate set.

        If the fresh set is empty, this is defined as True.

    mean_rank_displacement:
        For candidates appearing in both sets, mean absolute
        change in candidate-list position.

        Candidate rank is zero based.

        If there is no common candidate, the value is 0 and
        has_common_candidate is False.

    fresh_mean_metric:
        Mean PF metric among valid fresh candidates.

    stressed_mean_metric:
        Mean PF metric among valid stressed candidates.

    top1_metric_ratio:
        stressed top-1 PF metric / fresh top-1 PF metric.

        If the fresh top-1 metric is zero, this is defined as 1
        when both are zero and 0 otherwise.

    has_common_candidate:
        True if at least one valid candidate appears in both sets.
    """

    fresh_valid_count: torch.Tensor
    stressed_valid_count: torch.Tensor

    overlap_count: torch.Tensor
    union_count: torch.Tensor

    jaccard: torch.Tensor
    fresh_recall: torch.Tensor

    fresh_top1_retained: torch.Tensor

    mean_rank_displacement: torch.Tensor
    has_common_candidate: torch.Tensor

    fresh_mean_metric: torch.Tensor
    stressed_mean_metric: torch.Tensor

    top1_metric_ratio: torch.Tensor


def _validate_pf_result(
    result: PFTimeDomainResult,
    *,
    name: str,
) -> None:
    """
    Validate the basic PFTimeDomainResult tensor contract.
    """

    candidate_indices = (
        result.candidate_indices
    )

    candidate_metrics = (
        result.candidate_metrics
    )

    candidate_valid_mask = (
        result.candidate_valid_mask
    )

    if candidate_indices.ndim != 3:
        raise ValueError(
            f"{name}.candidate_indices must have shape "
            "[batch, cell, candidate]."
        )

    expected_shape = tuple(
        candidate_indices.shape
    )

    if tuple(
        candidate_metrics.shape
    ) != expected_shape:
        raise ValueError(
            f"{name}.candidate_metrics must have shape "
            f"{expected_shape}, got "
            f"{tuple(candidate_metrics.shape)}."
        )

    if tuple(
        candidate_valid_mask.shape
    ) != expected_shape:
        raise ValueError(
            f"{name}.candidate_valid_mask must have shape "
            f"{expected_shape}, got "
            f"{tuple(candidate_valid_mask.shape)}."
        )

    if candidate_indices.dtype != torch.long:
        raise TypeError(
            f"{name}.candidate_indices must use torch.long."
        )


def _safe_mean(
    values: torch.Tensor,
) -> float:
    """
    Return a Python float mean, with empty input mapped to zero.
    """

    if values.numel() == 0:
        return 0.0

    return float(
        values.to(
            dtype=torch.float32
        ).mean().item()
    )


def _safe_top1_metric_ratio(
    fresh_metric: float,
    stressed_metric: float,
) -> float:
    """
    Safely form stressed / fresh for top-1 PF metrics.
    """

    if fresh_metric == 0.0:
        if stressed_metric == 0.0:
            return 1.0

        return 0.0

    return (
        stressed_metric
        / fresh_metric
    )


def compare_candidate_sets(
    fresh: PFTimeDomainResult,
    stressed: PFTimeDomainResult,
) -> CandidateSetComparison:
    """
    Compare fresh and stressed PF-TDS candidate selections.

    Both PFTimeDomainResult objects must have identical tensor
    shapes.

    Candidate indices are local indices into the same padded
    per-cell UE population.

    Invalid candidate slots are always ignored through
    candidate_valid_mask.

    The function is diagnostic only. It does not alter either
    PF-TDS result.
    """

    _validate_pf_result(
        fresh,
        name="fresh",
    )

    _validate_pf_result(
        stressed,
        name="stressed",
    )

    expected_shape = tuple(
        fresh.candidate_indices.shape
    )

    if tuple(
        stressed.candidate_indices.shape
    ) != expected_shape:
        raise ValueError(
            "fresh and stressed candidate sets must have "
            "identical shape."
        )

    device = (
        fresh.candidate_indices.device
    )

    if (
        stressed.candidate_indices.device
        != device
    ):
        raise ValueError(
            "fresh and stressed candidate tensors must "
            "be on the same device."
        )

    batch_size = (
        fresh.candidate_indices.shape[0]
    )

    num_cells = (
        fresh.candidate_indices.shape[1]
    )

    output_shape = (
        batch_size,
        num_cells,
    )

    fresh_valid_count = torch.zeros(
        output_shape,
        dtype=torch.long,
        device=device,
    )

    stressed_valid_count = torch.zeros(
        output_shape,
        dtype=torch.long,
        device=device,
    )

    overlap_count = torch.zeros(
        output_shape,
        dtype=torch.long,
        device=device,
    )

    union_count = torch.zeros(
        output_shape,
        dtype=torch.long,
        device=device,
    )

    jaccard = torch.zeros(
        output_shape,
        dtype=torch.float32,
        device=device,
    )

    fresh_recall = torch.zeros(
        output_shape,
        dtype=torch.float32,
        device=device,
    )

    fresh_top1_retained = torch.ones(
        output_shape,
        dtype=torch.bool,
        device=device,
    )

    mean_rank_displacement = torch.zeros(
        output_shape,
        dtype=torch.float32,
        device=device,
    )

    has_common_candidate = torch.zeros(
        output_shape,
        dtype=torch.bool,
        device=device,
    )

    fresh_mean_metric = torch.zeros(
        output_shape,
        dtype=torch.float32,
        device=device,
    )

    stressed_mean_metric = torch.zeros(
        output_shape,
        dtype=torch.float32,
        device=device,
    )

    top1_metric_ratio = torch.ones(
        output_shape,
        dtype=torch.float32,
        device=device,
    )

    fresh_valid_mask = (
        fresh.candidate_valid_mask.to(
            dtype=torch.bool,
        )
    )

    stressed_valid_mask = (
        stressed.candidate_valid_mask.to(
            dtype=torch.bool,
        )
    )

    for batch_index in range(
        batch_size
    ):
        for cell_index in range(
            num_cells
        ):
            fresh_mask = (
                fresh_valid_mask[
                    batch_index,
                    cell_index,
                ]
            )

            stressed_mask = (
                stressed_valid_mask[
                    batch_index,
                    cell_index,
                ]
            )

            fresh_indices = (
                fresh.candidate_indices[
                    batch_index,
                    cell_index,
                ][fresh_mask]
            )

            stressed_indices = (
                stressed.candidate_indices[
                    batch_index,
                    cell_index,
                ][stressed_mask]
            )

            fresh_metrics = (
                fresh.candidate_metrics[
                    batch_index,
                    cell_index,
                ][fresh_mask]
            )

            stressed_metrics = (
                stressed.candidate_metrics[
                    batch_index,
                    cell_index,
                ][stressed_mask]
            )

            fresh_values = [
                int(value)
                for value
                in fresh_indices.tolist()
            ]

            stressed_values = [
                int(value)
                for value
                in stressed_indices.tolist()
            ]

            fresh_set = set(
                fresh_values
            )

            stressed_set = set(
                stressed_values
            )

            common_set = (
                fresh_set
                & stressed_set
            )

            union_set = (
                fresh_set
                | stressed_set
            )

            num_fresh = len(
                fresh_set
            )

            num_stressed = len(
                stressed_set
            )

            num_common = len(
                common_set
            )

            num_union = len(
                union_set
            )

            fresh_valid_count[
                batch_index,
                cell_index,
            ] = num_fresh

            stressed_valid_count[
                batch_index,
                cell_index,
            ] = num_stressed

            overlap_count[
                batch_index,
                cell_index,
            ] = num_common

            union_count[
                batch_index,
                cell_index,
            ] = num_union

            if num_union == 0:
                jaccard_value = 1.0
            else:
                jaccard_value = (
                    num_common
                    / num_union
                )

            jaccard[
                batch_index,
                cell_index,
            ] = jaccard_value

            if num_fresh == 0:
                recall_value = 1.0
            else:
                recall_value = (
                    num_common
                    / num_fresh
                )

            fresh_recall[
                batch_index,
                cell_index,
            ] = recall_value

            if fresh_values:
                fresh_top1 = (
                    fresh_values[0]
                )

                fresh_top1_retained[
                    batch_index,
                    cell_index,
                ] = (
                    fresh_top1
                    in stressed_set
                )

            if common_set:
                has_common_candidate[
                    batch_index,
                    cell_index,
                ] = True

                fresh_rank = {
                    candidate: rank
                    for rank, candidate
                    in enumerate(
                        fresh_values
                    )
                }

                stressed_rank = {
                    candidate: rank
                    for rank, candidate
                    in enumerate(
                        stressed_values
                    )
                }

                displacement = [
                    abs(
                        fresh_rank[
                            candidate
                        ]
                        - stressed_rank[
                            candidate
                        ]
                    )
                    for candidate
                    in common_set
                ]

                mean_rank_displacement[
                    batch_index,
                    cell_index,
                ] = (
                    sum(
                        displacement
                    )
                    / len(
                        displacement
                    )
                )

            fresh_mean_metric[
                batch_index,
                cell_index,
            ] = _safe_mean(
                fresh_metrics
            )

            stressed_mean_metric[
                batch_index,
                cell_index,
            ] = _safe_mean(
                stressed_metrics
            )

            fresh_top1_metric = (
                float(
                    fresh_metrics[0].item()
                )
                if fresh_metrics.numel() > 0
                else 0.0
            )

            stressed_top1_metric = (
                float(
                    stressed_metrics[0].item()
                )
                if stressed_metrics.numel() > 0
                else 0.0
            )

            top1_metric_ratio[
                batch_index,
                cell_index,
            ] = (
                _safe_top1_metric_ratio(
                    fresh_metric=(
                        fresh_top1_metric
                    ),
                    stressed_metric=(
                        stressed_top1_metric
                    ),
                )
            )

    return CandidateSetComparison(
        fresh_valid_count=(
            fresh_valid_count
        ),
        stressed_valid_count=(
            stressed_valid_count
        ),
        overlap_count=(
            overlap_count
        ),
        union_count=(
            union_count
        ),
        jaccard=jaccard,
        fresh_recall=(
            fresh_recall
        ),
        fresh_top1_retained=(
            fresh_top1_retained
        ),
        mean_rank_displacement=(
            mean_rank_displacement
        ),
        has_common_candidate=(
            has_common_candidate
        ),
        fresh_mean_metric=(
            fresh_mean_metric
        ),
        stressed_mean_metric=(
            stressed_mean_metric
        ),
        top1_metric_ratio=(
            top1_metric_ratio
        ),
    )


def summarize_candidate_comparison(
    comparison: CandidateSetComparison,
) -> dict[str, float]:
    """
    Reduce per-cell candidate diagnostics to scalar summaries.

    Intended primarily for CSV logging and experiment reports.
    """

    top1_retained = (
        comparison
        .fresh_top1_retained
        .to(
            dtype=torch.float32
        )
    )

    common_mask = (
        comparison
        .has_common_candidate
    )

    if torch.any(
        common_mask
    ):
        mean_rank_displacement = float(
            comparison
            .mean_rank_displacement[
                common_mask
            ]
            .mean()
            .item()
        )
    else:
        mean_rank_displacement = 0.0

    return {
        "candidate_jaccard_mean": float(
            comparison
            .jaccard
            .mean()
            .item()
        ),
        "candidate_fresh_recall_mean": float(
            comparison
            .fresh_recall
            .mean()
            .item()
        ),
        "candidate_top1_retention_rate": float(
            top1_retained
            .mean()
            .item()
        ),
        "candidate_rank_displacement_mean": (
            mean_rank_displacement
        ),
        "candidate_fresh_count_mean": float(
            comparison
            .fresh_valid_count
            .to(
                dtype=torch.float32
            )
            .mean()
            .item()
        ),
        "candidate_stressed_count_mean": float(
            comparison
            .stressed_valid_count
            .to(
                dtype=torch.float32
            )
            .mean()
            .item()
        ),
        "candidate_fresh_pf_mean": float(
            comparison
            .fresh_mean_metric
            .mean()
            .item()
        ),
        "candidate_stressed_pf_mean": float(
            comparison
            .stressed_mean_metric
            .mean()
            .item()
        ),
        "candidate_top1_pf_ratio_mean": float(
            comparison
            .top1_metric_ratio
            .mean()
            .item()
        ),
    }
