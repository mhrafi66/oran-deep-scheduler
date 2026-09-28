import torch

from oran_scheduler.schedulers.candidate_diagnostics import (
    compare_candidate_sets,
    summarize_candidate_comparison,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainResult,
)


def _make_result(
    indices,
    metrics,
    valid,
) -> PFTimeDomainResult:
    return PFTimeDomainResult(
        candidate_indices=torch.tensor(
            indices,
            dtype=torch.long,
        ),
        candidate_metrics=torch.tensor(
            metrics,
            dtype=torch.float32,
        ),
        candidate_valid_mask=torch.tensor(
            valid,
            dtype=torch.bool,
        ),
    )


def test_identical_candidate_sets() -> None:
    fresh = _make_result(
        indices=[
            [
                [7, 12, 3, 18]
            ]
        ],
        metrics=[
            [
                [10.0, 8.0, 6.0, 4.0]
            ]
        ],
        valid=[
            [
                [True, True, True, True]
            ]
        ],
    )

    stressed = _make_result(
        indices=[
            [
                [7, 12, 3, 18]
            ]
        ],
        metrics=[
            [
                [10.0, 8.0, 6.0, 4.0]
            ]
        ],
        valid=[
            [
                [True, True, True, True]
            ]
        ],
    )

    result = compare_candidate_sets(
        fresh=fresh,
        stressed=stressed,
    )

    assert (
        result
        .overlap_count
        .item()
        == 4
    )

    assert (
        result
        .union_count
        .item()
        == 4
    )

    torch.testing.assert_close(
        result.jaccard,
        torch.tensor(
            [[1.0]]
        ),
    )

    torch.testing.assert_close(
        result.fresh_recall,
        torch.tensor(
            [[1.0]]
        ),
    )

    assert (
        result
        .fresh_top1_retained
        .item()
        is True
    )

    torch.testing.assert_close(
        result.mean_rank_displacement,
        torch.tensor(
            [[0.0]]
        ),
    )

    torch.testing.assert_close(
        result.top1_metric_ratio,
        torch.tensor(
            [[1.0]]
        ),
    )


def test_partial_overlap() -> None:
    fresh = _make_result(
        indices=[
            [
                [1, 2, 3, 4]
            ]
        ],
        metrics=[
            [
                [10.0, 8.0, 6.0, 4.0]
            ]
        ],
        valid=[
            [
                [True, True, True, True]
            ]
        ],
    )

    stressed = _make_result(
        indices=[
            [
                [2, 4, 5, 6]
            ]
        ],
        metrics=[
            [
                [9.0, 7.0, 5.0, 3.0]
            ]
        ],
        valid=[
            [
                [True, True, True, True]
            ]
        ],
    )

    result = compare_candidate_sets(
        fresh=fresh,
        stressed=stressed,
    )

    assert (
        result
        .overlap_count
        .item()
        == 2
    )

    assert (
        result
        .union_count
        .item()
        == 6
    )

    torch.testing.assert_close(
        result.jaccard,
        torch.tensor(
            [[2.0 / 6.0]]
        ),
    )

    torch.testing.assert_close(
        result.fresh_recall,
        torch.tensor(
            [[0.5]]
        ),
    )

    assert (
        result
        .fresh_top1_retained
        .item()
        is False
    )

    torch.testing.assert_close(
        result.mean_rank_displacement,
        torch.tensor(
            [[1.5]]
        ),
    )

    assert (
        result
        .has_common_candidate
        .item()
        is True
    )

    torch.testing.assert_close(
        result.top1_metric_ratio,
        torch.tensor(
            [[0.9]]
        ),
    )


def test_padding_is_ignored() -> None:
    fresh = _make_result(
        indices=[
            [
                [4, 9, 0, 0]
            ]
        ],
        metrics=[
            [
                [10.0, 8.0, 0.0, 0.0]
            ]
        ],
        valid=[
            [
                [True, True, False, False]
            ]
        ],
    )

    stressed = _make_result(
        indices=[
            [
                [9, 4, 123, 456]
            ]
        ],
        metrics=[
            [
                [9.0, 7.0, 0.0, 0.0]
            ]
        ],
        valid=[
            [
                [True, True, False, False]
            ]
        ],
    )

    result = compare_candidate_sets(
        fresh=fresh,
        stressed=stressed,
    )

    assert (
        result
        .fresh_valid_count
        .item()
        == 2
    )

    assert (
        result
        .stressed_valid_count
        .item()
        == 2
    )

    assert (
        result
        .overlap_count
        .item()
        == 2
    )

    assert (
        result
        .union_count
        .item()
        == 2
    )

    torch.testing.assert_close(
        result.jaccard,
        torch.tensor(
            [[1.0]]
        ),
    )

    torch.testing.assert_close(
        result.mean_rank_displacement,
        torch.tensor(
            [[1.0]]
        ),
    )


def test_no_common_candidates() -> None:
    fresh = _make_result(
        indices=[
            [
                [1, 2]
            ]
        ],
        metrics=[
            [
                [10.0, 8.0]
            ]
        ],
        valid=[
            [
                [True, True]
            ]
        ],
    )

    stressed = _make_result(
        indices=[
            [
                [3, 4]
            ]
        ],
        metrics=[
            [
                [7.0, 5.0]
            ]
        ],
        valid=[
            [
                [True, True]
            ]
        ],
    )

    result = compare_candidate_sets(
        fresh=fresh,
        stressed=stressed,
    )

    assert (
        result
        .overlap_count
        .item()
        == 0
    )

    torch.testing.assert_close(
        result.jaccard,
        torch.tensor(
            [[0.0]]
        ),
    )

    torch.testing.assert_close(
        result.fresh_recall,
        torch.tensor(
            [[0.0]]
        ),
    )

    assert (
        result
        .fresh_top1_retained
        .item()
        is False
    )

    assert (
        result
        .has_common_candidate
        .item()
        is False
    )

    torch.testing.assert_close(
        result.mean_rank_displacement,
        torch.tensor(
            [[0.0]]
        ),
    )


def test_both_sets_empty() -> None:
    fresh = _make_result(
        indices=[
            [
                [0, 0]
            ]
        ],
        metrics=[
            [
                [0.0, 0.0]
            ]
        ],
        valid=[
            [
                [False, False]
            ]
        ],
    )

    stressed = _make_result(
        indices=[
            [
                [0, 0]
            ]
        ],
        metrics=[
            [
                [0.0, 0.0]
            ]
        ],
        valid=[
            [
                [False, False]
            ]
        ],
    )

    result = compare_candidate_sets(
        fresh=fresh,
        stressed=stressed,
    )

    torch.testing.assert_close(
        result.jaccard,
        torch.tensor(
            [[1.0]]
        ),
    )

    torch.testing.assert_close(
        result.fresh_recall,
        torch.tensor(
            [[1.0]]
        ),
    )

    assert (
        result
        .fresh_top1_retained
        .item()
        is True
    )


def test_summary_reduces_cells() -> None:
    fresh = _make_result(
        indices=[
            [
                [1, 2],
                [3, 4],
            ]
        ],
        metrics=[
            [
                [10.0, 8.0],
                [9.0, 7.0],
            ]
        ],
        valid=[
            [
                [True, True],
                [True, True],
            ]
        ],
    )

    stressed = _make_result(
        indices=[
            [
                [1, 2],
                [4, 5],
            ]
        ],
        metrics=[
            [
                [10.0, 8.0],
                [8.0, 6.0],
            ]
        ],
        valid=[
            [
                [True, True],
                [True, True],
            ]
        ],
    )

    comparison = compare_candidate_sets(
        fresh=fresh,
        stressed=stressed,
    )

    summary = (
        summarize_candidate_comparison(
            comparison
        )
    )

    assert (
        "candidate_jaccard_mean"
        in summary
    )

    assert (
        "candidate_fresh_recall_mean"
        in summary
    )

    assert (
        "candidate_top1_retention_rate"
        in summary
    )

    assert (
        summary[
            "candidate_fresh_count_mean"
        ]
        == 2.0
    )
