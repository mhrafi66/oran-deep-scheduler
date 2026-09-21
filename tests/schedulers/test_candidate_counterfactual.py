import torch

from oran_scheduler.schedulers.candidate_counterfactual import (
    compare_fresh_and_stressed_tds,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)


def test_identical_rates_produce_identical_candidates() -> None:
    fresh_rate = torch.tensor(
        [
            [
                [
                    100.0,
                    80.0,
                    60.0,
                    40.0,
                ]
            ]
        ],
        dtype=torch.float32,
    )

    history = torch.tensor(
        [
            [
                [
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                ]
            ]
        ],
        dtype=torch.float32,
    )

    valid = torch.ones_like(
        fresh_rate,
        dtype=torch.bool,
    )

    result = (
        compare_fresh_and_stressed_tds(
            fresh_instantaneous_rate=(
                fresh_rate
            ),
            stressed_instantaneous_rate=(
                fresh_rate.clone()
            ),
            past_average_throughput=(
                history
            ),
            valid_ue_mask=(
                valid
            ),
            config=PFTimeDomainConfig(
                num_candidates=2,
            ),
        )
    )

    torch.testing.assert_close(
        result
        .fresh_result
        .candidate_indices,
        result
        .stressed_result
        .candidate_indices,
    )

    torch.testing.assert_close(
        result.comparison.jaccard,
        torch.tensor(
            [[1.0]]
        ),
    )


def test_rate_stress_changes_candidate_set() -> None:
    fresh_rate = torch.tensor(
        [
            [
                [
                    100.0,
                    90.0,
                    20.0,
                    10.0,
                ]
            ]
        ],
        dtype=torch.float32,
    )

    stressed_rate = torch.tensor(
        [
            [
                [
                    5.0,
                    4.0,
                    100.0,
                    90.0,
                ]
            ]
        ],
        dtype=torch.float32,
    )

    history = torch.full_like(
        fresh_rate,
        fill_value=10.0,
    )

    valid = torch.ones_like(
        fresh_rate,
        dtype=torch.bool,
    )

    result = (
        compare_fresh_and_stressed_tds(
            fresh_instantaneous_rate=(
                fresh_rate
            ),
            stressed_instantaneous_rate=(
                stressed_rate
            ),
            past_average_throughput=(
                history
            ),
            valid_ue_mask=(
                valid
            ),
            config=PFTimeDomainConfig(
                num_candidates=2,
            ),
        )
    )

    expected_fresh = torch.tensor(
        [
            [
                [0, 1]
            ]
        ],
        dtype=torch.long,
    )

    expected_stressed = torch.tensor(
        [
            [
                [2, 3]
            ]
        ],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        result
        .fresh_result
        .candidate_indices,
        expected_fresh,
    )

    torch.testing.assert_close(
        result
        .stressed_result
        .candidate_indices,
        expected_stressed,
    )

    torch.testing.assert_close(
        result
        .comparison
        .jaccard,
        torch.tensor(
            [[0.0]]
        ),
    )

    assert (
        result
        .comparison
        .fresh_top1_retained
        .item()
        is False
    )


def test_pf_history_is_held_constant() -> None:
    fresh_rate = torch.tensor(
        [
            [
                [
                    100.0,
                    90.0,
                    80.0,
                ]
            ]
        ],
        dtype=torch.float32,
    )

    stressed_rate = fresh_rate.clone()

    history = torch.tensor(
        [
            [
                [
                    100.0,
                    1.0,
                    1.0,
                ]
            ]
        ],
        dtype=torch.float32,
    )

    valid = torch.ones_like(
        fresh_rate,
        dtype=torch.bool,
    )

    result = (
        compare_fresh_and_stressed_tds(
            fresh_instantaneous_rate=(
                fresh_rate
            ),
            stressed_instantaneous_rate=(
                stressed_rate
            ),
            past_average_throughput=(
                history
            ),
            valid_ue_mask=(
                valid
            ),
            config=PFTimeDomainConfig(
                num_candidates=2,
            ),
        )
    )

    torch.testing.assert_close(
        result
        .fresh_result
        .candidate_indices,
        result
        .stressed_result
        .candidate_indices,
    )

    torch.testing.assert_close(
        result
        .comparison
        .jaccard,
        torch.tensor(
            [[1.0]]
        ),
    )


def test_invalid_ues_are_never_selected() -> None:
    fresh_rate = torch.tensor(
        [
            [
                [
                    10.0,
                    9999.0,
                    8.0,
                ]
            ]
        ],
        dtype=torch.float32,
    )

    stressed_rate = fresh_rate.clone()

    history = torch.ones_like(
        fresh_rate
    )

    valid = torch.tensor(
        [
            [
                [
                    True,
                    False,
                    True,
                ]
            ]
        ],
        dtype=torch.bool,
    )

    result = (
        compare_fresh_and_stressed_tds(
            fresh_instantaneous_rate=(
                fresh_rate
            ),
            stressed_instantaneous_rate=(
                stressed_rate
            ),
            past_average_throughput=(
                history
            ),
            valid_ue_mask=(
                valid
            ),
            config=PFTimeDomainConfig(
                num_candidates=2,
            ),
        )
    )

    candidates = (
        result
        .fresh_result
        .candidate_indices[
            0,
            0,
        ]
        .tolist()
    )

    assert 1 not in candidates
