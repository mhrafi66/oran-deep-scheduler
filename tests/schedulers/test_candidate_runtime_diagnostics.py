import pytest
import torch

from oran_scheduler.schedulers.candidate_runtime_diagnostics import (
    build_candidate_runtime_diagnostics,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    run_pf_tds,
)


def _actual_stressed_result(
    *,
    stressed_rate: torch.Tensor,
    history: torch.Tensor,
    valid: torch.Tensor,
    config: PFTimeDomainConfig,
):
    return run_pf_tds(
        instantaneous_rate=(
            stressed_rate[
                None,
                None,
                :,
            ]
        ),
        past_average_throughput=(
            history[
                None,
                None,
                :,
            ]
        ),
        config=config,
        valid_ue_mask=(
            valid[
                None,
                None,
                :,
            ]
        ),
    )


def test_identical_rates_have_perfect_retention() -> None:
    fresh = torch.tensor(
        [
            100.0,
            90.0,
            20.0,
            10.0,
        ],
        dtype=torch.float32,
    )

    stressed = fresh.clone()

    history = torch.full_like(
        fresh,
        10.0,
    )

    valid = torch.ones_like(
        fresh,
        dtype=torch.bool,
    )

    config = PFTimeDomainConfig(
        num_candidates=2,
    )

    actual = _actual_stressed_result(
        stressed_rate=stressed,
        history=history,
        valid=valid,
        config=config,
    )

    diagnostics = (
        build_candidate_runtime_diagnostics(
            fresh_instantaneous_rate_bps=(
                fresh
            ),
            stressed_instantaneous_rate_bps=(
                stressed
            ),
            past_average_throughput_bps=(
                history
            ),
            eligible_mask=valid,
            config=config,
            actual_stressed_result=(
                actual
            ),
        )
    )

    assert diagnostics[
        "candidate_jaccard"
    ] == pytest.approx(
        1.0
    )

    assert diagnostics[
        "candidate_fresh_recall"
    ] == pytest.approx(
        1.0
    )

    assert diagnostics[
        "candidate_top1_retained"
    ] == pytest.approx(
        1.0
    )

    assert diagnostics[
        "candidate_set_changed"
    ] == pytest.approx(
        0.0
    )

    assert diagnostics[
        "candidate_order_exact_match"
    ] == pytest.approx(
        1.0
    )

    assert diagnostics[
        "candidate_fresh_truth_pf_retention"
    ] == pytest.approx(
        1.0
    )

    assert diagnostics[
        "candidate_stressed_top1_fresh_pf_ratio"
    ] == pytest.approx(
        1.0
    )


def test_bad_stressed_candidates_are_scored_under_fresh_truth() -> None:
    #
    # Fresh PF ordering:
    #
    #   UE0 = 10
    #   UE1 = 9
    #   UE2 = 2
    #   UE3 = 1
    #
    fresh = torch.tensor(
        [
            100.0,
            90.0,
            20.0,
            10.0,
        ],
        dtype=torch.float32,
    )

    #
    # Corrupted observation instead makes:
    #
    #   UE2 = 10
    #   UE3 = 9
    #
    stressed = torch.tensor(
        [
            5.0,
            4.0,
            100.0,
            90.0,
        ],
        dtype=torch.float32,
    )

    history = torch.full_like(
        fresh,
        10.0,
    )

    valid = torch.ones_like(
        fresh,
        dtype=torch.bool,
    )

    config = PFTimeDomainConfig(
        num_candidates=2,
    )

    actual = _actual_stressed_result(
        stressed_rate=stressed,
        history=history,
        valid=valid,
        config=config,
    )

    diagnostics = (
        build_candidate_runtime_diagnostics(
            fresh_instantaneous_rate_bps=(
                fresh
            ),
            stressed_instantaneous_rate_bps=(
                stressed
            ),
            past_average_throughput_bps=(
                history
            ),
            eligible_mask=valid,
            config=config,
            actual_stressed_result=(
                actual
            ),
        )
    )

    assert diagnostics[
        "candidate_jaccard"
    ] == pytest.approx(
        0.0
    )

    assert diagnostics[
        "candidate_fresh_recall"
    ] == pytest.approx(
        0.0
    )

    assert diagnostics[
        "candidate_top1_retained"
    ] == pytest.approx(
        0.0
    )

    assert diagnostics[
        "candidate_set_changed"
    ] == pytest.approx(
        1.0
    )

    #
    # Fresh selected-set mean:
    #
    #     (10 + 9) / 2 = 9.5
    #
    assert diagnostics[
        "candidate_fresh_truth_pf_mean"
    ] == pytest.approx(
        9.5
    )

    #
    # Stressed selection = UE2 + UE3.
    #
    # Under FRESH truth:
    #
    #     (2 + 1) / 2 = 1.5
    #
    assert diagnostics[
        "candidate_stressed_set_fresh_pf_mean"
    ] == pytest.approx(
        1.5
    )

    assert diagnostics[
        "candidate_fresh_truth_pf_retention"
    ] == pytest.approx(
        1.5 / 9.5
    )

    #
    # Stressed top-1 is UE2.
    #
    # Fresh PF score:
    #
    #     UE2 / true fresh top1
    #       =
    #     2 / 10
    #
    assert diagnostics[
        "candidate_stressed_top1_fresh_pf_ratio"
    ] == pytest.approx(
        0.2
    )


def test_runtime_diagnostic_checks_actual_scheduler_result() -> None:
    fresh = torch.tensor(
        [
            100.0,
            90.0,
            20.0,
        ],
        dtype=torch.float32,
    )

    stressed = fresh.clone()

    history = torch.ones_like(
        fresh
    )

    valid = torch.ones_like(
        fresh,
        dtype=torch.bool,
    )

    config = PFTimeDomainConfig(
        num_candidates=2,
    )

    #
    # Deliberately construct the "actual" result from
    # DIFFERENT rates.
    #
    wrong_stressed = torch.tensor(
        [
            1.0,
            2.0,
            1000.0,
        ],
        dtype=torch.float32,
    )

    wrong_actual = _actual_stressed_result(
        stressed_rate=(
            wrong_stressed
        ),
        history=history,
        valid=valid,
        config=config,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"do(?:es)? not match the actual "
            r"PF-TDS result"
        ),
    ):
        build_candidate_runtime_diagnostics(
            fresh_instantaneous_rate_bps=(
                fresh
            ),
            stressed_instantaneous_rate_bps=(
                stressed
            ),
            past_average_throughput_bps=(
                history
            ),
            eligible_mask=valid,
            config=config,
            actual_stressed_result=(
                wrong_actual
            ),
        )
