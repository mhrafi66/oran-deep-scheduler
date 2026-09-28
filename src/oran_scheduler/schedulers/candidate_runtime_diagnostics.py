from __future__ import annotations

import torch

from oran_scheduler.schedulers.candidate_counterfactual import (
    compare_fresh_and_stressed_tds,
)
from oran_scheduler.schedulers.candidate_diagnostics import (
    summarize_candidate_comparison,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    PFTimeDomainResult,
    compute_pf_metric,
)


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """
    Form a diagnostic ratio without creating NaN/Inf.

    When both quantities are zero, treat the ratio as
    perfect retention (= 1). If only the denominator is
    zero, return zero.
    """

    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0

        return 0.0

    return numerator / denominator


def _masked_candidate_mean(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """
    Mean over valid candidate positions.

    Both tensors must have candidate-set shape:

        [batch, cell, candidate]
    """

    valid_values = values[
        valid_mask
    ]

    if valid_values.numel() == 0:
        return 0.0

    return float(
        valid_values
        .to(
            dtype=torch.float32
        )
        .mean()
        .item()
    )


def _masked_candidate_top1(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """
    Return the first valid ranked candidate's value.

    The PF-TDS result is sorted in descending PF order.
    """

    if values.ndim != 3:
        raise ValueError(
            "values must have shape "
            "[batch, cell, candidate]."
        )

    if tuple(
        valid_mask.shape
    ) != tuple(
        values.shape
    ):
        raise ValueError(
            "valid_mask must match values."
        )

    #
    # This runtime diagnostic currently represents
    # exactly one cell:
    #
    #     [1, 1, candidate]
    #
    if (
        values.shape[0] != 1
        or values.shape[1] != 1
    ):
        raise ValueError(
            "Runtime candidate diagnostics expect "
            "one [batch=1, cell=1] candidate set."
        )

    valid = valid_mask[
        0,
        0,
    ]

    if not torch.any(
        valid
    ):
        return 0.0

    first_valid_position = int(
        torch.nonzero(
            valid,
            as_tuple=False,
        )[
            0,
            0,
        ]
        .item()
    )

    return float(
        values[
            0,
            0,
            first_valid_position,
        ]
        .item()
    )


def _validate_actual_stressed_result(
    *,
    counterfactual: PFTimeDomainResult,
    actual: PFTimeDomainResult,
) -> None:
    """
    Prove that our diagnostic stressed PF-TDS call exactly
    reproduces the candidate set actually used by PPO.

    If this fails, the diagnostic is not measuring the real
    scheduling path and the experiment should stop immediately.
    """

    if not torch.equal(
        counterfactual
        .candidate_valid_mask,
        actual
        .candidate_valid_mask,
    ):
        raise RuntimeError(
            "Candidate diagnostic valid mask does not "
            "match the actual PF-TDS result."
        )

    if not torch.equal(
        counterfactual
        .candidate_indices,
        actual
        .candidate_indices,
    ):
        raise RuntimeError(
            "Candidate diagnostic indices do not "
            "match the actual PF-TDS result."
        )

    if not torch.allclose(
        counterfactual
        .candidate_metrics,
        actual
        .candidate_metrics,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(
            "Candidate diagnostic PF metrics do not "
            "match the actual PF-TDS result."
        )


def build_candidate_runtime_diagnostics(
    *,
    fresh_instantaneous_rate_bps: torch.Tensor,
    stressed_instantaneous_rate_bps: torch.Tensor,
    past_average_throughput_bps: torch.Tensor,
    eligible_mask: torch.Tensor,
    config: PFTimeDomainConfig,
    actual_stressed_result: PFTimeDomainResult,
) -> dict[str, float]:
    """
    Build one cell's PF-TDS candidate diagnostics.

    Input shapes:

        fresh_instantaneous_rate_bps:
            [serving_ue]

        stressed_instantaneous_rate_bps:
            [serving_ue]

        past_average_throughput_bps:
            [serving_ue]

        eligible_mask:
            [serving_ue]

    The actual PPO candidate set was generated from the
    stressed rate.

    We rerun PF-TDS twice while holding constant:

        * pre-TTI PF throughput history
        * TDS eligibility
        * PF configuration
        * serving-UE population

    Only the instantaneous TD-rate information differs.

    Two different notions of PF quality are reported:

    1. reported/stressed PF quality:
         what the corrupted scheduler believes;

    2. fresh-truth PF quality:
         how the stressed candidate set scores under the
         uncorrupted/current TD-rate observation.

    The second quantity is the more useful indication of
    candidate-selection damage.
    """

    tensors = {
        "fresh_instantaneous_rate_bps": (
            fresh_instantaneous_rate_bps
        ),
        "stressed_instantaneous_rate_bps": (
            stressed_instantaneous_rate_bps
        ),
        "past_average_throughput_bps": (
            past_average_throughput_bps
        ),
        "eligible_mask": eligible_mask,
    }

    expected_shape = tuple(
        fresh_instantaneous_rate_bps.shape
    )

    if len(
        expected_shape
    ) != 1:
        raise ValueError(
            "Runtime diagnostic inputs must have "
            "shape [serving_ue]."
        )

    for name, tensor in tensors.items():
        if tuple(
            tensor.shape
        ) != expected_shape:
            raise ValueError(
                f"{name} must have shape "
                f"{expected_shape}, got "
                f"{tuple(tensor.shape)}."
            )

    if eligible_mask.dtype != torch.bool:
        raise ValueError(
            "eligible_mask must use torch.bool."
        )

    device = (
        fresh_instantaneous_rate_bps
        .device
    )

    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(
                f"{name} is on the wrong device."
            )

    fresh_rate = (
        fresh_instantaneous_rate_bps[
            None,
            None,
            :,
        ]
    )

    stressed_rate = (
        stressed_instantaneous_rate_bps[
            None,
            None,
            :,
        ]
    )

    history = (
        past_average_throughput_bps[
            None,
            None,
            :,
        ]
    )

    valid = (
        eligible_mask[
            None,
            None,
            :,
        ]
    )

    counterfactual = (
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
            valid_ue_mask=valid,
            config=config,
        )
    )

    #
    # This is a critical correctness assertion.
    #
    # The "stressed" diagnostic calculation must be
    # exactly the same PF-TDS candidate selection that
    # the real PPO scheduling path actually used.
    #
    _validate_actual_stressed_result(
        counterfactual=(
            counterfactual
            .stressed_result
        ),
        actual=(
            actual_stressed_result
        ),
    )

    comparison_summary = (
        summarize_candidate_comparison(
            counterfactual
            .comparison
        )
    )

    #
    # Compute PF metric under CURRENT/FRESH TD-rate
    # information for every serving UE.
    #
    fresh_pf_metric = compute_pf_metric(
        instantaneous_rate=fresh_rate,
        past_average_throughput=history,
        config=config,
    )

    fresh_result = (
        counterfactual
        .fresh_result
    )

    stressed_result = (
        counterfactual
        .stressed_result
    )

    #
    # Fresh set scored under fresh PF truth.
    #
    fresh_set_fresh_pf = torch.gather(
        fresh_pf_metric,
        dim=-1,
        index=(
            fresh_result
            .candidate_indices
        ),
    )

    fresh_set_fresh_pf = torch.where(
        fresh_result
        .candidate_valid_mask,
        fresh_set_fresh_pf,
        torch.zeros_like(
            fresh_set_fresh_pf
        ),
    )

    #
    # The ACTUALLY USED stressed candidate set,
    # but scored using fresh/current PF truth.
    #
    stressed_set_fresh_pf = torch.gather(
        fresh_pf_metric,
        dim=-1,
        index=(
            stressed_result
            .candidate_indices
        ),
    )

    stressed_set_fresh_pf = torch.where(
        stressed_result
        .candidate_valid_mask,
        stressed_set_fresh_pf,
        torch.zeros_like(
            stressed_set_fresh_pf
        ),
    )

    fresh_truth_mean = (
        _masked_candidate_mean(
            fresh_set_fresh_pf,
            fresh_result
            .candidate_valid_mask,
        )
    )

    stressed_set_fresh_truth_mean = (
        _masked_candidate_mean(
            stressed_set_fresh_pf,
            stressed_result
            .candidate_valid_mask,
        )
    )

    fresh_truth_top1 = (
        _masked_candidate_top1(
            fresh_set_fresh_pf,
            fresh_result
            .candidate_valid_mask,
        )
    )

    stressed_top1_fresh_truth = (
        _masked_candidate_top1(
            stressed_set_fresh_pf,
            stressed_result
            .candidate_valid_mask,
        )
    )

    candidate_set_changed = float(
        comparison_summary[
            "candidate_jaccard_mean"
        ]
        < (
            1.0
            - 1.0e-12
        )
    )

    candidate_order_exact_match = float(
        torch.equal(
            fresh_result
            .candidate_valid_mask,
            stressed_result
            .candidate_valid_mask,
        )
        and torch.equal(
            fresh_result
            .candidate_indices,
            stressed_result
            .candidate_indices,
        )
    )

    has_common_candidate = float(
        counterfactual
        .comparison
        .has_common_candidate[
            0,
            0,
        ]
        .item()
    )

    return {
        "candidate_eligible_count": float(
            eligible_mask
            .sum()
            .item()
        ),

        "candidate_fresh_count": (
            comparison_summary[
                "candidate_fresh_count_mean"
            ]
        ),

        "candidate_stressed_count": (
            comparison_summary[
                "candidate_stressed_count_mean"
            ]
        ),

        "candidate_jaccard": (
            comparison_summary[
                "candidate_jaccard_mean"
            ]
        ),

        "candidate_fresh_recall": (
            comparison_summary[
                "candidate_fresh_recall_mean"
            ]
        ),

        "candidate_top1_retained": (
            comparison_summary[
                "candidate_top1_retention_rate"
            ]
        ),

        "candidate_has_common": (
            has_common_candidate
        ),

        "candidate_rank_displacement": (
            comparison_summary[
                "candidate_rank_displacement_mean"
            ]
        ),

        "candidate_set_changed": (
            candidate_set_changed
        ),

        "candidate_order_exact_match": (
            candidate_order_exact_match
        ),

        #
        # These two values come from each scheduler's
        # OWN observation.
        #
        "candidate_fresh_reported_pf_mean": (
            comparison_summary[
                "candidate_fresh_pf_mean"
            ]
        ),

        "candidate_stressed_reported_pf_mean": (
            comparison_summary[
                "candidate_stressed_pf_mean"
            ]
        ),

        "candidate_reported_top1_pf_ratio": (
            comparison_summary[
                "candidate_top1_pf_ratio_mean"
            ]
        ),

        #
        # These are the scientifically stronger
        # counterfactual measurements:
        #
        # both candidate sets evaluated using the SAME
        # fresh/current PF truth.
        #
        "candidate_fresh_truth_pf_mean": (
            fresh_truth_mean
        ),

        "candidate_stressed_set_fresh_pf_mean": (
            stressed_set_fresh_truth_mean
        ),

        "candidate_fresh_truth_pf_retention": (
            _safe_ratio(
                stressed_set_fresh_truth_mean,
                fresh_truth_mean,
            )
        ),

        "candidate_stressed_top1_fresh_pf_ratio": (
            _safe_ratio(
                stressed_top1_fresh_truth,
                fresh_truth_top1,
            )
        ),
    }
