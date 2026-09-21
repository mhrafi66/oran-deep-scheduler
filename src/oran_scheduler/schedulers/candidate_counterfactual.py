from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.candidate_diagnostics import (
    CandidateSetComparison,
    compare_candidate_sets,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    PFTimeDomainResult,
    run_pf_tds,
)


@dataclass(frozen=True)
class CandidateCounterfactualResult:
    """
    Fresh-versus-stressed PF-TDS counterfactual.

    fresh_result:
        Candidate set obtained from the reference/current
        instantaneous TD rates.

    stressed_result:
        Candidate set obtained from stressed instantaneous
        TD rates.

    comparison:
        Candidate-set diagnostics comparing the two results.
    """

    fresh_result: PFTimeDomainResult

    stressed_result: PFTimeDomainResult

    comparison: CandidateSetComparison


def _validate_counterfactual_inputs(
    fresh_instantaneous_rate: torch.Tensor,
    stressed_instantaneous_rate: torch.Tensor,
    past_average_throughput: torch.Tensor,
    valid_ue_mask: torch.Tensor,
) -> None:
    """
    Validate fresh/stressed PF-TDS counterfactual inputs.
    """

    if fresh_instantaneous_rate.ndim != 3:
        raise ValueError(
            "fresh_instantaneous_rate must have shape "
            "[batch, cell, UE]."
        )

    expected_shape = tuple(
        fresh_instantaneous_rate.shape
    )

    tensors = {
        "stressed_instantaneous_rate": (
            stressed_instantaneous_rate
        ),
        "past_average_throughput": (
            past_average_throughput
        ),
        "valid_ue_mask": (
            valid_ue_mask
        ),
    }

    for name, tensor in tensors.items():
        if tuple(
            tensor.shape
        ) != expected_shape:
            raise ValueError(
                f"{name} must have shape "
                f"{expected_shape}, got "
                f"{tuple(tensor.shape)}."
            )

    if torch.any(
        fresh_instantaneous_rate < 0
    ):
        raise ValueError(
            "Fresh instantaneous rates cannot be negative."
        )

    if torch.any(
        stressed_instantaneous_rate < 0
    ):
        raise ValueError(
            "Stressed instantaneous rates cannot be negative."
        )

    if torch.any(
        past_average_throughput < 0
    ):
        raise ValueError(
            "Past-average throughput cannot be negative."
        )


def compare_fresh_and_stressed_tds(
    fresh_instantaneous_rate: torch.Tensor,
    stressed_instantaneous_rate: torch.Tensor,
    past_average_throughput: torch.Tensor,
    valid_ue_mask: torch.Tensor,
    config: PFTimeDomainConfig,
) -> CandidateCounterfactualResult:
    """
    Run PF-TDS under fresh and stressed TD-rate observations.

    This helper is diagnostic only.

    The fresh and stressed candidate sets are produced using:

        identical throughput history
        identical UE-validity mask
        identical PF-TDS configuration

    The only intended difference is the instantaneous TD-rate
    tensor.

    This isolates how errors in PF-TDS rate information change
    the candidate list presented to downstream FDS/SDS.
    """

    _validate_counterfactual_inputs(
        fresh_instantaneous_rate=(
            fresh_instantaneous_rate
        ),
        stressed_instantaneous_rate=(
            stressed_instantaneous_rate
        ),
        past_average_throughput=(
            past_average_throughput
        ),
        valid_ue_mask=(
            valid_ue_mask
        ),
    )

    valid_ue_mask = (
        valid_ue_mask.to(
            dtype=torch.bool
        )
    )

    fresh_result = run_pf_tds(
        instantaneous_rate=(
            fresh_instantaneous_rate
        ),
        past_average_throughput=(
            past_average_throughput
        ),
        config=config,
        valid_ue_mask=(
            valid_ue_mask
        ),
    )

    stressed_result = run_pf_tds(
        instantaneous_rate=(
            stressed_instantaneous_rate
        ),
        past_average_throughput=(
            past_average_throughput
        ),
        config=config,
        valid_ue_mask=(
            valid_ue_mask
        ),
    )

    comparison = compare_candidate_sets(
        fresh=fresh_result,
        stressed=stressed_result,
    )

    return CandidateCounterfactualResult(
        fresh_result=fresh_result,
        stressed_result=stressed_result,
        comparison=comparison,
    )
