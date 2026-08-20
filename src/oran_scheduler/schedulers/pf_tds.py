from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class PFTimeDomainConfig:
    """Configuration for proportional-fair time-domain scheduling."""

    num_candidates: int = 10

    # Small positive value used only to prevent division by zero.
    denominator_epsilon: float = 1e-8

def compute_pf_metric(
    instantaneous_rate: torch.Tensor,
    past_average_throughput: torch.Tensor,
    config: PFTimeDomainConfig,
) -> torch.Tensor:
    """
    Compute the proportional-fair metric.

    PF metric:

        M_u = instantaneous_rate_u / past_average_throughput_u

    Expected tensor shape:

        [batch, cell, UE]

    Example training shape:

        [1, 21, 20]

    Returns a tensor with the same shape.
    """

    if instantaneous_rate.shape != past_average_throughput.shape:
        raise ValueError(
            "instantaneous_rate and past_average_throughput "
            "must have identical shapes. "
            f"Got {tuple(instantaneous_rate.shape)} and "
            f"{tuple(past_average_throughput.shape)}."
        )

    safe_history = torch.clamp(
        past_average_throughput,
        min=config.denominator_epsilon,
    )

    pf_metric = (
        instantaneous_rate
        / safe_history
    )

    return pf_metric

def select_pf_candidates(
    pf_metric: torch.Tensor,
    config: PFTimeDomainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select the highest-PF candidate UEs independently in each cell.

    Input shape:

        [batch, cell, UE]

    Output:

        candidate_indices:
            [batch, cell, num_candidates]

        candidate_metrics:
            [batch, cell, num_candidates]
    """

    if pf_metric.ndim != 3:
        raise ValueError(
            "Expected PF metric shape [batch, cell, UE], "
            f"got {tuple(pf_metric.shape)}."
        )

    num_ues = pf_metric.shape[-1]

    if config.num_candidates > num_ues:
        raise ValueError(
            f"Requested {config.num_candidates} candidates, "
            f"but only {num_ues} UEs are available per cell."
        )
    candidate_metrics, candidate_indices = torch.topk(
        pf_metric,
        k=config.num_candidates,
        dim=-1,
        largest=True,
        sorted=True,
    )

    return candidate_indices, candidate_metrics


def run_pf_tds(
    instantaneous_rate: torch.Tensor,
    past_average_throughput: torch.Tensor,
    config: PFTimeDomainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute PF metrics and return the shortlisted candidate UEs.
    """

    pf_metric = compute_pf_metric(
        instantaneous_rate=instantaneous_rate,
        past_average_throughput=past_average_throughput,
        config=config,
    )

    return select_pf_candidates(
        pf_metric=pf_metric,
        config=config,
    )

def update_past_average_throughput(
    previous_average: torch.Tensor,
    delivered_rate: torch.Tensor,
    forgetting_factor: float = 0.95,
) -> torch.Tensor:
    """
    Update the exponentially smoothed past-average throughput.

    Paper equation:

        R_new
        =
        (1 - epsilon) * delivered_rate
        +
        epsilon * R_old

    The numerical value of epsilon is not publicly specified by
    the paper, so callers must provide it explicitly.
    """

    if not 0.0 <= forgetting_factor < 1.0:
        raise ValueError(
            "forgetting_factor must satisfy 0 <= epsilon < 1."
        )

    if previous_average.shape != delivered_rate.shape:
        raise ValueError(
            "previous_average and delivered_rate "
            "must have identical shapes."
        )

    return (
        (1.0 - forgetting_factor) * delivered_rate
        +
        forgetting_factor * previous_average
    )

