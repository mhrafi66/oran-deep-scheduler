from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PFTimeDomainConfig:
    """Configuration for proportional-fair time-domain scheduling."""

    # Bell Labs paper:
    # maximum TDS candidate-set size = 10.
    num_candidates: int = 10

    denominator_epsilon: float = 1e-8

@dataclass
class PFTimeDomainResult:
    """
    Result of masked proportional-fair candidate selection.

    candidate_indices:
        [batch, cell, num_candidates]

        Local indices into the padded per-cell UE pool.

        Invalid candidate slots contain the safe index 0.
        They must never be interpreted without candidate_valid_mask.

    candidate_metrics:
        [batch, cell, num_candidates]

        PF metric for valid candidates.
        Invalid candidate slots contain 0.

    candidate_valid_mask:
        [batch, cell, num_candidates]

        True only when the corresponding candidate slot contains
        a real UE.
    """

    candidate_indices: torch.Tensor

    candidate_metrics: torch.Tensor

    candidate_valid_mask: torch.Tensor

    def __iter__(self):
        """
        Preserve compatibility with older code that did:

            indices, metrics = run_pf_tds(...)

        New code should normally use the named attributes.
        """

        yield self.candidate_indices
        yield self.candidate_metrics


def compute_pf_metric(
    instantaneous_rate: torch.Tensor,
    past_average_throughput: torch.Tensor,
    config: PFTimeDomainConfig,
) -> torch.Tensor:
    """
    Compute the proportional-fair metric.

        M_u = instantaneous_rate_u
              / past_average_throughput_u

    Shape:

        [batch, cell, padded_UE]
    """

    if instantaneous_rate.shape != past_average_throughput.shape:
        raise ValueError(
            "instantaneous_rate and past_average_throughput "
            "must have identical shapes. "
            f"Got {tuple(instantaneous_rate.shape)} and "
            f"{tuple(past_average_throughput.shape)}."
        )

    if instantaneous_rate.ndim != 3:
        raise ValueError(
            "Expected rate shape [batch, cell, UE], "
            f"got {tuple(instantaneous_rate.shape)}."
        )

    if torch.any(instantaneous_rate < 0):
        raise ValueError(
            "instantaneous_rate cannot be negative."
        )

    if torch.any(past_average_throughput < 0):
        raise ValueError(
            "past_average_throughput cannot be negative."
        )

    safe_history = torch.clamp(
        past_average_throughput,
        min=config.denominator_epsilon,
    )

    return (
        instantaneous_rate
        / safe_history
    )

def select_pf_candidates(
    pf_metric: torch.Tensor,
    config: PFTimeDomainConfig,
    valid_ue_mask: torch.Tensor | None = None,
) -> PFTimeDomainResult:
    """
    Select up to num_candidates valid PF UEs per cell.

    The returned tensor always has num_candidates slots.

    Cells with fewer valid UEs use invalid padded candidate slots,
    indicated by candidate_valid_mask.
    """

    if pf_metric.ndim != 3:
        raise ValueError(
            "Expected PF metric shape [batch, cell, UE], "
            f"got {tuple(pf_metric.shape)}."
        )

    if config.num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive."
        )

    if valid_ue_mask is None:
        valid_ue_mask = torch.ones_like(
            pf_metric,
            dtype=torch.bool,
        )

    if valid_ue_mask.shape != pf_metric.shape:
        raise ValueError(
            "valid_ue_mask must have the same shape "
            "as pf_metric."
        )

    valid_ue_mask = valid_ue_mask.to(
        dtype=torch.bool,
    )

    valid_metrics = pf_metric[
        valid_ue_mask
    ]

    if (
        valid_metrics.numel() > 0
        and not torch.isfinite(valid_metrics).all()
    ):
        raise ValueError(
            "Valid UE slots contain non-finite PF metrics."
        )
    negative_infinity = torch.tensor(
        float("-inf"),
        dtype=pf_metric.dtype,
        device=pf_metric.device,
    )

    masked_metric = torch.where(
        valid_ue_mask,
        pf_metric,
        negative_infinity,
    )

    num_ue_slots = masked_metric.shape[-1]

    if num_ue_slots < config.num_candidates:

        num_padding_slots = (
            config.num_candidates
            - num_ue_slots
        )

        metric_padding = torch.full(
            (
                masked_metric.shape[0],
                masked_metric.shape[1],
                num_padding_slots,
            ),
            fill_value=float("-inf"),
            dtype=masked_metric.dtype,
            device=masked_metric.device,
        )

        mask_padding = torch.zeros(
            (
                valid_ue_mask.shape[0],
                valid_ue_mask.shape[1],
                num_padding_slots,
            ),
            dtype=torch.bool,
            device=valid_ue_mask.device,
        )

        masked_metric = torch.cat(
            [
                masked_metric,
                metric_padding,
            ],
            dim=-1,
        )

        topk_source_mask = torch.cat(
            [
                valid_ue_mask,
                mask_padding,
            ],
            dim=-1,
        )

    else:
        topk_source_mask = valid_ue_mask

    candidate_metrics, candidate_indices = torch.topk(
        masked_metric,
        k=config.num_candidates,
        dim=-1,
        largest=True,
        sorted=True,
    )

    candidate_valid_mask = torch.gather(
        topk_source_mask,
        dim=-1,
        index=candidate_indices,
    )

    candidate_indices = torch.where(
        candidate_valid_mask,
        candidate_indices,
        torch.zeros_like(candidate_indices),
    )

    candidate_metrics = torch.where(
        candidate_valid_mask,
        candidate_metrics,
        torch.zeros_like(candidate_metrics),
    )

    return PFTimeDomainResult(
        candidate_indices=candidate_indices,
        candidate_metrics=candidate_metrics,
        candidate_valid_mask=candidate_valid_mask,
    )

   
def run_pf_tds(
    instantaneous_rate: torch.Tensor,
    past_average_throughput: torch.Tensor,
    config: PFTimeDomainConfig,
    valid_ue_mask: torch.Tensor | None = None,
) -> PFTimeDomainResult:
    """
    Compute PF metrics and shortlist up to K valid UEs per cell.
    """

    pf_metric = compute_pf_metric(
        instantaneous_rate=instantaneous_rate,
        past_average_throughput=past_average_throughput,
        config=config,
    )

    return select_pf_candidates(
        pf_metric=pf_metric,
        config=config,
        valid_ue_mask=valid_ue_mask,
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
    the paper.
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
