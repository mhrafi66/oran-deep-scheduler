from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class SUMIMOFDSConfig:
    """
    Configuration for first-layer SU-MIMO frequency-domain scheduling.

    Important:
        This PF-based first-layer FDS is an open-reproduction
        assumption. The public Bell Labs paper does not provide
        enough implementation detail to claim this is the exact
        proprietary Nokia FDS.
    """

    denominator_epsilon: float = 1e-8

@dataclass
class SUMIMOFDSResult:
    """
    Result of first-layer SU-MIMO frequency-domain scheduling.

    selected_candidate_indices:
        [batch, cell, RBG]

        Index into the PF-TDS candidate dimension.

    selected_pf_metric:
        [batch, cell, RBG]

        PF metric of the selected candidate.

    selected_valid_mask:
        [batch, cell, RBG]

        True when a real candidate was selected.

        If a cell has no valid candidate, this is False and the
        corresponding selected_candidate_indices entry is the safe
        placeholder 0.

    candidate_pf_metric:
        [batch, cell, candidate, RBG]

        Full PF metric tensor before final selection.
        Invalid candidates contain -inf.
    """

    selected_candidate_indices: torch.Tensor

    selected_pf_metric: torch.Tensor

    selected_valid_mask: torch.Tensor

    candidate_pf_metric: torch.Tensor


def compute_candidate_rbg_pf_metric(
    candidate_rate: torch.Tensor,
    candidate_past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    config: SUMIMOFDSConfig,
) -> torch.Tensor:
    """
    Compute proportional-fair metrics for every candidate/RBG pair.

    Inputs:

        candidate_rate:
            [batch, cell, candidate, RBG]

        candidate_past_average_throughput:
            [batch, cell, candidate]

        candidate_valid_mask:
            [batch, cell, candidate]

    Output:

        candidate_pf_metric:
            [batch, cell, candidate, RBG]

    PF metric:

        M[u, r]
            =
        candidate_rate[u, r]
        /
        past_average_throughput[u]
    """

    if candidate_rate.ndim != 4:
        raise ValueError(
            "Expected candidate_rate shape "
            "[batch, cell, candidate, RBG], "
            f"got {tuple(candidate_rate.shape)}."
        )

    if candidate_past_average_throughput.ndim != 3:
        raise ValueError(
            "Expected candidate_past_average_throughput shape "
            "[batch, cell, candidate], "
            f"got "
            f"{tuple(candidate_past_average_throughput.shape)}."
        )

    if candidate_valid_mask.ndim != 3:
        raise ValueError(
            "Expected candidate_valid_mask shape "
            "[batch, cell, candidate], "
            f"got {tuple(candidate_valid_mask.shape)}."
        )

    expected_candidate_shape = (
        candidate_rate.shape[0],
        candidate_rate.shape[1],
        candidate_rate.shape[2],
    )

    if (
        tuple(candidate_past_average_throughput.shape)
        != expected_candidate_shape
    ):
        raise ValueError(
            "Candidate history dimensions do not match "
            "candidate_rate."
        )

    if (
        tuple(candidate_valid_mask.shape)
        != expected_candidate_shape
    ):
        raise ValueError(
            "candidate_valid_mask dimensions do not match "
            "candidate_rate."
        )

    if candidate_valid_mask.device != candidate_rate.device:
        raise ValueError(
            "candidate_valid_mask and candidate_rate "
            "must be on the same device."
        )

    if (
        candidate_past_average_throughput.device
        != candidate_rate.device
    ):
        raise ValueError(
            "Candidate throughput history and candidate_rate "
            "must be on the same device."
        )

    if torch.any(candidate_rate < 0):
        raise ValueError(
            "Candidate achievable rates cannot be negative."
        )

    if torch.any(
        candidate_past_average_throughput < 0
    ):
        raise ValueError(
            "Past-average throughput cannot be negative."
        )


    safe_history = torch.clamp(
        candidate_past_average_throughput,
        min=config.denominator_epsilon,
    )

    # [B, C, K] -> [B, C, K, 1]
    safe_history = safe_history.unsqueeze(
        dim=-1,
    )

    # Broadcasting:
    #
    # candidate_rate: [B, C, K, RBG]
    # safe_history:   [B, C, K, 1]
    #
    # result:         [B, C, K, RBG]
    pf_metric = (
        candidate_rate
        / safe_history
    )

    expanded_valid_mask = (
        candidate_valid_mask
        .unsqueeze(-1)
        .expand_as(pf_metric)
    )

    negative_infinity = torch.tensor(
        float("-inf"),
        dtype=pf_metric.dtype,
        device=pf_metric.device,
    )

    pf_metric = torch.where(
        expanded_valid_mask,
        pf_metric,
        negative_infinity,
    )

    return pf_metric


def select_su_mimo_fds_candidates(
    candidate_pf_metric: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> SUMIMOFDSResult:
    """
    Select one valid PF candidate independently on every RBG.

    candidate_pf_metric:
        [batch, cell, candidate, RBG]

    candidate_valid_mask:
        [batch, cell, candidate]

    Returns one candidate slot per RBG:

        [batch, cell, RBG]
    """

    if candidate_pf_metric.ndim != 4:
        raise ValueError(
            "Expected candidate PF metric shape "
            "[batch, cell, candidate, RBG]."
        )

    if candidate_valid_mask.ndim != 3:
        raise ValueError(
            "Expected candidate_valid_mask shape "
            "[batch, cell, candidate]."
        )

    if (
        tuple(candidate_pf_metric.shape[:3])
        != tuple(candidate_valid_mask.shape)
    ):
        raise ValueError(
            "Candidate dimensions do not match."
        )

    selected_pf_metric, selected_candidate_indices = (
        torch.max(
            candidate_pf_metric,
            dim=2,
        )
    )

    cell_has_valid_candidate = (
        candidate_valid_mask.any(
            dim=2,
        )
    )

    selected_valid_mask = (
        cell_has_valid_candidate
        .unsqueeze(-1)
        .expand(
            -1,
            -1,
            candidate_pf_metric.shape[-1],
        )
    )

    selected_candidate_indices = torch.where(
        selected_valid_mask,
        selected_candidate_indices,
        torch.zeros_like(
            selected_candidate_indices
        ),
    )

    selected_pf_metric = torch.where(
        selected_valid_mask,
        selected_pf_metric,
        torch.zeros_like(
            selected_pf_metric
        ),
    )

    return SUMIMOFDSResult(
        selected_candidate_indices=(
            selected_candidate_indices
        ),
        selected_pf_metric=selected_pf_metric,
        selected_valid_mask=selected_valid_mask,
        candidate_pf_metric=candidate_pf_metric,
    )

def run_su_mimo_fds(
    candidate_rate: torch.Tensor,
    candidate_past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    config: SUMIMOFDSConfig,
) -> SUMIMOFDSResult:
    """
    Run first-layer PF frequency-domain scheduling.
    """

    candidate_pf_metric = (
        compute_candidate_rbg_pf_metric(
            candidate_rate=candidate_rate,
            candidate_past_average_throughput=(
                candidate_past_average_throughput
            ),
            candidate_valid_mask=candidate_valid_mask,
            config=config,
        )
    )

    return select_su_mimo_fds_candidates(
        candidate_pf_metric=candidate_pf_metric,
        candidate_valid_mask=candidate_valid_mask,
    )



