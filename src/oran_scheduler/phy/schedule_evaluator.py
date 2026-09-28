from dataclasses import dataclass

import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.mu_mimo_rate import (
    compute_mu_mimo_rbg_rates,
)
from oran_scheduler.phy.mu_mimo_rbg import (
    compute_isotropic_inter_cell_covariance,
    evaluate_selected_users_on_rbg,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.schedulers.allocation import (
    CellAllocation,
    selected_candidates_for_rbg,
    validate_cell_allocation,
)


@dataclass
class CellAllocationEvaluationData:
    """
    Physical evaluation of one cell's complete RBG allocation.

    candidate_nominal_rate_bps:
        [candidate, RBG]

    candidate_expected_goodput_bps:
        [candidate, RBG]

    candidate_target_compliant_rate_bps:
        [candidate, RBG]

    total_rbg_target_compliant_rate_bps:
        [RBG]

    total_cell_target_compliant_rate_bps:
        scalar

    num_scheduled_ues_per_rbg:
        [RBG]

    num_physical_layers_per_rbg:
        [RBG]

    rzf_alpha_by_rbg:
        [RBG]

        Zero for an unused RBG.
    """

    candidate_nominal_rate_bps: torch.Tensor

    candidate_expected_goodput_bps: torch.Tensor

    candidate_target_compliant_rate_bps: torch.Tensor

    total_rbg_target_compliant_rate_bps: torch.Tensor

    total_cell_target_compliant_rate_bps: torch.Tensor

    num_scheduled_ues_per_rbg: torch.Tensor

    num_physical_layers_per_rbg: torch.Tensor

    rzf_alpha_by_rbg: torch.Tensor


@dataclass
class RBGAllocationEvaluationData:
    """
    Physical evaluation of one selected candidate set
    on one RBG.

    selected_candidate_indices:
        [scheduled_UE]

    selected_global_ue_indices:
        [scheduled_UE]

    selected_ranks:
        [scheduled_UE]

    nominal_rate_bps:
        [scheduled_UE]

    expected_goodput_bps:
        [scheduled_UE]

    target_compliant_rate_bps:
        [scheduled_UE]

    total_target_compliant_rate_bps:
        scalar

    num_scheduled_ues:
        Number of spatially co-scheduled UEs.

    num_physical_layers:
        Sum of the selected UE ranks.

    rzf_alpha:
        RZF regularization value used for this RBG.
    """

    selected_candidate_indices: torch.Tensor
    selected_global_ue_indices: torch.Tensor
    selected_ranks: torch.Tensor

    nominal_rate_bps: torch.Tensor
    expected_goodput_bps: torch.Tensor
    target_compliant_rate_bps: torch.Tensor

    total_target_compliant_rate_bps: torch.Tensor

    num_scheduled_ues: int
    num_physical_layers: int

    rzf_alpha: float


def _resolve_candidate_physical_ue_indices(
    *,
    candidate_global_ue_indices: torch.Tensor,
    candidate_physical_ue_indices: (
        torch.Tensor | None
    ),
    h_freq: torch.Tensor,
) -> torch.Tensor:
    """
    Resolve scheduler-global UE identities to the
    UE indices used by the currently stored PHY tensor.

    Normal/full-channel mode:
        candidate_physical_ue_indices is None

        global UE ID == h_freq UE-axis index

    Memory-safe/chunked mode:
        candidate_global_ue_indices
            preserve persistent simulator identity

        candidate_physical_ue_indices
            identify where those same UEs live in the
            local/chunked PHY tensors

    Example:

        global identities:
            [103, 151, 209]

        local PHY indices:
            [0, 1, 2]

    This mapping is an OPEN-REPRODUCTION ENGINEERING
    mechanism. It does not alter scheduler semantics.
    """

    if candidate_global_ue_indices.ndim != 1:
        raise ValueError(
            "candidate_global_ue_indices must have "
            "shape [candidate]."
        )

    if candidate_physical_ue_indices is None:
        resolved = (
            candidate_global_ue_indices
        )

    else:
        if (
            candidate_physical_ue_indices.ndim
            != 1
        ):
            raise ValueError(
                "candidate_physical_ue_indices must "
                "have shape [candidate]."
            )

        if tuple(
            candidate_physical_ue_indices.shape
        ) != tuple(
            candidate_global_ue_indices.shape
        ):
            raise ValueError(
                "Global and physical candidate UE "
                "indices must have the same shape."
            )

        if torch.is_floating_point(
            candidate_physical_ue_indices
        ):
            raise ValueError(
                "candidate_physical_ue_indices must "
                "use an integer dtype."
            )

        if (
            candidate_physical_ue_indices.dtype
            == torch.bool
        ):
            raise ValueError(
                "candidate_physical_ue_indices cannot "
                "use torch.bool."
            )

        resolved = (
            candidate_physical_ue_indices
        )

    if resolved.device != h_freq.device:
        raise ValueError(
            "Physical UE indices and h_freq must be "
            "on the same device."
        )

    return resolved.to(
        dtype=torch.long,
    )


def _validate_selected_physical_ues(
    *,
    selected_physical_ues: torch.Tensor,
    h_freq: torch.Tensor,
    recommended_rank: torch.Tensor,
    rx_combiners: torch.Tensor,
) -> None:
    """
    Validate UE indices used to access the locally
    stored PHY tensors.

    Invalid padded candidate slots may contain -1,
    but an actually selected UE may never do so.
    """

    if selected_physical_ues.numel() == 0:
        return

    if torch.any(
        selected_physical_ues < 0
    ):
        raise ValueError(
            "A selected candidate cannot have a "
            "negative physical UE index."
        )

    num_phy_ues = int(
        h_freq.shape[1]
    )

    if int(
        recommended_rank.shape[1]
    ) != num_phy_ues:
        raise ValueError(
            "recommended_rank and h_freq must use "
            "the same physical UE dimension."
        )

    if int(
        rx_combiners.shape[1]
    ) != num_phy_ues:
        raise ValueError(
            "rx_combiners and h_freq must use the "
            "same physical UE dimension."
        )

    if torch.any(
        selected_physical_ues >= num_phy_ues
    ):
        raise ValueError(
            "A selected physical UE index is outside "
            "the stored PHY UE dimension."
        )

def _select_precomputed_candidate_phy(
    *,
    selected_candidate_indices: torch.Tensor,
    num_candidates: int,
    h_freq: torch.Tensor,
    first_subcarrier: int,
    last_subcarrier: int,
    precomputed_candidate_serving_channel: (
        torch.Tensor | None
    ),
    precomputed_candidate_inter_cell_covariance: (
        torch.Tensor | None
    ),
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
    ]
    | None
):
    """
    Select precomputed PHY tensors for one hypothetical
    candidate set.

    OPEN-REPRODUCTION ENGINEERING OPTIMIZATION.

    This does not approximate or alter the PHY.
    It only reuses quantities invariant to the
    hypothetical same-cell candidate combination.
    """

    has_serving = (
        precomputed_candidate_serving_channel
        is not None
    )

    has_covariance = (
        precomputed_candidate_inter_cell_covariance
        is not None
    )

    if has_serving != has_covariance:
        raise ValueError(
            "Precomputed serving channel and "
            "inter-cell covariance must either both "
            "be provided or both be None."
        )

    if not has_serving:
        return None

    assert (
        precomputed_candidate_serving_channel
        is not None
    )

    assert (
        precomputed_candidate_inter_cell_covariance
        is not None
    )

    device = h_freq.device

    if (
        precomputed_candidate_serving_channel.device
        != device
    ):
        raise ValueError(
            "Precomputed serving channel is on the "
            "wrong device."
        )

    if (
        precomputed_candidate_inter_cell_covariance.device
        != device
    ):
        raise ValueError(
            "Precomputed interference covariance is "
            "on the wrong device."
        )

    num_symbols = int(
        h_freq.shape[5]
    )

    num_subcarriers = int(
        h_freq.shape[6]
    )

    num_rx = int(
        h_freq.shape[2]
    )

    num_tx = int(
        h_freq.shape[4]
    )

    expected_serving_shape = (
        num_candidates,
        num_symbols,
        num_subcarriers,
        num_rx,
        num_tx,
    )

    expected_covariance_shape = (
        num_candidates,
        num_symbols,
        num_subcarriers,
        num_rx,
        num_rx,
    )

    if tuple(
        precomputed_candidate_serving_channel.shape
    ) != expected_serving_shape:
        raise ValueError(
            "Unexpected precomputed serving-channel "
            "shape."
        )

    if tuple(
        precomputed_candidate_inter_cell_covariance.shape
    ) != expected_covariance_shape:
        raise ValueError(
            "Unexpected precomputed interference-"
            "covariance shape."
        )

    selected_serving_channel = (
        precomputed_candidate_serving_channel[
            selected_candidate_indices,
            :,
            first_subcarrier:last_subcarrier,
            :,
            :,
        ]
    )

    selected_inter_cell_covariance = (
        precomputed_candidate_inter_cell_covariance[
            selected_candidate_indices,
            :,
            first_subcarrier:last_subcarrier,
            :,
            :,
        ]
    )

    return (
        selected_serving_channel,
        selected_inter_cell_covariance,
    )

def evaluate_rbg_candidate_set(
    selected_candidate_indices: torch.Tensor,
    candidate_global_ue_indices: torch.Tensor,
    h_freq: torch.Tensor,
    serving_cell_index: int,
    recommended_rank: torch.Tensor,
    rx_combiners: torch.Tensor,
    rbg_index: int,
    csi_subcarrier_index: int,
    subcarriers_per_rbg: int,
    tx_power_per_subcarrier_w: float | torch.Tensor,
    noise_power_per_subcarrier_w: float | torch.Tensor,
    link_adaptation_config: LinkAdaptationConfig,
    rate_config: RateConfig,
    batch_index: int = 0,
    candidate_physical_ue_indices: (
        torch.Tensor | None
    ) = None,

    precomputed_candidate_serving_channel: (
        torch.Tensor | None
    ) = None,

    precomputed_candidate_inter_cell_covariance: (
        torch.Tensor | None
    ) = None,
) -> RBGAllocationEvaluationData:
    """
    Evaluate one hypothetical candidate set on one RBG.

    selected_candidate_indices contains positions in the
    PF-TDS candidate list, NOT global UE IDs.

    This function is intentionally scheduler-agnostic.
    Baseline SDS, PF-Greedy SDS, or an RL policy can all
    call the same PHY evaluator.
    """

    if selected_candidate_indices.ndim != 1:
        raise ValueError(
            "selected_candidate_indices must have shape "
            "[scheduled_UE]."
        )

    if candidate_global_ue_indices.ndim != 1:
        raise ValueError(
            "candidate_global_ue_indices must have shape "
            "[candidate]."
        )

    if h_freq.ndim != 7:
        raise ValueError(
            "h_freq must have seven dimensions."
        )

    num_candidates = int(
        candidate_global_ue_indices.numel()
    )

    physical_ue_indices = (
        _resolve_candidate_physical_ue_indices(
            candidate_global_ue_indices=(
                candidate_global_ue_indices
            ),
            candidate_physical_ue_indices=(
                candidate_physical_ue_indices
            ),
            h_freq=h_freq,
        )
    )

    if torch.any(
        selected_candidate_indices < 0
    ):
        raise ValueError(
            "selected_candidate_indices cannot be negative."
        )

    if torch.any(
        selected_candidate_indices >= num_candidates
    ):
        raise ValueError(
            "selected_candidate_indices contains "
            "an invalid candidate."
        )

    if (
        torch.unique(
            selected_candidate_indices
        ).numel()
        != selected_candidate_indices.numel()
    ):
        raise ValueError(
            "The same candidate cannot appear twice "
            "on one RBG."
        )


    num_subcarriers = (
        h_freq.shape[-1]
    )

    if (
        num_subcarriers
        % subcarriers_per_rbg
        != 0
    ):
        raise ValueError(
            "Subcarrier count is not divisible by "
            "subcarriers_per_rbg."
        )

    num_rbgs = (
        num_subcarriers
        // subcarriers_per_rbg
    )

    if not (
        0 <= rbg_index < num_rbgs
    ):
        raise ValueError(
            "rbg_index is invalid."
        )

    if not (
        0 <= batch_index < h_freq.shape[0]
    ):
        raise ValueError(
            "batch_index is invalid."
        )


    real_dtype = (
        h_freq.real.dtype
    )

    device = (
        h_freq.device
    )

    if selected_candidate_indices.numel() == 0:

        empty_long = torch.empty(
            0,
            dtype=torch.long,
            device=device,
        )

        empty_real = torch.empty(
            0,
            dtype=real_dtype,
            device=device,
        )

        zero_rate = torch.zeros(
            (),
            dtype=real_dtype,
            device=device,
        )

        return RBGAllocationEvaluationData(
            selected_candidate_indices=(
                empty_long
            ),
            selected_global_ue_indices=(
                empty_long
            ),
            selected_ranks=(
                empty_long
            ),
            nominal_rate_bps=(
                empty_real
            ),
            expected_goodput_bps=(
                empty_real
            ),
            target_compliant_rate_bps=(
                empty_real
            ),
            total_target_compliant_rate_bps=(
                zero_rate
            ),
            num_scheduled_ues=0,
            num_physical_layers=0,
            rzf_alpha=0.0,
        )

    selected_global_ues = (
        candidate_global_ue_indices[
            selected_candidate_indices
        ]
    )

    selected_physical_ues = (
        physical_ue_indices[
            selected_candidate_indices
        ]
    )

    _validate_selected_physical_ues(
        selected_physical_ues=(
            selected_physical_ues
        ),
        h_freq=h_freq,
        recommended_rank=(
            recommended_rank
        ),
        rx_combiners=(
            rx_combiners
        ),
    )

    selected_ranks = (
        recommended_rank[
            batch_index,
            selected_physical_ues,
        ]
    )

    selected_rx_combiners = (
        rx_combiners[
            batch_index,
            selected_physical_ues,
            rbg_index,
            :,
            :,
        ]
    )

    first_subcarrier = (
        rbg_index
        * subcarriers_per_rbg
    )

    last_subcarrier = (
        first_subcarrier
        + subcarriers_per_rbg
    )

    precomputed_phy = (
        _select_precomputed_candidate_phy(
            selected_candidate_indices=(
                selected_candidate_indices
            ),
            num_candidates=num_candidates,
            h_freq=h_freq,
            first_subcarrier=(
                first_subcarrier
            ),
            last_subcarrier=(
                last_subcarrier
            ),
            precomputed_candidate_serving_channel=(
                precomputed_candidate_serving_channel
            ),
            precomputed_candidate_inter_cell_covariance=(
                precomputed_candidate_inter_cell_covariance
            ),
        )
    )

    if precomputed_phy is not None:

        (
            selected_serving_channel,
            inter_cell_covariance,
        ) = precomputed_phy

    else:

        selected_all_bs_channel = (
            h_freq[
                batch_index,
                selected_physical_ues,
                :,
                :,
                :,
                :,
                first_subcarrier:last_subcarrier,
            ]
        )

        selected_all_bs_channel = (
            selected_all_bs_channel
            .permute(
                0,
                4,
                5,
                2,
                1,
                3,
            )
            .contiguous()
        )

        inter_cell_covariance = (
            compute_isotropic_inter_cell_covariance(
                all_bs_channel=(
                    selected_all_bs_channel
                ),
                serving_cell_index=(
                    serving_cell_index
                ),
                tx_power_per_subcarrier_w=(
                    tx_power_per_subcarrier_w
                ),
            )
        )

        selected_serving_channel = (
            selected_all_bs_channel[
                :,
                :,
                :,
                serving_cell_index,
                :,
                :,
            ]
        )


    rzf_alpha = estimate_rzf_alpha(
        selected_ranks=selected_ranks,
        selected_rx_combiners=(
            selected_rx_combiners
        ),
        inter_cell_covariance=(
            inter_cell_covariance
        ),
        total_tx_power_w=(
            tx_power_per_subcarrier_w
        ),
        noise_power_w=(
            noise_power_per_subcarrier_w
        ),
    )

    rbg_phy_data = (
        evaluate_selected_users_on_rbg(
            selected_ue_channel=(
                selected_serving_channel
            ),
            selected_ranks=(
                selected_ranks
            ),
            selected_rx_combiners=(
                selected_rx_combiners
            ),
            total_tx_power_w=(
                tx_power_per_subcarrier_w
            ),
            noise_power_w=(
                noise_power_per_subcarrier_w
            ),
            rzf_alpha=rzf_alpha,
            csi_subcarrier_index=(
                csi_subcarrier_index
            ),
            inter_cell_covariance=(
                inter_cell_covariance
            ),
        )
    )

    rate_data = (
        compute_mu_mimo_rbg_rates(
            layer_sinr_linear=(
                rbg_phy_data
                .sinr_data
                .sinr_linear
            ),
            layer_ue_indices=(
                rbg_phy_data
                .layer_ue_indices
            ),
            layer_index_within_ue=(
                rbg_phy_data
                .layer_index_within_ue
            ),
            selected_ranks=(
                selected_ranks
            ),
            link_adaptation_config=(
                link_adaptation_config
            ),
            rate_config=(
                rate_config
            ),
        )
    )

    return RBGAllocationEvaluationData(
        selected_candidate_indices=(
            selected_candidate_indices
        ),
        selected_global_ue_indices=(
            selected_global_ues
        ),
        selected_ranks=(
            selected_ranks
        ),
        nominal_rate_bps=(
            rate_data.ue_rate_bps
        ),
        expected_goodput_bps=(
            rate_data
            .expected_ue_goodput_bps
        ),
        target_compliant_rate_bps=(
            rate_data
            .target_compliant_ue_rate_bps
        ),
        total_target_compliant_rate_bps=(
            rate_data
            .total_target_compliant_rate_bps
        ),
        num_scheduled_ues=int(
            selected_global_ues.numel()
        ),
        num_physical_layers=int(
            selected_ranks.sum().item()
        ),
        rzf_alpha=rzf_alpha,
    )







def estimate_rzf_alpha(
    selected_ranks: torch.Tensor,
    selected_rx_combiners: torch.Tensor,
    inter_cell_covariance: torch.Tensor,
    total_tx_power_w: float | torch.Tensor,
    noise_power_w: float | torch.Tensor,
) -> float:
    """
    Estimate one RZF regularization value for an RBG.

    Open-reproduction heuristic:

        alpha
            =
        L * mean(inter-cell interference + noise)
        / transmit power

    where L is the number of active physical streams.

    This is NOT specified by the Bell Labs paper.
    """

    if selected_ranks.ndim != 1:
        raise ValueError(
            "selected_ranks must have shape [selected_UE]."
        )

    if selected_rx_combiners.ndim != 3:
        raise ValueError(
            "selected_rx_combiners must have shape "
            "[selected_UE, 2, RX_ant]."
        )

    num_selected_ues = (
        selected_ranks.numel()
    )

    num_rx_ant = (
        selected_rx_combiners.shape[-1]
    )


    if inter_cell_covariance.ndim != 5:
        raise ValueError(
            "inter_cell_covariance must have shape "
            "[selected_UE, symbol, subcarrier, "
            "RX_ant, RX_ant]."
        )

    if inter_cell_covariance.shape[0] != (
        num_selected_ues
    ):
        raise ValueError(
            "Interference covariance UE dimension "
            "does not match selected_ranks."
        )

    if tuple(
        inter_cell_covariance.shape[-2:]
    ) != (
        num_rx_ant,
        num_rx_ant,
    ):
        raise ValueError(
            "Interference covariance RX dimensions "
            "do not match the combiners."
        )


    possible_layers = torch.arange(
        2,
        dtype=torch.long,
        device=selected_ranks.device,
    )

    valid_layer_mask = (
        possible_layers.unsqueeze(0)
        < selected_ranks.unsqueeze(1)
    )

    layer_pairs = torch.nonzero(
        valid_layer_mask,
        as_tuple=False,
    )

    layer_ue_indices = (
        layer_pairs[:, 0]
    )

    layer_index_within_ue = (
        layer_pairs[:, 1]
    )

    layer_combiners = (
        selected_rx_combiners[
            layer_ue_indices,
            layer_index_within_ue,
            :,
        ]
    )

    combiner_norm = torch.linalg.vector_norm(
        layer_combiners,
        dim=-1,
        keepdim=True,
    )

    if torch.any(
        combiner_norm <= 0
    ):
        raise ValueError(
            "A receive combiner has zero norm."
        )

    layer_combiners = (
        layer_combiners
        / combiner_norm
    )


    layer_covariance = (
        inter_cell_covariance[
            layer_ue_indices,
            :,
            :,
            :,
            :,
        ]
    )

    projected_interference = torch.einsum(
        "lr,lscrq,lq->lsc",
        layer_combiners.conj(),
        layer_covariance,
        layer_combiners,
    ).real

    projected_interference = torch.clamp(
        projected_interference,
        min=0.0,
    )

    mean_interference = (
        projected_interference.mean()
    )

    tx_power = torch.as_tensor(
        total_tx_power_w,
        dtype=projected_interference.dtype,
        device=projected_interference.device,
    )

    noise_power = torch.as_tensor(
        noise_power_w,
        dtype=projected_interference.dtype,
        device=projected_interference.device,
    )

    if torch.any(tx_power <= 0):
        raise ValueError(
            "total_tx_power_w must be positive."
        )

    if torch.any(noise_power < 0):
        raise ValueError(
            "noise_power_w cannot be negative."
        )

    num_physical_layers = int(
        selected_ranks.sum().item()
    )

    effective_disturbance = (
        mean_interference
        + noise_power
    )

    alpha_tensor = (
        float(num_physical_layers)
        * effective_disturbance
        / tx_power
    )

    return float(
        alpha_tensor.item()
    )

def evaluate_cell_allocation(
    allocation: CellAllocation,
    candidate_global_ue_indices: torch.Tensor,
    h_freq: torch.Tensor,
    serving_cell_index: int,
    recommended_rank: torch.Tensor,
    rx_combiners: torch.Tensor,
    csi_subcarrier_index: int,
    subcarriers_per_rbg: int,
    tx_power_per_subcarrier_w: float | torch.Tensor,
    noise_power_per_subcarrier_w: float | torch.Tensor,
    link_adaptation_config: LinkAdaptationConfig,
    rate_config: RateConfig,
    batch_index: int = 0,
    candidate_physical_ue_indices: (
        torch.Tensor | None
    ) = None,

    precomputed_candidate_serving_channel: (
        torch.Tensor | None
    ) = None,

    precomputed_candidate_inter_cell_covariance: (
        torch.Tensor | None
    ) = None,
) -> CellAllocationEvaluationData:
    """
    Physically evaluate every RBG of one cell allocation.

    candidate_global_ue_indices:
        [candidate]

    recommended_rank:
        [batch, global_UE]

    rx_combiners:
        [batch, global_UE, RBG, 2, RX_ant]

    h_freq:
        [
            batch,
            global_UE,
            RX_ant,
            BS,
            TX_ant,
            symbol,
            subcarrier,
        ]
    """

    if candidate_global_ue_indices.ndim != 1:
        raise ValueError(
            "candidate_global_ue_indices must have "
            "shape [candidate]."
        )

    if h_freq.ndim != 7:
        raise ValueError(
            "h_freq must have seven dimensions."
        )

    if recommended_rank.ndim != 2:
        raise ValueError(
            "recommended_rank must have shape "
            "[batch, global_UE]."
        )

    if rx_combiners.ndim != 5:
        raise ValueError(
            "rx_combiners must have shape "
            "[batch, global_UE, RBG, 2, RX_ant]."
        )

    if not (
        0
        <= batch_index
        < h_freq.shape[0]
    ):
        raise ValueError(
            "batch_index is invalid."
        )


    num_candidates = int(
        candidate_global_ue_indices.numel()
    )
    physical_ue_indices = (
        _resolve_candidate_physical_ue_indices(
            candidate_global_ue_indices=(
                candidate_global_ue_indices
            ),
            candidate_physical_ue_indices=(
                candidate_physical_ue_indices
            ),
            h_freq=h_freq,
        )
    )
    num_rbgs = (
        allocation.num_rbgs
    )

    validate_cell_allocation(
        allocation=allocation,
        num_candidates=num_candidates,
    )

    expected_num_subcarriers = (
        num_rbgs
        * subcarriers_per_rbg
    )

    if h_freq.shape[-1] != (
        expected_num_subcarriers
    ):
        raise ValueError(
            "h_freq subcarrier count does not match "
            "num_rbgs * subcarriers_per_rbg."
        )

    if rx_combiners.shape[2] != (
        num_rbgs
    ):
        raise ValueError(
            "rx_combiners RBG dimension does not "
            "match the allocation."
        )


    real_dtype = (
        h_freq.real.dtype
    )

    device = (
        h_freq.device
    )

    candidate_nominal_rate_bps = torch.zeros(
        (
            num_candidates,
            num_rbgs,
        ),
        dtype=real_dtype,
        device=device,
    )

    candidate_expected_goodput_bps = torch.zeros(
        (
            num_candidates,
            num_rbgs,
        ),
        dtype=real_dtype,
        device=device,
    )

    candidate_target_compliant_rate_bps = (
        torch.zeros(
            (
                num_candidates,
                num_rbgs,
            ),
            dtype=real_dtype,
            device=device,
        )
    )

    total_rbg_target_compliant_rate_bps = (
        torch.zeros(
            num_rbgs,
            dtype=real_dtype,
            device=device,
        )
    )

    num_scheduled_ues_per_rbg = torch.zeros(
        num_rbgs,
        dtype=torch.long,
        device=device,
    )

    num_physical_layers_per_rbg = torch.zeros(
        num_rbgs,
        dtype=torch.long,
        device=device,
    )

    rzf_alpha_by_rbg = torch.zeros(
        num_rbgs,
        dtype=real_dtype,
        device=device,
    )

    for rbg_index in range(
        num_rbgs
    ):

        selected_candidate_indices = (
            selected_candidates_for_rbg(
                allocation=allocation,
                rbg_index=rbg_index,
            )
        )

        if (
            selected_candidate_indices.numel()
            == 0
        ):
            continue

        selected_global_ues = (
            candidate_global_ue_indices[
                selected_candidate_indices
            ]
        )

        selected_physical_ues = (
            physical_ue_indices[
                selected_candidate_indices
            ]
        )

        _validate_selected_physical_ues(
            selected_physical_ues=(
                selected_physical_ues
            ),
            h_freq=h_freq,
            recommended_rank=(
                recommended_rank
            ),
            rx_combiners=(
                rx_combiners
            ),
        )

        num_selected_ues = int(
            selected_global_ues.numel()
        )

        num_scheduled_ues_per_rbg[
            rbg_index
        ] = num_selected_ues

        selected_ranks = (
            recommended_rank[
                batch_index,
                selected_physical_ues,
            ]
        )

        selected_rx_combiners = (
            rx_combiners[
                batch_index,
                selected_physical_ues,
                rbg_index,
                :,
                :,
            ]
        )

        num_physical_layers = int(
            selected_ranks.sum().item()
        )

        num_physical_layers_per_rbg[
            rbg_index
        ] = num_physical_layers

        first_subcarrier = (
            rbg_index
            * subcarriers_per_rbg
        )

        last_subcarrier = (
            first_subcarrier
            + subcarriers_per_rbg
        )

        precomputed_phy = (
            _select_precomputed_candidate_phy(
                selected_candidate_indices=(
                    selected_candidate_indices
                ),
                num_candidates=num_candidates,
                h_freq=h_freq,
                first_subcarrier=(
                    first_subcarrier
                ),
                last_subcarrier=(
                    last_subcarrier
                ),
                precomputed_candidate_serving_channel=(
                    precomputed_candidate_serving_channel
                ),
                precomputed_candidate_inter_cell_covariance=(
                    precomputed_candidate_inter_cell_covariance
                ),
            )
        )

        if precomputed_phy is not None:

            (
                selected_serving_channel,
                inter_cell_covariance,
            ) = precomputed_phy

        else:

            selected_all_bs_channel = (
                h_freq[
                    batch_index,
                    selected_physical_ues,
                    :,
                    :,
                    :,
                    :,
                    first_subcarrier:last_subcarrier,
                ]
            )

            selected_all_bs_channel = (
                selected_all_bs_channel
                .permute(
                    0,
                    4,
                    5,
                    2,
                    1,
                    3,
                )
                .contiguous()
            )

            inter_cell_covariance = (
                compute_isotropic_inter_cell_covariance(
                    all_bs_channel=(
                        selected_all_bs_channel
                    ),
                    serving_cell_index=(
                        serving_cell_index
                    ),
                    tx_power_per_subcarrier_w=(
                        tx_power_per_subcarrier_w
                    ),
                )
            )

            selected_serving_channel = (
                selected_all_bs_channel[
                    :,
                    :,
                    :,
                    serving_cell_index,
                    :,
                    :,
                ]
            )


        rzf_alpha = estimate_rzf_alpha(
            selected_ranks=selected_ranks,
            selected_rx_combiners=(
                selected_rx_combiners
            ),
            inter_cell_covariance=(
                inter_cell_covariance
            ),
            total_tx_power_w=(
                tx_power_per_subcarrier_w
            ),
            noise_power_w=(
                noise_power_per_subcarrier_w
            ),
        )

        rzf_alpha_by_rbg[
            rbg_index
        ] = rzf_alpha


        rbg_phy_data = (
            evaluate_selected_users_on_rbg(
                selected_ue_channel=(
                    selected_serving_channel
                ),
                selected_ranks=(
                    selected_ranks
                ),
                selected_rx_combiners=(
                    selected_rx_combiners
                ),
                total_tx_power_w=(
                    tx_power_per_subcarrier_w
                ),
                noise_power_w=(
                    noise_power_per_subcarrier_w
                ),
                rzf_alpha=rzf_alpha,
                csi_subcarrier_index=(
                    csi_subcarrier_index
                ),
                inter_cell_covariance=(
                    inter_cell_covariance
                ),
            )
        )


        rate_data = (
            compute_mu_mimo_rbg_rates(
                layer_sinr_linear=(
                    rbg_phy_data
                    .sinr_data
                    .sinr_linear
                ),
                layer_ue_indices=(
                    rbg_phy_data
                    .layer_ue_indices
                ),
                layer_index_within_ue=(
                    rbg_phy_data
                    .layer_index_within_ue
                ),
                selected_ranks=(
                    selected_ranks
                ),
                link_adaptation_config=(
                    link_adaptation_config
                ),
                rate_config=(
                    rate_config
                ),
            )
        )


        candidate_nominal_rate_bps[
            selected_candidate_indices,
            rbg_index,
        ] = rate_data.ue_rate_bps

        candidate_expected_goodput_bps[
            selected_candidate_indices,
            rbg_index,
        ] = (
            rate_data
            .expected_ue_goodput_bps
        )

        candidate_target_compliant_rate_bps[
            selected_candidate_indices,
            rbg_index,
        ] = (
            rate_data
            .target_compliant_ue_rate_bps
        )


        total_rbg_target_compliant_rate_bps[
            rbg_index
        ] = (
            rate_data
            .total_target_compliant_rate_bps
        )

    total_cell_target_compliant_rate_bps = (
        total_rbg_target_compliant_rate_bps.sum()
    )

    return CellAllocationEvaluationData(
        candidate_nominal_rate_bps=(
            candidate_nominal_rate_bps
        ),
        candidate_expected_goodput_bps=(
            candidate_expected_goodput_bps
        ),
        candidate_target_compliant_rate_bps=(
            candidate_target_compliant_rate_bps
        ),
        total_rbg_target_compliant_rate_bps=(
            total_rbg_target_compliant_rate_bps
        ),
        total_cell_target_compliant_rate_bps=(
            total_cell_target_compliant_rate_bps
        ),
        num_scheduled_ues_per_rbg=(
            num_scheduled_ues_per_rbg
        ),
        num_physical_layers_per_rbg=(
            num_physical_layers_per_rbg
        ),
        rzf_alpha_by_rbg=(
            rzf_alpha_by_rbg
        ),
    )

