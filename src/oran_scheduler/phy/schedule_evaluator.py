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

        num_selected_ues = int(
            selected_global_ues.numel()
        )

        num_scheduled_ues_per_rbg[
            rbg_index
        ] = num_selected_ues

        selected_ranks = (
            recommended_rank[
                batch_index,
                selected_global_ues,
            ]
        )

        selected_rx_combiners = (
            rx_combiners[
                batch_index,
                selected_global_ues,
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

        selected_all_bs_channel = (
            h_freq[
                batch_index,
                selected_global_ues,
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

