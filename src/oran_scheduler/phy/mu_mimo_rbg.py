from dataclasses import dataclass

import torch

from oran_scheduler.phy.mu_mimo import (
    MUMIMOLayerSINRData,
    compute_rzf_matrix,
    evaluate_mu_mimo_layer_sinr_with_precoder,
)

@dataclass
class MUMIMORBGData:
    """
    MU-MIMO result for one cell and one RBG.

    layer_ue_indices:
        [layer]

        Index of the selected UE owning each layer.

    layer_index_within_ue:
        [layer]

        0 for the UE's first layer,
        1 for its second layer.

    precoding_matrix:
        [TX_ant, layer]

    sinr_data:
        Post-MRC SINR data across every actual
        symbol/subcarrier in the RBG.
    """

    layer_ue_indices: torch.Tensor
    layer_index_within_ue: torch.Tensor

    precoding_matrix: torch.Tensor

    sinr_data: MUMIMOLayerSINRData

def compute_isotropic_inter_cell_covariance(
    all_bs_channel: torch.Tensor,
    serving_cell_index: int,
    tx_power_per_subcarrier_w: float | torch.Tensor,
) -> torch.Tensor:
    """
    Build inter-cell interference covariance.

    Args:
        all_bs_channel:
            [
                selected_UE,
                OFDM_symbol,
                subcarrier,
                BS,
                RX_ant,
                TX_ant,
            ]

        serving_cell_index:
            Serving BS shared by these selected UEs.

        tx_power_per_subcarrier_w:
            Total transmit power of each interfering BS
            on one subcarrier.

    Returns:
        [
            selected_UE,
            OFDM_symbol,
            subcarrier,
            RX_ant,
            RX_ant,
        ]

    Open-reproduction assumption:
        Every non-serving BS is active and transmits its
        total subcarrier power isotropically over its
        antenna ports.
    """

    if all_bs_channel.ndim != 6:
        raise ValueError(
            "all_bs_channel must have shape "
            "[UE, symbol, subcarrier, BS, RX_ant, TX_ant]."
        )

    num_bs = all_bs_channel.shape[3]
    num_tx_ant = all_bs_channel.shape[-1]

    if not 0 <= serving_cell_index < num_bs:
        raise ValueError(
            "serving_cell_index is invalid."
        )

    tx_power = torch.as_tensor(
        tx_power_per_subcarrier_w,
        dtype=all_bs_channel.real.dtype,
        device=all_bs_channel.device,
    )

    if torch.any(tx_power <= 0):
        raise ValueError(
            "tx_power_per_subcarrier_w must be positive."
        )

    power_per_tx_port = (
        tx_power
        / float(num_tx_ant)
    )

    covariance_by_bs = torch.matmul(
        all_bs_channel,
        all_bs_channel.mH,
    )

    covariance_by_bs = (
        covariance_by_bs
        * power_per_tx_port
    )

    interfering_bs_mask = torch.ones(
        num_bs,
        dtype=torch.bool,
        device=all_bs_channel.device,
    )

    interfering_bs_mask[
        serving_cell_index
    ] = False

    inter_cell_covariance = (
        covariance_by_bs[
            ...,
            interfering_bs_mask,
            :,
            :,
        ]
        .sum(
            dim=-3
        )
    )

    return inter_cell_covariance


def evaluate_selected_users_on_rbg(
    selected_ue_channel: torch.Tensor,
    selected_ranks: torch.Tensor,
    selected_rx_combiners: torch.Tensor,
    total_tx_power_w: float | torch.Tensor,
    noise_power_w: float | torch.Tensor,
    rzf_alpha: float,
    csi_subcarrier_index: int,
    inter_cell_covariance: torch.Tensor | None = None,
    precision: str = "single",
) -> MUMIMORBGData:
    """
    Evaluate one selected MU-MIMO user set on one RBG.

    Args:
        selected_ue_channel:
            [
                selected_UE,
                OFDM_symbol,
                subcarrier,
                RX_ant,
                TX_ant,
            ]

        selected_ranks:
            [selected_UE]

            Values must be 1 or 2.

        selected_rx_combiners:
            [selected_UE, 2, RX_ant]

        inter_cell_covariance:
            Optional:
            [
                selected_UE,
                OFDM_symbol,
                subcarrier,
                RX_ant,
                RX_ant,
            ]
    """

    if selected_ue_channel.ndim != 5:
        raise ValueError(
            "selected_ue_channel must have shape "
            "[UE, symbol, subcarrier, RX_ant, TX_ant]."
        )

    num_selected_ues = (
        selected_ue_channel.shape[0]
    )

    num_subcarriers = (
        selected_ue_channel.shape[2]
    )

    num_rx_ant = (
        selected_ue_channel.shape[3]
    )

    if tuple(
        selected_ranks.shape
    ) != (
        num_selected_ues,
    ):
        raise ValueError(
            "selected_ranks must have shape [selected_UE]."
        )

    if tuple(
        selected_rx_combiners.shape
    ) != (
        num_selected_ues,
        2,
        num_rx_ant,
    ):
        raise ValueError(
            "selected_rx_combiners must have shape "
            "[selected_UE, 2, RX_ant]."
        )

    if torch.any(
        (selected_ranks < 1)
        | (selected_ranks > 2)
    ):
        raise ValueError(
            "Selected UE ranks must be either 1 or 2."
        )

    if not 0 <= csi_subcarrier_index < num_subcarriers:
        raise ValueError(
            "csi_subcarrier_index is invalid."
        )

    possible_layer_indices = torch.arange(
        2,
        device=selected_ue_channel.device,
    )

    layer_valid_mask = (
        possible_layer_indices.unsqueeze(0)
        < selected_ranks.unsqueeze(1)
    )

    layer_pairs = torch.nonzero(
        layer_valid_mask,
        as_tuple=False,
    )

    layer_ue_indices = (
        layer_pairs[:, 0]
    )

    layer_index_within_ue = (
        layer_pairs[:, 1]
    )

    layer_rx_combiner = (
        selected_rx_combiners[
            layer_ue_indices,
            layer_index_within_ue,
            :,
        ]
    )

    csi_channel = (
        selected_ue_channel[
            :,
            0,
            csi_subcarrier_index,
            :,
            :,
        ]
    )

    layer_csi_channel = (
        csi_channel[
            layer_ue_indices,
            :,
            :,
        ]
    )

    effective_csi_channel = torch.einsum(
        "lr,lrt->lt",
        layer_rx_combiner.conj(),
        layer_csi_channel,
    )

    precoding_matrix = compute_rzf_matrix(
        effective_channel=(
            effective_csi_channel
        ),
        alpha=rzf_alpha,
        precision=precision,
    )

    layer_physical_channel = (
        selected_ue_channel[
            layer_ue_indices,
            :,
            :,
            :,
            :,
        ]
        .permute(
            1,
            2,
            0,
            3,
            4,
        )
        .contiguous()
    )

    layer_inter_cell_covariance = None

    if inter_cell_covariance is not None:

        expected_shape = (
            num_selected_ues,
            selected_ue_channel.shape[1],
            selected_ue_channel.shape[2],
            num_rx_ant,
            num_rx_ant,
        )

        if tuple(
            inter_cell_covariance.shape
        ) != expected_shape:
            raise ValueError(
                "Unexpected inter-cell covariance shape."
            )

        layer_inter_cell_covariance = (
            inter_cell_covariance[
                layer_ue_indices,
                :,
                :,
                :,
                :,
            ]
            .permute(
                1,
                2,
                0,
                3,
                4,
            )
            .contiguous()
        )

    sinr_data = (
        evaluate_mu_mimo_layer_sinr_with_precoder(
            layer_physical_channel=(
                layer_physical_channel
            ),
            layer_rx_combiner=(
                layer_rx_combiner
            ),
            precoding_matrix=(
                precoding_matrix
            ),
            total_tx_power_w=(
                total_tx_power_w
            ),
            noise_power_w=(
                noise_power_w
            ),
            inter_cell_covariance=(
                layer_inter_cell_covariance
            ),
        )
    )

    return MUMIMORBGData(
        layer_ue_indices=layer_ue_indices,
        layer_index_within_ue=(
            layer_index_within_ue
        ),
        precoding_matrix=precoding_matrix,
        sinr_data=sinr_data,
    )