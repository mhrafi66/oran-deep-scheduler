from dataclasses import dataclass

import torch


@dataclass
class ServingMIMORBGChannelData:
    """
    Serving-link MIMO channel grouped by RBG.

    h_serving_rbg:
        [
            batch,
            UE,
            RBG,
            OFDM_symbol,
            subcarrier_in_RBG,
            RX_ant,
            TX_ant,
        ]
    """

    h_serving_rbg: torch.Tensor
    subcarriers_per_rbg: int


@dataclass
class SpatialModeData:
    """
    Spatial-mode information obtained from the serving MIMO channel.

    singular_values:
        [
            batch,
            UE,
            RBG,
            OFDM_symbol,
            subcarrier_in_RBG,
            spatial_mode,
        ]

    rbg_mode_power:
        [
            batch,
            UE,
            RBG,
            spatial_mode,
        ]

    second_to_first_power_ratio:
        [
            batch,
            UE,
            RBG,
        ]
    """

    singular_values: torch.Tensor
    rbg_mode_power: torch.Tensor
    second_to_first_power_ratio: torch.Tensor


@dataclass
class IdealSVDCSIData:
    """
    Ideal-SVD CSI surrogate for the open reproduction.

    IMPORTANT:
        This is not literal 3GPP RI/PMI feedback.

    csi_snapshot:
        [batch, UE, RBG, RX_ant, TX_ant]

    recommended_rank:
        [batch, UE]

        Wideband rank recommendation, limited to 1 or 2.

    singular_values:
        [batch, UE, RBG, layer]

        First two singular values.

    rx_combiners:
        [batch, UE, RBG, layer, RX_ant]

        Left-singular-vector receive directions.

    precoder_directions:
        [batch, UE, RBG, layer, TX_ant]

        Right-singular-vector transmit directions.

    effective_channels:
        [batch, UE, RBG, layer, TX_ant]

        Effective channel rows:

            f^H H

    layer_valid_mask:
        [batch, UE, RBG, layer]

        Indicates which layers are enabled by the recommended rank.
    """

    csi_snapshot: torch.Tensor

    recommended_rank: torch.Tensor

    singular_values: torch.Tensor

    rx_combiners: torch.Tensor

    precoder_directions: torch.Tensor

    effective_channels: torch.Tensor

    layer_valid_mask: torch.Tensor

    csi_subcarrier_index: int

def compute_ideal_svd_csi(
    h_serving_rbg: torch.Tensor,
    rank1_rbg_score: torch.Tensor,
    rank2_rbg_score: torch.Tensor,
) -> IdealSVDCSIData:
    """
    Build the ideal-SVD CSI surrogate used by the open reproduction.

    Args:
        h_serving_rbg:
            [
                batch,
                UE,
                RBG,
                OFDM_symbol,
                subcarrier_in_RBG,
                RX_ant,
                TX_ant,
            ]

        rank1_rbg_score:
            [batch, UE, RBG]

        rank2_rbg_score:
            [batch, UE, RBG]

    The scores are currently the ideal-SVD rank-1 and rank-2
    spectral efficiencies from rank_diagnostic.py.

    CSI approximation:

        - Rank is chosen wideband by comparing summed rank-1
          and rank-2 scores across RBGs.

        - One representative subcarrier is used for each RBG.

        - SVD provides ideal spatial directions.

        - At most two layers are retained.

    This is an explicitly documented reproduction substitution,
    not a literal implementation of 3GPP RI/PMI reporting.
    """

    if h_serving_rbg.ndim != 7:
        raise ValueError(
            "h_serving_rbg must have shape "
            "[batch, UE, RBG, symbol, subcarrier, "
            "RX_ant, TX_ant]."
        )

    expected_score_shape = (
        h_serving_rbg.shape[0],
        h_serving_rbg.shape[1],
        h_serving_rbg.shape[2],
    )

    if tuple(rank1_rbg_score.shape) != expected_score_shape:
        raise ValueError(
            "rank1_rbg_score must have shape "
            "[batch, UE, RBG]."
        )

    if tuple(rank2_rbg_score.shape) != expected_score_shape:
        raise ValueError(
            "rank2_rbg_score must have shape "
            "[batch, UE, RBG]."
        )

    if rank1_rbg_score.device != h_serving_rbg.device:
        raise ValueError(
            "rank1_rbg_score and h_serving_rbg "
            "must be on the same device."
        )

    if rank2_rbg_score.device != h_serving_rbg.device:
        raise ValueError(
            "rank2_rbg_score and h_serving_rbg "
            "must be on the same device."
        )

    num_ofdm_symbols = (
        h_serving_rbg.shape[3]
    )

    subcarriers_per_rbg = (
        h_serving_rbg.shape[4]
    )

    if num_ofdm_symbols <= 0:
        raise ValueError(
            "At least one OFDM symbol is required."
        )

    if subcarriers_per_rbg <= 0:
        raise ValueError(
            "At least one subcarrier per RBG is required."
        )

    csi_subcarrier_index = (
        subcarriers_per_rbg // 2
    )

    csi_snapshot = h_serving_rbg[
        :,
        :,
        :,
        0,
        csi_subcarrier_index,
        :,
        :,
    ]

    u, singular_values_all, vh = torch.linalg.svd(
        csi_snapshot,
        full_matrices=False,
    )

    singular_values = (
        singular_values_all[..., :2]
    )

    rx_combiners = (
        u[
            ...,
            :,
            :2,
        ]
        .movedim(
            -1,
            -2,
        )
        .contiguous()
    )

    precoder_directions = (
        vh[..., :2, :]
        .conj()
        .contiguous()
    )

    effective_channels = torch.einsum(
        "...lr,...rt->...lt",
        rx_combiners.conj(),
        csi_snapshot,
    )

    rank1_wideband_score = (
        rank1_rbg_score.sum(
            dim=-1
        )
    )

    rank2_wideband_score = (
        rank2_rbg_score.sum(
            dim=-1
        )
    )

    recommended_rank = torch.where(
        rank2_wideband_score
        > rank1_wideband_score,
        torch.full_like(
            rank1_wideband_score,
            fill_value=2,
            dtype=torch.long,
        ),
        torch.full_like(
            rank1_wideband_score,
            fill_value=1,
            dtype=torch.long,
        ),
    )

    layer_numbers = torch.arange(
        1,
        3,
        dtype=torch.long,
        device=h_serving_rbg.device,
    )

    wideband_layer_mask = (
        layer_numbers
        <= recommended_rank.unsqueeze(-1)
    )

    layer_valid_mask = (
        wideband_layer_mask
        .unsqueeze(2)
        .expand(
            -1,
            -1,
            h_serving_rbg.shape[2],
            -1,
        )
    )

    return IdealSVDCSIData(
        csi_snapshot=csi_snapshot,
        recommended_rank=recommended_rank,
        singular_values=singular_values,
        rx_combiners=rx_combiners,
        precoder_directions=(
            precoder_directions
        ),
        effective_channels=(
            effective_channels
        ),
        layer_valid_mask=(
            layer_valid_mask
        ),
        csi_subcarrier_index=(
            csi_subcarrier_index
        ),
    )


def extract_serving_mimo_rbg_channel(
    h_freq: torch.Tensor,
    serving_bs: torch.Tensor,
    num_rbgs: int,
) -> ServingMIMORBGChannelData:
    """
    Extract each UE's serving-BS MIMO channel and group
    frequency subcarriers into RBGs.

    Args:
        h_freq:
            [
                batch,
                UE,
                RX_ant,
                BS,
                TX_ant,
                OFDM_symbol,
                subcarrier,
            ]

        serving_bs:
            [batch, UE]

        num_rbgs:
            Number of RBGs.

    Returns:
        Serving-link channel:

            [
                batch,
                UE,
                RBG,
                OFDM_symbol,
                subcarrier_in_RBG,
                RX_ant,
                TX_ant,
            ]
    """

    if h_freq.ndim != 7:
        raise ValueError(
            "h_freq must have shape "
            "[batch, UE, RX_ant, BS, TX_ant, "
            "OFDM_symbol, subcarrier]."
        )

    if serving_bs.ndim != 2:
        raise ValueError(
            "serving_bs must have shape [batch, UE]."
        )

    if tuple(serving_bs.shape) != tuple(
        h_freq.shape[:2]
    ):
        raise ValueError(
            "serving_bs batch and UE dimensions "
            "must match h_freq."
        )

    if serving_bs.device != h_freq.device:
        raise ValueError(
            "serving_bs and h_freq must be "
            "on the same device."
        )

    if torch.is_floating_point(serving_bs):
        raise ValueError(
            "serving_bs must contain integer BS indices."
        )

    if serving_bs.dtype == torch.bool:
        raise ValueError(
            "serving_bs must contain integer BS indices."
        )

    if num_rbgs <= 0:
        raise ValueError(
            "num_rbgs must be positive."
        )

    num_subcarriers = h_freq.shape[-1]

    if num_subcarriers % num_rbgs != 0:
        raise ValueError(
            "Number of subcarriers must be divisible "
            "by num_rbgs."
        )

    subcarriers_per_rbg = (
        num_subcarriers // num_rbgs
    )

    num_bs = h_freq.shape[3]

    serving_bs_long = serving_bs.to(
        dtype=torch.long,
    )

    if torch.any(serving_bs_long < 0):
        raise ValueError(
            "serving_bs contains a negative BS index."
        )

    if torch.any(serving_bs_long >= num_bs):
        raise ValueError(
            "serving_bs contains an index outside "
            "the BS dimension of h_freq."
        )

    batch_size = h_freq.shape[0]
    num_ues = h_freq.shape[1]
    num_rx_ant = h_freq.shape[2]
    num_tx_ant = h_freq.shape[4]
    num_ofdm_symbols = h_freq.shape[5]

    gather_indices = (
        serving_bs_long
        .view(
            batch_size,
            num_ues,
            1,
            1,
            1,
            1,
            1,
        )
        .expand(
            batch_size,
            num_ues,
            num_rx_ant,
            1,
            num_tx_ant,
            num_ofdm_symbols,
            num_subcarriers,
        )
    )

    serving_channel = torch.gather(
        h_freq,
        dim=3,
        index=gather_indices,
    )

    serving_channel = (
        serving_channel.squeeze(3)
    )

    serving_channel = (
        serving_channel.permute(
            0,
            1,
            4,
            5,
            2,
            3,
        )
    )


    serving_channel = (
        serving_channel.reshape(
            batch_size,
            num_ues,
            num_ofdm_symbols,
            num_rbgs,
            subcarriers_per_rbg,
            num_rx_ant,
            num_tx_ant,
        )
    )

    h_serving_rbg = (
        serving_channel.permute(
            0,
            1,
            3,
            2,
            4,
            5,
            6,
        )
        .contiguous()
    )

    return ServingMIMORBGChannelData(
        h_serving_rbg=h_serving_rbg,
        subcarriers_per_rbg=(
            subcarriers_per_rbg
        ),
    )


def compute_rbg_spatial_modes(
    h_serving_rbg: torch.Tensor,
) -> SpatialModeData:
    """
    Compute singular-value spatial-mode information.

    Input:
        [
            batch,
            UE,
            RBG,
            OFDM_symbol,
            subcarrier_in_RBG,
            RX_ant,
            TX_ant,
        ]

    This is a diagnostic physical-channel decomposition.

    It is NOT a PMI implementation and it does NOT choose RI.
    """

    if h_serving_rbg.ndim != 7:
        raise ValueError(
            "h_serving_rbg must have shape "
            "[batch, UE, RBG, symbol, subcarrier, "
            "RX_ant, TX_ant]."
        )

    if not torch.is_complex(
        h_serving_rbg
    ):
        raise ValueError(
            "MIMO channel must be complex-valued."
        )

    if not torch.isfinite(
        h_serving_rbg.real
    ).all():
        raise ValueError(
            "Channel contains non-finite real values."
        )

    if not torch.isfinite(
        h_serving_rbg.imag
    ).all():
        raise ValueError(
            "Channel contains non-finite imaginary values."
        )

    singular_values = torch.linalg.svdvals(
        h_serving_rbg
    )
    mode_power = (
        singular_values ** 2
    )

    rbg_mode_power = mode_power.mean(
        dim=(3, 4)
    )

    num_modes = (
        rbg_mode_power.shape[-1]
    )

    first_mode_power = (
        rbg_mode_power[..., 0]
    )

    if num_modes >= 2:
        second_mode_power = (
            rbg_mode_power[..., 1]
        )

        second_to_first_power_ratio = (
            torch.where(
                first_mode_power > 0,
                second_mode_power
                / first_mode_power,
                torch.zeros_like(
                    first_mode_power
                ),
            )
        )

    else:
        second_to_first_power_ratio = (
            torch.zeros_like(
                first_mode_power
            )
        )

    return SpatialModeData(
        singular_values=singular_values,
        rbg_mode_power=rbg_mode_power,
        second_to_first_power_ratio=(
            second_to_first_power_ratio
        ),
    )