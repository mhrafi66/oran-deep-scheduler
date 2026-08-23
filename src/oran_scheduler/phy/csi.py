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