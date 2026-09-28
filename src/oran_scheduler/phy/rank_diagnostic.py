from dataclasses import dataclass

import torch

from oran_scheduler.phy.csi import (
    extract_serving_mimo_rbg_channel,
)


@dataclass
class IdealSVDRankData:
    """
    Ideal-SVD rank-1 versus rank-2 diagnostic.

    This is NOT the final RI/PMI implementation.

    spatial_mode_power:
        [batch, UE, RBG, symbol, subcarrier, mode]

    interference_power_per_mode:
        [batch, UE, RBG, symbol, subcarrier, mode]

    rank1_sinr:
        [batch, UE, RBG, symbol, subcarrier]

    rank2_sinr:
        [batch, UE, RBG, symbol, subcarrier, 2]

    rank1_rbg_spectral_efficiency:
        [batch, UE, RBG]

    rank2_rbg_spectral_efficiency:
        [batch, UE, RBG]

    diagnostic_best_rank:
        [batch, UE, RBG], values 1 or 2
    """

    #
    # Reusable PHY intermediates.
    #
    # h_serving_rbg:
    #
    #   [batch, UE, RBG, symbol, subcarrier, RX, TX]
    #
    # inter_cell_covariance:
    #
    #   [batch, UE, RBG, symbol, subcarrier, RX, RX]
    #
    # These quantities are already computed while
    # deriving RI. Retaining them avoids reconstructing
    # exactly the same PHY information later.
    #

    h_serving_rbg: torch.Tensor

    inter_cell_covariance: torch.Tensor

    spatial_mode_power: torch.Tensor
    interference_power_per_mode: torch.Tensor

    rank1_sinr: torch.Tensor
    rank2_sinr: torch.Tensor

    rank1_rbg_spectral_efficiency: torch.Tensor
    rank2_rbg_spectral_efficiency: torch.Tensor

    diagnostic_best_rank: torch.Tensor


def compute_ideal_svd_rank_diagnostic(
    h_freq: torch.Tensor,
    serving_bs: torch.Tensor,
    num_rbgs: int,
    tx_power_per_subcarrier_w: float | torch.Tensor,
    noise_power_per_subcarrier_w: float | torch.Tensor,
) -> IdealSVDRankData:
    """
    Compare ideal rank-1 and rank-2 transmission.

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

        tx_power_per_subcarrier_w:
            Total transmit power available to one BS on one
            subcarrier, summed over all TX antenna ports.

        noise_power_per_subcarrier_w:
            Thermal receiver noise power per RX branch on one
            subcarrier.

    Open-reproduction assumptions for this diagnostic:

        1. Serving transmission uses ideal singular-vector
           eigenmodes rather than a finite PMI codebook.

        2. Rank 1 places all available subcarrier power on
           spatial mode 1.

        3. Rank 2 splits total subcarrier power equally between
           modes 1 and 2.

        4. Other BSs are assumed active and transmit their total
           subcarrier power isotropically across their TX ports.

        5. This computes Shannon spectral efficiency only.
           It is not yet NR MCS/TBLER throughput and must not be
           treated as the paper's actual RI algorithm.
    """


    if h_freq.ndim != 7:
        raise ValueError(
            "h_freq must have shape "
            "[batch, UE, RX_ant, BS, TX_ant, "
            "OFDM_symbol, subcarrier]."
        )

    if not torch.is_complex(h_freq):
        raise ValueError(
            "h_freq must be complex-valued."
        )

    if serving_bs.shape != h_freq.shape[:2]:
        raise ValueError(
            "serving_bs must have shape [batch, UE]."
        )

    if serving_bs.device != h_freq.device:
        raise ValueError(
            "serving_bs and h_freq must be on the same device."
        )

    num_rx_ant = h_freq.shape[2]
    num_bs = h_freq.shape[3]
    num_tx_ant = h_freq.shape[4]
    num_subcarriers = h_freq.shape[-1]

    if num_rx_ant < 2:
        raise ValueError(
            "Rank-2 diagnostic requires at least "
            "two RX antenna ports."
        )

    if num_tx_ant < 2:
        raise ValueError(
            "Rank-2 diagnostic requires at least "
            "two TX antenna ports."
        )

    if num_subcarriers % num_rbgs != 0:
        raise ValueError(
            "Number of subcarriers must be divisible "
            "by num_rbgs."
        )

    if torch.any(serving_bs < 0):
        raise ValueError(
            "serving_bs contains a negative BS index."
        )

    if torch.any(serving_bs >= num_bs):
        raise ValueError(
            "serving_bs contains an invalid BS index."
        )

    real_dtype = h_freq.real.dtype
    device = h_freq.device

    tx_power = torch.as_tensor(
        tx_power_per_subcarrier_w,
        dtype=real_dtype,
        device=device,
    )

    noise_power = torch.as_tensor(
        noise_power_per_subcarrier_w,
        dtype=real_dtype,
        device=device,
    )

    if torch.any(tx_power <= 0):
        raise ValueError(
            "tx_power_per_subcarrier_w must be positive."
        )

    if torch.any(noise_power <= 0):
        raise ValueError(
            "noise_power_per_subcarrier_w must be positive."
        )


    serving_data = extract_serving_mimo_rbg_channel(
        h_freq=h_freq,
        serving_bs=serving_bs,
        num_rbgs=num_rbgs,
    )

    h_serving = serving_data.h_serving_rbg


    serving_gram = (
        h_serving
        @ h_serving.mH
    )

    eigenvalues, receive_modes = torch.linalg.eigh(
        serving_gram
    )

    eigenvalues = torch.flip(
        eigenvalues,
        dims=(-1,),
    )

    receive_modes = torch.flip(
        receive_modes,
        dims=(-1,),
    )

    spatial_mode_power = torch.clamp(
        eigenvalues.real,
        min=0.0,
    )

    batch_size = h_freq.shape[0]
    num_ues = h_freq.shape[1]
    num_symbols = h_freq.shape[5]

    subcarriers_per_rbg = (
        num_subcarriers // num_rbgs
    )

    h_all = h_freq.permute(
        0,
        1,
        5,
        6,
        3,
        2,
        4,
    )

    h_all = h_all.reshape(
        batch_size,
        num_ues,
        num_symbols,
        num_rbgs,
        subcarriers_per_rbg,
        num_bs,
        num_rx_ant,
        num_tx_ant,
    )

    h_all = h_all.permute(
        0,
        1,
        3,
        2,
        4,
        5,
        6,
        7,
    ).contiguous()


    all_link_covariance = torch.matmul(
        h_all,
        h_all.mH,
    )

    isotropic_power_per_tx_port = (
        tx_power
        / float(num_tx_ant)
    )

    all_link_covariance = (
        all_link_covariance
        * isotropic_power_per_tx_port
    )

    serving_one_hot = torch.nn.functional.one_hot(
        serving_bs.to(torch.long),
        num_classes=num_bs,
    ).to(torch.bool)

    serving_mask = serving_one_hot[
        :,
        :,
        None,
        None,
        None,
        :,
        None,
        None,
    ]

    inter_cell_covariance = torch.where(
        serving_mask,
        torch.zeros_like(
            all_link_covariance
        ),
        all_link_covariance,
    ).sum(
        dim=5
    )

    interference_power_per_mode = torch.einsum(
        "...ri,...rq,...qi->...i",
        receive_modes.conj(),
        inter_cell_covariance,
        receive_modes,
    ).real

    interference_power_per_mode = torch.clamp(
        interference_power_per_mode,
        min=0.0,
    )

    first_mode_power = (
        spatial_mode_power[..., 0]
    )

    first_mode_interference = (
        interference_power_per_mode[..., 0]
    )

    rank1_sinr = (
        tx_power
        * first_mode_power
        / (
            first_mode_interference
            + noise_power
        )
    )

    first_two_mode_power = (
        spatial_mode_power[..., :2]
    )

    first_two_mode_interference = (
        interference_power_per_mode[..., :2]
    )

    rank2_stream_power = (
        tx_power
        / 2.0
    )

    rank2_sinr = (
        rank2_stream_power
        * first_two_mode_power
        / (
            first_two_mode_interference
            + noise_power
        )
    )

    rank1_spectral_efficiency = torch.log2(
        1.0 + rank1_sinr
    )

    rank2_layer_spectral_efficiency = torch.log2(
        1.0 + rank2_sinr
    )

    rank2_sum_spectral_efficiency = (
        rank2_layer_spectral_efficiency.sum(
            dim=-1
        )
    )

    rank1_rbg_spectral_efficiency = (
        rank1_spectral_efficiency.mean(
            dim=(3, 4)
        )
    )

    rank2_rbg_spectral_efficiency = (
        rank2_sum_spectral_efficiency.mean(
            dim=(3, 4)
        )
    )

    diagnostic_best_rank = torch.where(
        rank2_rbg_spectral_efficiency
        > rank1_rbg_spectral_efficiency,
        torch.full_like(
            rank1_rbg_spectral_efficiency,
            fill_value=2,
            dtype=torch.long,
        ),
        torch.full_like(
            rank1_rbg_spectral_efficiency,
            fill_value=1,
            dtype=torch.long,
        ),
    )

    return IdealSVDRankData(
        h_serving_rbg=(
            h_serving
        ),

        inter_cell_covariance=(
            inter_cell_covariance
        ),

        spatial_mode_power=(
            spatial_mode_power
        ),
        interference_power_per_mode=(
            interference_power_per_mode
        ),
        rank1_sinr=rank1_sinr,
        rank2_sinr=rank2_sinr,
        rank1_rbg_spectral_efficiency=(
            rank1_rbg_spectral_efficiency
        ),
        rank2_rbg_spectral_efficiency=(
            rank2_rbg_spectral_efficiency
        ),
        diagnostic_best_rank=(
            diagnostic_best_rank
        ),
    )



