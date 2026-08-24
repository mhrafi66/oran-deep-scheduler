from dataclasses import dataclass

import torch

from sionna.phy.constants import (
    BOLTZMANN_CONSTANT,
)
from sionna.phy.utils import (
    dbm_to_watt,
)

@dataclass(frozen=True)
class SINRConfig:
    """
    Configuration for the initial physical multicell SINR model.

    Paper-specified:
        gNB transmit power = 44 dBm
        subcarrier spacing = 30 kHz

    Open-reproduction assumptions for this initial sanity model:
        - 44 dBm is treated as total occupied-band transmit power
          per cell.
        - Power is distributed uniformly across occupied subcarriers.
        - All cells are active on all subcarriers unless an explicit
          activity mask is supplied.
        - Thermal-noise temperature is configurable.
        - receiver_noise_figure_db=None means thermal noise only.

    This is currently for the temporary 1x1 SISO channel validation.
    """

    tx_power_dbm: float = 44.0

    subcarrier_spacing_hz: float = 30e3

    temperature_k: float = 294.0

    receiver_noise_figure_db: float | None = None

    precision: str = "single"

    device: str = "cuda:0"

@dataclass
class SINRData:
    """
    Physical SISO multicell SINR quantities.

    all_link_received_power_w:
        [batch, UE, BS, OFDM_symbol, subcarrier]

    desired_power_w:
        [batch, UE, OFDM_symbol, subcarrier]

    interference_power_w:
        [batch, UE, OFDM_symbol, subcarrier]

    noise_power_w:
        Scalar tensor containing noise power per subcarrier.

    sinr_linear:
        [batch, UE, OFDM_symbol, subcarrier]
    """

    all_link_received_power_w: torch.Tensor

    desired_power_w: torch.Tensor

    interference_power_w: torch.Tensor

    noise_power_w: torch.Tensor

    sinr_linear: torch.Tensor


def compute_total_tx_power_w(
    config: SINRConfig,
) -> torch.Tensor:
    """
    Convert total per-cell transmit power from dBm to watts.

    Paper value:

        44 dBm
    """

    return dbm_to_watt(
        config.tx_power_dbm,
        precision=config.precision,
        device=config.device,
    )


def compute_subcarrier_tx_power_w(
    config: SINRConfig,
    num_subcarriers: int,
) -> torch.Tensor:
    """
    Compute transmit power per occupied subcarrier.

    Current open-reproduction assumption:

        total cell transmit power is distributed uniformly
        over all occupied subcarriers.
    """

    if num_subcarriers <= 0:
        raise ValueError(
            "num_subcarriers must be positive."
        )

    total_tx_power_w = compute_total_tx_power_w(
        config
    )

    return (
        total_tx_power_w
        / num_subcarriers
    )


def compute_noise_power_per_subcarrier_w(
    config: SINRConfig,
) -> torch.Tensor:
    """
    Compute receiver noise power over one subcarrier bandwidth.

    Base thermal-noise model:

        N = k_B * T * Delta_f

    If a receiver noise figure is explicitly supplied:

        N = k_B * T * Delta_f * F

    where:

        F = 10^(NF_dB / 10)
    """

    if config.temperature_k <= 0:
        raise ValueError(
            "temperature_k must be positive."
        )

    if config.subcarrier_spacing_hz <= 0:
        raise ValueError(
            "subcarrier_spacing_hz must be positive."
        )

    noise_power_w = (
        BOLTZMANN_CONSTANT
        * config.temperature_k
        * config.subcarrier_spacing_hz
    )

    noise_power_w = torch.tensor(
        noise_power_w,
        dtype=torch.float32
        if config.precision == "single"
        else torch.float64,
        device=config.device,
    )

    if config.receiver_noise_figure_db is not None:

        noise_factor = 10.0 ** (
            config.receiver_noise_figure_db
            / 10.0
        )

        noise_power_w = (
            noise_power_w
            * noise_factor
        )

    return noise_power_w


def validate_siso_channel_shape(
    h_freq: torch.Tensor,
) -> None:
    """
    Validate that the current channel is compatible with the
    temporary 1x1 SISO physical-SINR model.

    Expected Sionna channel layout:

        [batch,
         UE,
         RX antenna,
         BS,
         TX antenna,
         OFDM symbol,
         subcarrier]
    """

    if h_freq.ndim != 7:
        raise ValueError(
            "Expected h_freq shape "
            "[batch, UE, RX_ant, BS, TX_ant, "
            "OFDM_symbol, subcarrier], "
            f"got {tuple(h_freq.shape)}."
        )

    if h_freq.shape[2] != 1:
        raise ValueError(
            "Current SISO SINR model requires exactly "
            "one UE RX antenna."
        )

    if h_freq.shape[4] != 1:
        raise ValueError(
            "Current SISO SINR model requires exactly "
            "one BS TX antenna."
        )

def compute_all_link_received_power_w(
    h_freq: torch.Tensor,
    subcarrier_tx_power_w: torch.Tensor,
) -> torch.Tensor:
    """
    Compute received power from every BS at every UE/subcarrier.

    Current 1x1 relation:

        P_rx[u,b,f]
            =
        P_tx,f * |H[u,b,f]|^2

    Input:

        h_freq:
            [B, U, 1, BS, 1, S, F]

    Output:

        [B, U, BS, S, F]
    """

    validate_siso_channel_shape(
        h_freq
    )

    channel_power_gain = (
        torch.abs(h_freq) ** 2
    )

    channel_power_gain = (
        channel_power_gain
        .squeeze(dim=2)
        .squeeze(dim=3)
    )

    return (
        subcarrier_tx_power_w
        * channel_power_gain
    )



def compute_siso_multicell_sinr(
    h_freq: torch.Tensor,
    serving_bs: torch.Tensor,
    config: SINRConfig,
    bs_activity_mask: torch.Tensor | None = None,
) -> SINRData:
    """
    Compute physical multicell downlink SINR for the current 1x1 model.

    Args:
        h_freq:
            [B, U, 1, BS, 1, S, F]

        serving_bs:
            [B, U]

        bs_activity_mask:
            Optional boolean tensor:

                [B, BS, S, F]

            True means that BS transmits on that resource.

            If None, every BS is assumed active on every resource.
    """

    validate_siso_channel_shape(
        h_freq
    )

    batch_size = h_freq.shape[0]
    num_ues = h_freq.shape[1]
    num_bs = h_freq.shape[3]
    num_symbols = h_freq.shape[5]
    num_subcarriers = h_freq.shape[6]

    if tuple(serving_bs.shape) != (
        batch_size,
        num_ues,
    ):
        raise ValueError(
            "serving_bs must have shape [batch, UE]. "
            f"Expected {(batch_size, num_ues)}, "
            f"got {tuple(serving_bs.shape)}."
        )

    if serving_bs.device != h_freq.device:
        raise ValueError(
            "serving_bs and h_freq must be on "
            "the same device."
        )

    if torch.any(serving_bs < 0):
        raise ValueError(
            "serving_bs cannot contain negative indices."
        )

    if torch.any(serving_bs >= num_bs):
        raise ValueError(
            "serving_bs contains an invalid BS index."
        )

    subcarrier_tx_power_w = (
        compute_subcarrier_tx_power_w(
            config=config,
            num_subcarriers=num_subcarriers,
        )
    )

    noise_power_w = (
        compute_noise_power_per_subcarrier_w(
            config
        )
    )

    all_link_received_power_w = (
        compute_all_link_received_power_w(
            h_freq=h_freq,
            subcarrier_tx_power_w=(
                subcarrier_tx_power_w
            ),
        )
    )

    if bs_activity_mask is None:

        bs_activity_mask = torch.ones(
            batch_size,
            num_bs,
            num_symbols,
            num_subcarriers,
            dtype=torch.bool,
            device=h_freq.device,
        )

    else:

        expected_activity_shape = (
            batch_size,
            num_bs,
            num_symbols,
            num_subcarriers,
        )

        if tuple(bs_activity_mask.shape) != (
            expected_activity_shape
        ):
            raise ValueError(
                "bs_activity_mask must have shape "
                f"{expected_activity_shape}, got "
                f"{tuple(bs_activity_mask.shape)}."
            )

        bs_activity_mask = (
            bs_activity_mask.to(
                device=h_freq.device,
                dtype=torch.bool,
            )
        )

    expanded_activity_mask = (
        bs_activity_mask.unsqueeze(
            dim=1
        )
    )

    active_received_power_w = torch.where(
        expanded_activity_mask,
        all_link_received_power_w,
        torch.zeros_like(
            all_link_received_power_w
        ),
    )

    serving_mask = torch.nn.functional.one_hot(
        serving_bs,
        num_classes=num_bs,
    ).to(
        dtype=torch.bool,
    )

    serving_mask = (
        serving_mask
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

    desired_power_w = torch.where(
        serving_mask,
        active_received_power_w,
        torch.zeros_like(
            active_received_power_w
        ),
    ).sum(
        dim=2
    )

    interference_power_w = torch.where(
        ~serving_mask,
        active_received_power_w,
        torch.zeros_like(
            active_received_power_w
        ),
    ).sum(
        dim=2
    )

    denominator_w = (
        interference_power_w
        + noise_power_w
    )

    sinr_linear = (
        desired_power_w
        / denominator_w
    )

    if not torch.isfinite(
        sinr_linear
    ).all():
        raise ValueError(
            "Computed SINR contains non-finite values."
        )

    if torch.any(
        sinr_linear < 0
    ):
        raise ValueError(
            "SINR cannot be negative."
        )

    return SINRData(
        all_link_received_power_w=(
            all_link_received_power_w
        ),
        desired_power_w=desired_power_w,
        interference_power_w=(
            interference_power_w
        ),
        noise_power_w=noise_power_w,
        sinr_linear=sinr_linear,
    )


