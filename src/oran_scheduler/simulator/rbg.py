from dataclasses import dataclass

import torch

from oran_scheduler.simulator.channel import (
    ChannelData
)

@dataclass(frozen=True)
class RBGConfig:
    """
    RBG configuration for the reduced-bandwidth training setup.
    """
    num_rbgs: int = 18
    subcarriers_per_rb: int = 12

    # In the paper's reduced training configuration:
    #
    # 18 RBs -> 18 RBGs
    #
    # Therefore each RBG currently contains exactly one RB.
    rbs_per_rbg: int = 1

    @property
    def subcarriers_per_rbg(self) -> int:
        return (
            self.subcarriers_per_rb * self.rbs_per_rbg
        )

    @property
    def total_subcarriers(self) -> int:
        return (
            self.subcarriers_per_rbg * self.num_rbgs
        )

@dataclass
class RBGChannelData:
    """
    Frequency-selective channel quatities at RBG granularity, derived from the underlying ChannelData.
    """

    #shape should be [batch, UE, BS, RBG]

    power: torch.Tensor

def compute_rbg_channel_power(
        channel: ChannelData,
        config: RBGConfig
) -> RBGChannelData:

    """
    Conver the frequency-domain channel into RBG-level channel power.
    
    Input channel shape:
    [
    batch, UE, RX antennas, TX antennas, BS, TX antennas, OFDM symbols, subcarriers
    ]
    Current sanity configuration:

        [1, 210, 1, 21, 1, 1, 216]

    Output shape:

        [batch, UE, BS, RBG]

    Current expected output:

        [1, 210, 21, 18]   
    """

    h_freq = channel.h_freq

    actual_num_subcarriers = h_freq.shape[-1]

    if actual_num_subcarriers != config.total_subcarriers:
        raise ValueError(
            f"Expected {config.total_subcarriers} subcarriers, "
            f"but got {actual_num_subcarriers}."
        )

    channel_power = torch.abs(
        h_freq
    ) ** 2

    grouped_power = channel_power.reshape(
        *channel_power.shape[:-1],
        config.num_rbgs,
        config.subcarriers_per_rbg
    )

    power_per_rbg = grouped_power.mean(dim=-1)

    link_rbg_power = power_per_rbg.mean(
        dim = (2, 4, 5)
    )

    return RBGChannelData(
        power = link_rbg_power
    )


def validate_rbg_channel(
    rbg_data: RBGChannelData,
    channel: ChannelData,
    config: RBGConfig,
) -> None:
    """Validate dimensions and numerical properties."""

    expected_shape = (
        channel.h_freq.shape[0],
        channel.h_freq.shape[1],
        channel.h_freq.shape[3],
        config.num_rbgs,
    )

    assert tuple(rbg_data.power.shape) == expected_shape, (
        f"Expected RBG channel shape {expected_shape}, "
        f"got {tuple(rbg_data.power.shape)}."
    )

    assert torch.isfinite(
        rbg_data.power
    ).all(), (
        "RBG channel contains non-finite values."
    )

    assert torch.all(
        rbg_data.power >= 0
    ), (
        "Channel power cannot be negative."
    )

    assert torch.any(
        rbg_data.power > 0
    ), (
        "RBG channel is identically zero."
    )

def strongest_bs_per_ue(
    rbg_data: RBGChannelData,
) -> torch.Tensor:
    """
    Select the BS with the highest average channel power for each UE.

    Returns shape:

        [batch, UE]
    """

    average_link_power = rbg_data.power.mean(
        dim=-1,
    )

    return torch.argmax(
        average_link_power,
        dim=-1,
    )

def strongest_link_rbg_power(
    rbg_data: RBGChannelData,
) -> torch.Tensor:
    """
    Return every UE's 18-RBG channel-power vector for its
    strongest average BS.

    Output shape:

        [batch, UE, RBG]
    """

    strongest_bs = strongest_bs_per_ue(
        rbg_data
    )

    batch_size = rbg_data.power.shape[0]

    num_ues = rbg_data.power.shape[1]

    batch_indices = torch.arange(
        batch_size,
        device=rbg_data.power.device,
    ).unsqueeze(1)

    batch_indices = batch_indices.expand(
        batch_size,
        num_ues,
    )

    ue_indices = torch.arange(
        num_ues,
        device=rbg_data.power.device,
    ).unsqueeze(0)

    ue_indices = ue_indices.expand(
        batch_size,
        num_ues,
    )

    return rbg_data.power[
        batch_indices,
        ue_indices,
        strongest_bs,
        :,
    ]       



def compute_frequency_selectivity_statistics(
    rbg_data: RBGChannelData,
) -> dict[str, float]:
    """
    Quantify variation across the 18 RBGs.

    Statistics are calculated on each UE's strongest-BS link.
    """

    serving_power = strongest_link_rbg_power(
        rbg_data
    )

    tiny = torch.finfo(
        serving_power.dtype
    ).tiny

    power_db = 10.0 * torch.log10(
        torch.clamp(
            serving_power,
            min=tiny,
        )
    )

    std_db_per_ue = power_db.std(
        dim=-1,
    )

    span_db_per_ue = (
        power_db.max(dim=-1).values
        -
        power_db.min(dim=-1).values
    )

    return {
        "mean_rbg_std_db": (
            std_db_per_ue.mean().item()
        ),
        "median_rbg_std_db": (
            std_db_per_ue.median().item()
        ),
        "mean_rbg_span_db": (
            span_db_per_ue.mean().item()
        ),
        "median_rbg_span_db": (
            span_db_per_ue.median().item()
        ),
    }

def print_example_rbg_profile(
    rbg_data: RBGChannelData,
    ue_index: int = 0,
) -> None:
    """Print one UE's RBG power profile."""

    strongest_bs = strongest_bs_per_ue(
        rbg_data
    )

    bs_index = int(
        strongest_bs[0, ue_index].item()
    )

    power = rbg_data.power[
        0,
        ue_index,
        bs_index,
        :,
    ]

    tiny = torch.finfo(
        power.dtype
    ).tiny

    power_db = 10.0 * torch.log10(
        torch.clamp(
            power,
            min=tiny,
        )
    )

    print()
    print(
        f"Example UE {ue_index} "
        f"(strongest BS = {bs_index})"
    )

    for rbg_index, value_db in enumerate(
        power_db,
        start=1,
    ):
        print(
            f"  RBG {rbg_index:2d}: "
            f"{value_db.item():9.3f} dB"
        )

def print_rbg_summary(
    rbg_data: RBGChannelData,
    config: RBGConfig,
) -> None:
    """Print RBG mapping and frequency-selectivity diagnostics."""

    statistics = (
        compute_frequency_selectivity_statistics(
            rbg_data
        )
    )

    print("=" * 72)
    print("18-RBG Frequency-Selectivity Sanity Check")
    print("=" * 72)

    print(
        f"Number of RBGs:           "
        f"{config.num_rbgs}"
    )

    print(
        f"Subcarriers per RBG:      "
        f"{config.subcarriers_per_rbg}"
    )

    print(
        f"RBG tensor shape:         "
        f"{tuple(rbg_data.power.shape)}"
    )

    print(
        "Mean RBG std deviation:   "
        f"{statistics['mean_rbg_std_db']:.3f} dB"
    )

    print(
        "Median RBG std deviation: "
        f"{statistics['median_rbg_std_db']:.3f} dB"
    )

    print(
        "Mean max-min RBG span:     "
        f"{statistics['mean_rbg_span_db']:.3f} dB"
    )

    print(
        "Median max-min RBG span:   "
        f"{statistics['median_rbg_span_db']:.3f} dB"
    )

    print("=" * 72)

