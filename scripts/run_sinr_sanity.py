import torch

from oran_scheduler.phy.sinr import (
    SINRConfig,
    compute_noise_power_per_subcarrier_w,
    compute_siso_multicell_sinr,
    compute_subcarrier_tx_power_w,
    compute_total_tx_power_w,
)
from oran_scheduler.simulator.cell_association import (
    build_cell_association,
    print_cell_association_summary,
    validate_cell_association,
)
from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
    validate_frequency_channel,
)
from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
    validate_evaluation_topology,
)

def watt_to_dbm(
    power_w: torch.Tensor,
) -> torch.Tensor:
    """
    Convert watts to dBm.

        P_dBm = 10 log10(P_W) + 30
    """

    tiny = torch.finfo(
        power_w.dtype
    ).tiny

    safe_power = torch.clamp(
        power_w,
        min=tiny,
    )

    return (
        10.0 * torch.log10(safe_power)
        + 30.0
    )


def linear_to_db(
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a positive linear power ratio to dB.
    """

    tiny = torch.finfo(
        value.dtype
    ).tiny

    safe_value = torch.clamp(
        value,
        min=tiny,
    )

    return (
        10.0 * torch.log10(
            safe_value
        )
    )

def print_statistics(
    name: str,
    values: torch.Tensor,
    unit: str,
) -> None:
    """
    Print useful distribution statistics.
    """

    flattened = (
        values
        .detach()
        .flatten()
    )

    quantiles = torch.tensor(
        [
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
        ],
        dtype=flattened.dtype,
        device=flattened.device,
    )

    q = torch.quantile(
        flattened,
        quantiles,
    )

    print(f"{name}")
    print(
        f"  min:       "
        f"{flattened.min().item():10.3f} {unit}"
    )
    print(
        f"  5th pct:   "
        f"{q[0].item():10.3f} {unit}"
    )
    print(
        f"  25th pct:  "
        f"{q[1].item():10.3f} {unit}"
    )
    print(
        f"  median:    "
        f"{q[2].item():10.3f} {unit}"
    )
    print(
        f"  mean:      "
        f"{flattened.mean().item():10.3f} {unit}"
    )
    print(
        f"  75th pct:  "
        f"{q[3].item():10.3f} {unit}"
    )
    print(
        f"  95th pct:  "
        f"{q[4].item():10.3f} {unit}"
    )
    print(
        f"  max:       "
        f"{flattened.max().item():10.3f} {unit}"
    )


def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Milestone 4A: Physical Multicell SISO SINR")
    print("=" * 72)
    print(f"Device: {device}")
    print()

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,

        # Paper training setup:
        # 420 generated UEs over 21 sectors.
        num_ut_per_sector=20,

        device=device,
    )

    topology = generate_topology(
        topology_config
    )

    validate_evaluation_topology(
        topology=topology,
        config=topology_config,
    )

    print(
        "Topology:                 "
        f"{topology_config.num_cells} cells, "
        f"{topology_config.num_ues} UEs"
    )

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,
        subcarrier_spacing_hz=30e3,
        num_rbs=18,
        subcarriers_per_rb=12,
        num_ofdm_symbols=1,
        device=device,
    )

    channel = generate_frequency_channel(
        topology=topology,
        topology_config=topology_config,
        channel_config=channel_config,
    )

    validate_frequency_channel(
        channel=channel,
        topology=topology,
        channel_config=channel_config,
    )

    print(
        "Channel shape:            "
        f"{tuple(channel.h_freq.shape)}"
    )

    print(
        "Channel device:           "
        f"{channel.h_freq.device}"
    )

    association = build_cell_association(
        h_freq=channel.h_freq,
        precision=channel_config.precision,
    )

    validate_cell_association(
        association=association,
        expected_batch_size=(
            topology_config.batch_size
        ),
        expected_num_ues=(
            topology_config.num_ues
        ),
        expected_num_bs=(
            topology_config.num_cells
        ),
    )

    print_cell_association_summary(
        association
    )

    sinr_config = SINRConfig(
        # Bell Labs paper:
        tx_power_dbm=44.0,

        # Bell Labs paper:
        subcarrier_spacing_hz=30e3,

        # Open-reproduction modeling value.
        temperature_k=294.0,

        # Paper does not publicly specify this.
        # Therefore do not invent a value yet.
        receiver_noise_figure_db=None,

        precision=channel_config.precision,

        device=device,
    )

    total_tx_power_w = (
        compute_total_tx_power_w(
            sinr_config
        )
    )

    subcarrier_tx_power_w = (
        compute_subcarrier_tx_power_w(
            config=sinr_config,
            num_subcarriers=(
                channel_config.num_subcarriers
            ),
        )
    )

    noise_power_w = (
        compute_noise_power_per_subcarrier_w(
            sinr_config
        )
    )

    print()
    print("=" * 72)
    print("Link Budget Assumptions")
    print("=" * 72)

    print(
        "Total BS Tx power:        "
        f"{sinr_config.tx_power_dbm:.2f} dBm"
    )

    print(
        "Total BS Tx power:        "
        f"{total_tx_power_w.item():.6f} W"
    )

    print(
        "Occupied subcarriers:     "
        f"{channel_config.num_subcarriers}"
    )

    print(
        "Tx power/subcarrier:      "
        f"{subcarrier_tx_power_w.item():.6e} W"
    )

    print(
        "Tx power/subcarrier:      "
        f"{watt_to_dbm(subcarrier_tx_power_w).item():.3f} dBm"
    )

    print(
        "Thermal noise/subcarrier: "
        f"{noise_power_w.item():.6e} W"
    )

    print(
        "Thermal noise/subcarrier: "
        f"{watt_to_dbm(noise_power_w).item():.3f} dBm"
    )

    print(
        "Receiver noise figure:    "
        "not applied"
    )

    print("=" * 72)


    sinr_data = compute_siso_multicell_sinr(
        h_freq=channel.h_freq,
        serving_bs=association.serving_bs,
        config=sinr_config,

        # Current full-load sanity assumption:
        # every cell transmits on every subcarrier.
        bs_activity_mask=None,
    )

    print()
    print("=" * 72)
    print("SINR Tensor Shapes")
    print("=" * 72)

    all_link_shape = tuple(
        sinr_data.all_link_received_power_w.shape
    )

    desired_shape = tuple(
        sinr_data.desired_power_w.shape
    )

    interference_shape = tuple(
        sinr_data.interference_power_w.shape
    )

    sinr_shape = tuple(
        sinr_data.sinr_linear.shape
    )

    print(
        "All-link Rx power:        "
        f"{all_link_shape}"
    )

    print(
        "Desired power:            "
        f"{desired_shape}"
    )

    print(
        "Interference power:       "
        f"{interference_shape}"
    )

    print(
        "SINR:                     "
        f"{sinr_shape}"
    )
    print("=" * 72)

    desired_dbm = watt_to_dbm(
        sinr_data.desired_power_w
    )

    interference_dbm = watt_to_dbm(
        sinr_data.interference_power_w
    )

    sinr_db = linear_to_db(
        sinr_data.sinr_linear
    )

    print()
    print("=" * 72)
    print("Physical Link Statistics")
    print("=" * 72)
    print()

    print_statistics(
        name="Desired received power",
        values=desired_dbm,
        unit="dBm",
    )

    print()

    print_statistics(
        name="Aggregate inter-cell interference",
        values=interference_dbm,
        unit="dBm",
    )

    print()

    print_statistics(
        name="Downlink SINR",
        values=sinr_db,
        unit="dB",
    )

    print()

    interference_limited_fraction = (
        sinr_data.interference_power_w
        > sinr_data.noise_power_w
    ).float().mean()

    print(
        "Resources where interference > thermal noise: "
        f"{100.0 * interference_limited_fraction.item():.2f}%"
    )

    example_ue = 0

    example_serving_bs = int(
        association.serving_bs[
            0,
            example_ue,
        ].item()
    )

    example_sinr_db = (
        sinr_db[
            0,
            example_ue,
            0,
            :,
        ]
    )

    print()
    print("=" * 72)
    print(f"Example UE {example_ue}")
    print("=" * 72)

    print(
        f"Serving BS:               "
        f"{example_serving_bs}"
    )

    print(
        "Mean subcarrier SINR:     "
        f"{example_sinr_db.mean().item():.3f} dB"
    )

    print(
        "Min subcarrier SINR:      "
        f"{example_sinr_db.min().item():.3f} dB"
    )

    print(
        "Max subcarrier SINR:      "
        f"{example_sinr_db.max().item():.3f} dB"
    )

    example_rbg_sinr_db = (
        example_sinr_db.reshape(
            18,
            12,
        )
    )

    print()
    print("Per-RBG subcarrier SINR ranges")

    for rbg_index in range(18):

        values = example_rbg_sinr_db[
            rbg_index
        ]

        print(
            f"  RBG {rbg_index + 1:2d}: "
            f"min={values.min().item():7.3f} dB, "
            f"median={values.median().item():7.3f} dB, "
            f"max={values.max().item():7.3f} dB"
        )

    assert tuple(
        sinr_data.sinr_linear.shape
    ) == (
        topology_config.batch_size,
        topology_config.num_ues,
        channel_config.num_ofdm_symbols,
        channel_config.num_subcarriers,
    )

    assert torch.isfinite(
        sinr_data.sinr_linear
    ).all()

    assert torch.all(
        sinr_data.sinr_linear >= 0
    )

    assert torch.all(
        sinr_data.desired_power_w > 0
    )

    assert torch.all(
        sinr_data.interference_power_w >= 0
    )

    assert (
        sinr_data.noise_power_w.item()
        > 0.0
    )

    print()
    print("=" * 72)
    print("Physical multicell SISO SINR sanity check PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
