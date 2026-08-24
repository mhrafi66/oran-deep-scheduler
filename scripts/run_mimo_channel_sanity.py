import torch

from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
)
from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
)

def main() -> None:
    device = "cuda:0"

    print()
    print("=" * 72)
    print("Milestone 5A: Paper-Antenna MIMO Channel")
    print("=" * 72)

    topology_config = TopologyConfig(
        batch_size=1,
        num_rings=1,

        # Memory-safe MIMO smoke test:
        # one generated UE per sector.
        num_ut_per_sector=1,

        device=device,
    )

    topology = generate_topology(
        topology_config
    )

    num_cells = topology_config.num_cells
    num_ues = topology_config.num_ues

    print(
        "Cells:                    "
        f"{num_cells}"
    )

    print(
        "UEs for smoke test:       "
        f"{num_ues}"
    )

    channel_config = ChannelConfig(
        carrier_frequency_hz=4.0e9,
        subcarrier_spacing_hz=30e3,
        num_rbs=18,
        subcarriers_per_rb=12,
        num_ofdm_symbols=1,

        antenna_mode="paper",

        device=device,
    )

    channel = generate_frequency_channel(
        topology=topology,
        topology_config=topology_config,
        channel_config=channel_config,
    )

    num_bs_ant = int(
        channel.bs_array.num_ant
    )

    num_ut_ant = int(
        channel.ut_array.num_ant
    )

    channel_shape = tuple(
        channel.h_freq.shape
    )

    print()
    print(
        "gNB antenna ports:        "
        f"{num_bs_ant}"
    )

    print(
        "UE receive ports:         "
        f"{num_ut_ant}"
    )

    print(
        "Channel shape:            "
        f"{channel_shape}"
    )

    print(
        "Channel device:           "
        f"{channel.h_freq.device}"
    )

    expected_shape = (
        1,
        topology_config.num_ues,
        4,
        topology_config.num_cells,
        192,
        1,
        216,
    )

    assert channel_shape == expected_shape

    num_elements = (
        channel.h_freq.numel()
    )

    bytes_per_element = (
        channel.h_freq.element_size()
    )

    channel_bytes = (
        num_elements
        * bytes_per_element
    )

    channel_gib = (
        channel_bytes
        / (1024 ** 3)
    )

    print(
        "H tensor size:            "
        f"{channel_gib:.3f} GiB"
    )

    assert torch.is_complex(
        channel.h_freq
    )

    assert torch.isfinite(
        channel.h_freq.real
    ).all()

    assert torch.isfinite(
        channel.h_freq.imag
    ).all()

    assert torch.any(
        torch.abs(channel.h_freq) > 0
    )

    print()
    print("=" * 72)
    print(
        "Paper-antenna MIMO channel sanity check PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

    