from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
    validate_frequency_channel,
)

from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
)


def test_4ghz_uma_frequency_channel() -> None:
    topology_config = TopologyConfig(
        device="cpu",
    )

    channel_config = ChannelConfig()

    topology = generate_topology(
        config=topology_config,
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