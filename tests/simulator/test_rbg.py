from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
)

from oran_scheduler.simulator.rbg import (
    RBGConfig,
    compute_rbg_channel_power,
    validate_rbg_channel,
)

from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
)


def test_training_rbg_mapping() -> None:
    topology_config = TopologyConfig(
        device="cuda:0",
    )

    channel_config = ChannelConfig()

    rbg_config = RBGConfig()

    topology = generate_topology(
        config=topology_config,
    )

    channel = generate_frequency_channel(
        topology=topology,
        topology_config=topology_config,
        channel_config=channel_config,
    )

    rbg_data = compute_rbg_channel_power(
        channel=channel,
        config=rbg_config,
    )

    validate_rbg_channel(
        rbg_data=rbg_data,
        channel=channel,
        config=rbg_config,
    )