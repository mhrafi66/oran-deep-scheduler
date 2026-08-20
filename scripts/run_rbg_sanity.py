from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
    validate_frequency_channel,
)

from oran_scheduler.simulator.rbg import (
    RBGConfig,
    compute_rbg_channel_power,
    print_example_rbg_profile,
    print_rbg_summary,
    validate_rbg_channel,
)

from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
    validate_evaluation_topology,
)


def main() -> None:
    topology_config = TopologyConfig(
        device="cpu",
    )

    channel_config = ChannelConfig()

    rbg_config = RBGConfig()

    topology = generate_topology(
        config=topology_config,
    )

    validate_evaluation_topology(
        topology=topology,
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

    rbg_data = compute_rbg_channel_power(
        channel=channel,
        config=rbg_config,
    )

    validate_rbg_channel(
        rbg_data=rbg_data,
        channel=channel,
        config=rbg_config,
    )

    print_rbg_summary(
        rbg_data=rbg_data,
        config=rbg_config,
    )

    print_example_rbg_profile(
        rbg_data=rbg_data,
        ue_index=0,
    )

    print()
    print("Validation: PASSED")


if __name__ == "__main__":
    main()
