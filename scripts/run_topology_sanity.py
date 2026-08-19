from pathlib import Path

from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
    print_topology_summary,
    save_topology_plot,
    validate_evaluation_topology
)


def main() -> None:
    config = TopologyConfig(
        device="cuda:0",
    )

    topology = generate_topology(config)

    validate_evaluation_topology(topology, config)

    print_topology_summary(topology, config)


    figure_path = save_topology_plot(
        topology=topology,
        output_path=Path(
            "outputs/figures/topology_sanity.png"
        ),
    )

    print()
    print("Validation: PASSED")
    print(f"Topology plot saved to: {figure_path}")


if __name__ == "__main__":
    main()