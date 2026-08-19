from oran_scheduler.simulator.topology import (
    TopologyConfig,
    generate_topology,
    validate_evaluation_topology
)

def test_bell_labs_evaluation_topology() -> None:
    config = TopologyConfig(device="cuda:0")
    topology = generate_topology(config)
    validate_evaluation_topology(topology, config)