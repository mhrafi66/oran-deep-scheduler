"""CPU-light, PHY-free Task-1 smoke; run from a unified repository checkout.

This is a *traffic queue* smoke test, not a 5G PHY or scheduler simulation.
The budgets below stand in for already-successful per-flow payload bytes.
"""

from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival,
    QoSFlowSpec,
    ServiceAwareTrafficManager,
)


def main() -> None:
    queues = ServiceAwareTrafficManager(
        num_ues=2,
        tti_duration_s=0.0005,
        flows=[
            QoSFlowSpec(0, "control", five_qi=7, priority=20, pdb_ttis=2, gbr_bps=256_000),
            QoSFlowSpec(0, "download", five_qi=9, priority=90),
            QoSFlowSpec(1, "download", five_qi=9, priority=90),
        ],
    )
    trace = (
        (FlowPacketArrival(0, "control", 600), FlowPacketArrival(0, "download", 1000)),
        (FlowPacketArrival(1, "download", 500),),
        (),
    )
    # For this smoke, deliberately withhold control service until its deadline.
    grants = (
        {(0, "download"): 200},
        {(0, "control"): 300, (1, "download"): 500},
        {(0, "download"): 800},
    )
    for tti, (arrivals, flows) in enumerate(zip(trace, grants)):
        start = queues.begin_tti(tti_index=tti, arrivals=arrivals)
        result = queues.apply_service(service_bytes_by_flow=flows)
        print(
            f"TTI {tti}: arrivals={start.arrived_bytes} B, "
            f"expired={start.expired_dropped_bytes} B, "
            f"successfully_served={result.delivered_bytes} B, "
            f"pending={result.backlog_bytes} B"
        )
    assert queues.flow_snapshot(0, "control").deadline_misses == 1
    assert queues.backlog_bytes == 0
    print("SERVICE_AWARE_TASK1_SMOKE_PASS")


if __name__ == "__main__":
    main()
