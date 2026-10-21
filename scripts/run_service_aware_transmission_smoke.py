"""CPU-light Task 2 smoke: explicit TB outcomes -> real per-flow FIFO."""

from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival,
    QoSFlowSpec,
    ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_transmission import (
    DownlinkTB,
    ServiceAwareTransmissionEngine,
)


def main() -> None:
    queues = ServiceAwareTrafficManager(
        num_ues=2, tti_duration_s=0.0005,
        flows=[
            QoSFlowSpec(0, "urgent", five_qi=7, priority=20, pdb_ttis=1),
            QoSFlowSpec(0, "bulk", five_qi=9, priority=90),
            QoSFlowSpec(1, "bulk", five_qi=9, priority=90),
        ],
    )
    phy = ServiceAwareTransmissionEngine(seed=2026)
    queues.begin_tti(tti_index=0, arrivals=[
        FlowPacketArrival(0, "urgent", 120),
        FlowPacketArrival(0, "bulk", 200),
        FlowPacketArrival(1, "bulk", 80),
    ])
    first = phy.execute_tti(
        queues=queues,
        transport_blocks=[DownlinkTB(0, 140, 1.0), DownlinkTB(1, 100, 0.0)],
        planned_flow_bytes={(0, "urgent"): 120, (1, "bulk"): 80},
    )
    print("TTI 0:", "decoded", first.decoded_capacity_bytes,
          "delivered", first.traffic_service.delivered_bytes,
          "pending", queues.backlog_bytes)
    assert first.traffic_service.delivered_bytes_by_ue == (0, 80)
    assert queues.ue_backlog_bytes() == (320, 0)

    second_start = queues.begin_tti(tti_index=1, arrivals=[])
    assert second_start.new_deadline_misses == 1
    assert second_start.expired_dropped_bytes == 120
    second = phy.execute_tti(
        queues=queues, transport_blocks=[DownlinkTB(0, 200, 0.0)],
        planned_flow_bytes={(0, "bulk"): 200},
    )
    print("TTI 1:", "expired", second_start.expired_dropped_bytes,
          "delivered", second.traffic_service.delivered_bytes,
          "pending", queues.backlog_bytes)
    assert second.traffic_service.delivered_bytes == 200
    assert queues.backlog_bytes == 0
    print("SERVICE_AWARE_TASK2_SMOKE_PASS")


if __name__ == "__main__":
    main()
