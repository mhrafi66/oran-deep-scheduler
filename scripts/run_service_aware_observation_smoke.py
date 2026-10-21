#!/usr/bin/env python3
"""Deterministic Task 4 smoke: arrivals -> pre-service QoS -> TB -> next QoS.

No Sionna channels or PHY recalculation: TB-level success is explicitly
supplied by the test driver. The legacy Nokia scheduling path is unchanged.
"""

from oran_scheduler.simulator.service_aware_observation import ServiceAwareQoSObserver
from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival, QoSFlowSpec, ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_transmission import (
    DownlinkTB, ServiceAwareTransmissionEngine,
)


def main() -> None:
    q = ServiceAwareTrafficManager(
        num_ues=2,
        tti_duration_s=0.0005,
        flows=(
            QoSFlowSpec(0, 'control', five_qi=7, priority=10, pdb_ttis=3, gbr_bps=80_000),
            QoSFlowSpec(0, 'download', five_qi=9, priority=90),
            QoSFlowSpec(1, 'idle', five_qi=9, priority=90),
        ),
    )
    obs = ServiceAwareQoSObserver(queues=q, gbr_window_ttis=4)
    decoder = ServiceAwareTransmissionEngine(seed=7)

    arrivals = (
        FlowPacketArrival(0, 'control', 100),
        FlowPacketArrival(0, 'download', 50),
    )
    q.begin_tti(tti_index=0, arrivals=arrivals)
    first = obs.observe()
    control = first.flows[0]
    assert (first.total_backlog_bytes, control.deadline_slack_ttis, control.gbr_deficit_bytes) == (150, 2, 5)
    outcome = decoder.execute_tti(
        queues=q,
        transport_blocks=(DownlinkTB(0, 120, 1.0),),
        planned_flow_bytes={(0, 'control'): 100, (0, 'download'): 20},
    )
    assert outcome.traffic_service.delivered_bytes == 0
    print('TTI 0: pre-service 150 B, control slack 2, TB failed, delivered 0 B')

    q.begin_tti(tti_index=1, arrivals=())
    second = obs.observe()
    control = second.flows[0]
    assert (control.backlog_bytes, control.deadline_slack_ttis) == (100, 1)
    assert (control.gbr_window_delivered_bytes, control.gbr_deficit_bytes) == (0, 10)
    outcome = decoder.execute_tti(
        queues=q,
        transport_blocks=(DownlinkTB(0, 100, 0.0),),
        planned_flow_bytes={(0, 'control'): 70, (0, 'download'): 30},
    )
    assert outcome.traffic_service.delivered_bytes == 100
    print('TTI 1: control slack 1, GBR deficit 10 B, delivered 100 B')

    q.begin_tti(tti_index=2, arrivals=())
    third = obs.observe()
    control = third.flows[0]
    assert (control.backlog_bytes, control.deadline_slack_ttis) == (30, 0)
    assert (control.gbr_window_delivered_bytes, control.gbr_deficit_bytes) == (70, 0)
    outcome = decoder.execute_tti(
        queues=q,
        transport_blocks=(DownlinkTB(0, 30, 0.0),),
        planned_flow_bytes={(0, 'control'): 30},
    )
    assert outcome.traffic_service.delivered_bytes == 30
    assert q.backlog_bytes == 20  # Download flow retains its remaining bytes.
    assert q.flow_snapshot(0, 'control').on_time_completed_packets == 1
    print('TTI 2: control slack 0, final 30 B delivered on deadline')
    print('SERVICE_AWARE_TASK4_SMOKE_PASS')


if __name__ == '__main__':
    main()
