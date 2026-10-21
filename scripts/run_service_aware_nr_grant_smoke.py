#!/usr/bin/env python3
"""Task 3: real Sionna NR TB size + Task 1/2 decoded-service smoke.

Requires Sionna for the default full smoke (run on the CHPC compute node).
--geometry-only validates mapping/queue service with an intentionally fake
TBS backend, useful for testing the control path without Sionna, NEVER for
physical-rate or throughput evaluation.
"""
import argparse

import torch

from oran_scheduler.schedulers.allocation import CellAllocation
from oran_scheduler.simulator.service_aware_nr_grants import (
    NRBandwidthPart, PDSCHDataRegion, WholeGrantLink, NRGrantAssembler
)
from oran_scheduler.simulator.service_aware_traffic import (
    QoSFlowSpec, FlowPacketArrival, ServiceAwareTrafficManager
)
from oran_scheduler.simulator.service_aware_transmission import ServiceAwareTransmissionEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args()

    # The full-band mapping is geometry-only here: do not claim the legacy
    # 18-PRB PHY tensor has 273 PRBs!
    bwp = NRBandwidthPart(273, 16)
    assert bwp.num_rbgs == 18
    assert bwp.prbs_per_rbg == (16,) * 17 + (1,)

    fake = lambda specs: [8 * 30 for _ in specs]
    sizing = NRGrantAssembler(
        bandwidth=NRBandwidthPart(18, 1),
        region=PDSCHDataRegion(num_pdsch_symbols=14, dmrs_re_per_prb=12),
        channel_num_prbs=18,
        tbs_backend=fake if args.geometry_only else None,
    )
    actions = torch.full((2, 18), -1, dtype=torch.long)
    actions[0, 0:3] = 0  # 3 RBG, one UE, ONE TB
    actions[1, 0] = 1    # spatially shared first RBG, distinct UE
    allocation = CellAllocation(actions)
    links = {
        0: WholeGrantLink(0, (0,1,2), mcs_index=20, tbler=1.0, num_layers=2),
        1: WholeGrantLink(1, (0,), mcs_index=20, tbler=0.0, num_layers=1),
    }
    queues = ServiceAwareTrafficManager(
        num_ues=2, tti_duration_s=0.0005, flows=[
            QoSFlowSpec(0,'urgent',1,10,pdb_ttis=3),
            QoSFlowSpec(0,'bulk',9,50),
            QoSFlowSpec(1,'bulk',9,50),
        ]
    )
    decoder = ServiceAwareTransmissionEngine(seed=42)
    blocks = sizing.build(allocation=allocation, queue_ue_by_candidate=[0,1], whole_links=links)
    assert len(blocks) == 2
    print('RBG 0 shared by UE 0 and UE 1; UE 0 spans three RBGs yet has one TB')
    print('TB capacities:', [(x.ue_index, x.payload_capacity_bytes) for x in blocks])
    if min(x.payload_capacity_bytes for x in blocks) < 10:
        raise AssertionError('Unexpectedly small TB capacity in smoke scenario')
    queues.begin_tti(tti_index=0, arrivals=[
        FlowPacketArrival(0,'urgent',5), FlowPacketArrival(0,'bulk',5),
        FlowPacketArrival(1,'bulk',5),
    ])
    result = decoder.execute_tti(
        queues=queues, transport_blocks=blocks,
        planned_flow_bytes={(0,'urgent'):5,(0,'bulk'):5,(1,'bulk'):5},
    )
    assert result.traffic_service.delivered_bytes == 5  # UE 0 failed; UE 1 succeeded.
    assert queues.backlog_bytes == 10
    print('TTI 0: one UE TB failed, another decoded; delivered=5 B, backlog=10 B')
    links[0] = WholeGrantLink(0, (0,1,2), mcs_index=20, tbler=0.0, num_layers=2)
    blocks = sizing.build(allocation=allocation, queue_ue_by_candidate=[0,1], whole_links=links)
    queues.begin_tti(tti_index=1, arrivals=[])
    result = decoder.execute_tti(queues=queues,transport_blocks=blocks,planned_flow_bytes={(0,'urgent'):5,(0,'bulk'):5})
    assert result.traffic_service.delivered_bytes == 10
    assert queues.backlog_bytes == 0
    print('TTI 1: decoded whole UE TB, delivered=10 B, backlog=0 B')
    print('SERVICE_AWARE_TASK3_GEOMETRY_ONLY_PASS' if args.geometry_only else 'SERVICE_AWARE_TASK3_SMOKE_PASS')


if __name__ == '__main__':
    main()
