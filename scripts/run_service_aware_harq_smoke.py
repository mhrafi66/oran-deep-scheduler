#!/usr/bin/env python3
"""Task-6 HARQ lifecycle smoke. No heavy Sionna channel recomputation.

The existing Task-5 --real-sionna smoke separately validates the unchanged
physical path. Here fixed whole-TB link outcomes isolate HARQ correctness.
"""
from __future__ import annotations

import torch
from oran_scheduler.schedulers.allocation import CellAllocation
from oran_scheduler.simulator.service_aware_downlink import (
    DownlinkSchedulingDecision, ServiceAwareDownlinkCellRunner,
)
from oran_scheduler.simulator.service_aware_harq import ServiceAwareHARQEngine
from oran_scheduler.simulator.service_aware_nr_grants import (
    NRBandwidthPart, NRGrantAssembler, PDSCHDataRegion, WholeGrantLink,
)
from oran_scheduler.simulator.service_aware_observation import ServiceAwareQoSObserver
from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival, QoSFlowSpec, ServiceAwareTrafficManager,
)


def main():
    queues=ServiceAwareTrafficManager(
        num_ues=1,tti_duration_s=.0005,
        flows=(QoSFlowSpec(0,'critical',1,10,pdb_ttis=3,expiry_policy='retain'),),
    )
    harq=ServiceAwareHARQEngine(seed=3,feedback_delay_ttis=2,max_transmissions=3)
    plan=CellAllocation(torch.tensor([[0,0,0]],dtype=torch.long))
    empty=CellAllocation(torch.full((1,3),-1,dtype=torch.long))

    def policy(obs):
        states=harq.statuses()
        if states and states[0].state=='awaiting_feedback':
            return DownlinkSchedulingDecision(empty,(0,),None)
        return DownlinkSchedulingDecision(plan,(0,),None)

    attempt_count=0
    def link(obs,dec):
        nonlocal attempt_count
        if dec.allocation is empty:
            return {}
        attempt_count+=1
        # First TB fails; retry decodes. No artificial BLER combining.
        return {0:WholeGrantLink(0,(0,1,2),20,
                   tbler=1.0 if attempt_count==1 else 0.0)}

    def pack(obs,dec,tbs):
        states=harq.statuses()
        if states and states[0].state=='retransmit':
            return dict(states[0].planned_flow_bytes)
        return {(0,'critical'):min(tbs[0].payload_capacity_bytes,
                                    obs.flows[0].backlog_bytes)} if tbs else {}

    runner=ServiceAwareDownlinkCellRunner(
        queues=queues,observer=ServiceAwareQoSObserver(queues=queues),
        decoder=harq,
        grants=NRGrantAssembler(
            bandwidth=NRBandwidthPart(3,1),region=PDSCHDataRegion(),
            channel_num_prbs=3,tbs_backend=lambda spec:[8*64 for _ in spec],
        ),
        policy=policy,link_provider=link,flow_packer=pack,
    )
    for tti in range(4):
        result=runner.step(tti_index=tti,arrivals=(FlowPacketArrival(0,'critical',13),)
                           if tti==0 else ())
        print(f"TTI {tti}: delivered={result.transmission.traffic_service.delivered_bytes} B, "
              f"backlog={result.backlog_after_bytes} B, HARQ={harq.statuses()}")
    report=runner.report()
    c=harq.counters()
    assert report.arrived_bytes==13
    assert report.delivered_bytes==13 and report.final_backlog_bytes==0
    assert report.flows[0].on_time_completed_packets==1
    assert c.initial_transmissions==1 and c.retransmissions==1
    assert c.nack_received==1 and c.pending_processes==1
    print('SERVICE_AWARE_TASK6_HARQ_SMOKE_PASS')


if __name__=='__main__':
    main()
