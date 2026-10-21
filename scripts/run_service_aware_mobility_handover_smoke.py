#!/usr/bin/env python3
"""Two-cell geometry → handover → QoS candidate pool → TB packet delivery.

Uses GPU tensors for geometry/measurements with --device cuda:0. The physical
link estimate is a deterministic diagnostic, not a real Sionna transmission.
This is a *handover-state integration* smoke, not full NR multicell realism.
"""
from __future__ import annotations

import argparse
import torch

from oran_scheduler.simulator.topology import TopologyData
from oran_scheduler.simulator.mobility import MobilityConfig, MobilityEngine
from oran_scheduler.simulator.handover import HandoverController, HandoverConfig
from oran_scheduler.simulator.service_aware_traffic import (
    ServiceAwareTrafficManager, QoSFlowSpec, FlowPacketArrival,
)
from oran_scheduler.simulator.service_aware_observation import ServiceAwareQoSObserver
from oran_scheduler.simulator.service_aware_mobility_handover import ServiceAwareMobilityHandoverBridge
from oran_scheduler.simulator.service_aware_transmission import ServiceAwareTransmissionEngine
from oran_scheduler.simulator.service_aware_downlink import (
    ServiceAwareDownlinkCellRunner, DownlinkSchedulingDecision,
)
from oran_scheduler.simulator.service_aware_nr_grants import (
    NRBandwidthPart, NRGrantAssembler, PDSCHDataRegion, WholeGrantLink,
)
from oran_scheduler.schedulers.allocation import CellAllocation


def build_smoke(device: str):
    # Cell boundary x≈5m; 4m/s, 0.5ms/TTI => +2mm per slot.
    # Deliberately starts within millimeters of the boundary so a 5-TTI
    # smoke witnesses an event. No claim that real handovers happen in 5TTIs.
    loc=torch.tensor([[[4.997,0.,1.5],[1.,0.,1.5]]],device=device)
    bs=torch.tensor([[[0.,0.,25.],[10.,0.,25.]]],device=device)
    topo=TopologyData(
        ut_loc=loc,bs_loc=bs,
        ut_orientations=torch.zeros((1,2,3),device=device),
        bs_orientations=torch.zeros((1,2,3),device=device),
        ut_velocities=torch.tensor([[[4.,0.,0.],[0.,0.,0.]]],device=device),
        in_state=torch.zeros((1,2),dtype=torch.int32,device=device),
        los=None,bs_virtual_loc=torch.zeros((1,2,2,3),device=device),grid=object(),
    )
    q=ServiceAwareTrafficManager(
        num_ues=2,tti_duration_s=0.0005,
        flows=(QoSFlowSpec(0,'critical',1,10,pdb_ttis=10),
               QoSFlowSpec(1,'ftp',9,50)),
    )
    mobility=MobilityEngine(initial_topology=topo,config=MobilityConfig(tti_duration_s=0.0005))
    handover=HandoverController(initial_serving_bs=torch.zeros(2,device=device,dtype=torch.long),
                                num_bs=2,config=HandoverConfig(hysteresis_db=0.,time_to_trigger_ttis=2))
    def measure(snapshot):
        # Inverse path loss proxy for deterministic geometry smoke ONLY.
        u=snapshot.topology.ut_loc[0,:,:2]
        b=snapshot.topology.bs_loc[0,:,:2]
        return (1.+torch.sum((u[:,None,:]-b[None,:,:])**2,dim=-1)).reciprocal()
    route=ServiceAwareMobilityHandoverBridge(mobility=mobility,handover=handover,
                                              queues=q,measurements=measure)
    used_cell=[]
    events=[]
    def policy(obs):
        step=route.advance(obs.tti_index)
        for event in step.handovers:
            events.append((event.tti_index,event.global_ue_id,event.old_cell,event.new_cell,event.preserved_queue_bytes))
        # Only one cell scheduled in this diagnostic; production multicell
        # execution still needs a separate coordinated multi-cell runner.
        cell=1 if route.eligible_ue_ids(1) else 0
        view=route.cell_observation(obs,cell)
        candidates=tuple(u.ue_index for u in view.ues if u.backlog_bytes)
        if cell==1 and candidates:
            used_cell.append((obs.tti_index,cell,candidates))
            actions=torch.tensor([[0]+[-1]*17],dtype=torch.long,device=device)
            return DownlinkSchedulingDecision(CellAllocation(actions),candidates,None)
        actions=torch.full((1,18),-1,dtype=torch.long,device=device)
        return DownlinkSchedulingDecision(CellAllocation(actions),(0,),None)
    def link(obs,decision):
        if bool(torch.any(decision.allocation.candidate_by_user_slot>=0).item()):
            assert route.serving_cell(decision.queue_ue_by_candidate[0])==1
            return {0:WholeGrantLink(0,(0,),20,0.)}
        return {}
    def pack(obs,decision,tbs):
        if not tbs:return {}
        amount=min(tbs[0].payload_capacity_bytes,q.flow_snapshot(0,'critical').backlog_bytes)
        return {(0,'critical'):amount} if amount else {}
    grants=NRGrantAssembler(bandwidth=NRBandwidthPart(18,1),region=PDSCHDataRegion(),
                           channel_num_prbs=18,tbs_backend=lambda specs:[8*64 for _ in specs])
    runner=ServiceAwareDownlinkCellRunner(
        queues=q,observer=ServiceAwareQoSObserver(queues=q),
        decoder=ServiceAwareTransmissionEngine(seed=2),grants=grants,
        policy=policy,link_provider=link,flow_packer=pack,
    )
    return runner,route,events,used_cell


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA mode requires an allocated GPU')
    runner,route,events,used=build_smoke(args.device)
    with torch.no_grad():
        for t in range(5):
            runner.step(tti_index=t,arrivals=(FlowPacketArrival(0,'critical',40),) if t==0 else ())
        report=runner.report()
    assert events==[(3,0,0,1,40)],events
    assert used==[(3,1,(0,))],used
    assert report.arrived_bytes==report.delivered_bytes==40
    assert report.final_backlog_bytes==0
    assert route.measurement_calls==5
    assert route.membership_generation==1
    print('MOBILITY_HANDOVER_EVENT:',events)
    print('NEW_CELL_TDS_ELIGIBILITY:',used)
    print('PACKET_DELIVERY_BYTES:',report.delivered_bytes)
    print('SERVICE_AWARE_MOBILITY_HANDOVER_COMPONENT_PASS')


if __name__=='__main__':
    main()
