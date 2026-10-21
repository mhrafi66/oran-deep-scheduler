"""Movement → radio measurements → TTT → cell eligibility → queue persistence."""
from __future__ import annotations

import pytest
import torch

from oran_scheduler.simulator.topology import TopologyData
from oran_scheduler.simulator.mobility import MobilityEngine, MobilityConfig
from oran_scheduler.simulator.handover import HandoverController, HandoverConfig
from oran_scheduler.simulator.service_aware_traffic import (
    ServiceAwareTrafficManager, QoSFlowSpec, FlowPacketArrival,
)
from oran_scheduler.simulator.service_aware_observation import ServiceAwareQoSObserver
from oran_scheduler.simulator.service_aware_mobility_handover import (
    ServiceAwareMobilityHandoverBridge,
)
from oran_scheduler.simulator.service_aware_transmission import ServiceAwareTransmissionEngine
from oran_scheduler.simulator.service_aware_downlink import (
    ServiceAwareDownlinkCellRunner, DownlinkSchedulingDecision,
)
from oran_scheduler.simulator.service_aware_nr_grants import (
    NRGrantAssembler, NRBandwidthPart, PDSCHDataRegion, WholeGrantLink,
)
from oran_scheduler.schedulers.allocation import CellAllocation


def make_topology(*, device='cpu', speed=4.0):
    return TopologyData(
        ut_loc=torch.tensor([[[4.997, 0, 1.5], [1.0, 0, 1.5]]], device=device),
        bs_loc=torch.tensor([[[0.,0.,25.], [10.,0.,25.]]], device=device),
        ut_orientations=torch.zeros((1, 2, 3), device=device),
        bs_orientations=torch.zeros((1, 2, 3), device=device),
        ut_velocities=torch.tensor([[[speed,0.,0.],[0.,0.,0.]]], device=device),
        in_state=torch.zeros((1,2),dtype=torch.int32, device=device),
        los=None,
        bs_virtual_loc=torch.zeros((1,2,2,3), device=device),
        grid=object(),
    )


def distance_powers(snapshot):
    # Deterministic path-gain diagnostic, NOT a Sionna/3GPP propagation model.
    ue=snapshot.topology.ut_loc[0,:,:2]
    bs=snapshot.topology.bs_loc[0,:,:2]
    d2=torch.sum((ue[:,None,:]-bs[None,:,:])**2,dim=-1)
    return (1.0+d2).reciprocal()


def make_system(*, device='cpu', speed=4.0, ttt=2, measurement=distance_powers):
    queues=ServiceAwareTrafficManager(
        num_ues=2, tti_duration_s=0.0005,
        flows=(QoSFlowSpec(0,'critical',1,10,pdb_ttis=15),
               QoSFlowSpec(1,'ftp',9,50)),
    )
    mob=MobilityEngine(initial_topology=make_topology(device=device,speed=speed),
                       config=MobilityConfig(tti_duration_s=0.0005))
    handover=HandoverController(
        initial_serving_bs=torch.zeros(2,dtype=torch.long,device=device),
        num_bs=2,config=HandoverConfig(hysteresis_db=0.,time_to_trigger_ttis=ttt),
    )
    bridge=ServiceAwareMobilityHandoverBridge(
        mobility=mob,handover=handover,queues=queues,measurements=measurement,
    )
    return queues,bridge


def test_geometry_handover_ttt_and_global_queue():
    queues,bridge=make_system()
    observer=ServiceAwareQoSObserver(queues=queues)
    positions=[]
    for t in range(5):
        queues.begin_tti(tti_index=t,arrivals=(FlowPacketArrival(0,'critical',50),) if t==0 else ())
        obs=observer.observe()
        snap=bridge.advance(t)
        positions.append(float(snap.geometry.topology.ut_loc[0,0,0]))
        assert bridge.measurement_calls==t+1
        assert bridge.cell_observation(obs,0).total_backlog_bytes == (50 if t<3 else 0)
        assert bridge.cell_observation(obs,1).total_backlog_bytes == (0 if t<3 else 50)
        assert bridge.eligible_ue_ids(0) == ((0,) if t<3 else ())
        assert bridge.eligible_ue_ids(1) == (() if t<3 else (0,))
        if t==3:
            assert len(snap.handovers)==1
            assert (snap.handovers[0].old_cell,snap.handovers[0].new_cell)==(0,1)
            assert snap.handovers[0].preserved_queue_bytes==50
            assert snap.membership_generation==1
            assert snap.ue_ids_by_cell==((1,),(0,))
        else:
            assert not snap.handovers
            assert snap.membership_generation==(0 if t<3 else 1)
        requested={(0,'critical'):50} if t==4 else {}
        # Service comes from the new cell at t=4, but the *same packet* completes.
        queues.apply_service(service_bytes_by_flow=requested)
    assert positions[1]-positions[0]==pytest.approx(0.002,abs=1e-6)
    assert positions[4]-positions[0]==pytest.approx(0.008,abs=1e-6)
    assert queues.flow_snapshot(0,'critical').delivered_bytes==50
    assert queues.flow_snapshot(0,'critical').on_time_completed_packets==1
    assert queues.backlog_bytes==0


def test_reject_invalid_measurements_and_duplicate_ttis():
    q,b=make_system(measurement=lambda x:torch.zeros((3,2)))
    with pytest.raises(ValueError,match='shape'):
        b.advance(0)
    assert b.measurement_calls==0
    # Reset is by constructing a new bridge/engine; do not continue after a bad source.
    q,b=make_system()
    b.advance(0)
    with pytest.raises(ValueError,match='contiguous'):
        b.advance(0)
    with pytest.raises(ValueError,match='contiguous'):
        b.advance(2)


def test_queue_clock_and_constructor_guards():
    q,b=make_system()
    q.begin_tti(tti_index=0,arrivals=())
    q.apply_service(service_bytes_by_flow={})
    with pytest.raises(RuntimeError,match='timeline'):
        b.advance(0)
    with pytest.raises(ValueError,match='agree'):
        other=ServiceAwareTrafficManager(num_ues=1,tti_duration_s=0.0005,
                                         flows=(QoSFlowSpec(0,'x',1,5),))
        ServiceAwareMobilityHandoverBridge(mobility=b.mobility,
                                           handover=b.handover,queues=other,
                                           measurements=distance_powers)


def test_cell_observation_does_not_leak_other_cells_qos():
    q,b=make_system()
    observer=ServiceAwareQoSObserver(queues=q)
    q.begin_tti(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',8), FlowPacketArrival(1,'ftp',12)))
    obs=observer.observe()
    b.advance(0)
    view0=b.cell_observation(obs,0)
    view1=b.cell_observation(obs,1)
    assert view0.total_backlog_bytes==20 and len(view0.flows)==2
    assert view1.total_backlog_bytes==0 and not view1.flows
    q.apply_service(service_bytes_by_flow={})


def test_reset_csi_epoch_generation_only_when_moving():
    q,b=make_system(speed=0)
    for t in range(6):
        q.begin_tti(tti_index=t,arrivals=())
        b.advance(t)
        q.apply_service(service_bytes_by_flow={})
    assert b.membership_generation==0
    assert b.measurement_calls==6
    assert b.eligible_ue_ids(0,require_backlog=False)==(0,1)
    assert b.eligible_ue_ids(1,require_backlog=False)==()


def test_actual_downlink_runner_delivers_after_new_cell_becomes_eligible():
    q,b=make_system()
    observer=ServiceAwareQoSObserver(queues=q)
    cell_used=[]
    def policy(global_obs):
        b.advance(global_obs.tti_index)
        cell=1 if b.eligible_ue_ids(1) else 0
        view=b.cell_observation(global_obs,cell)
        candidates=tuple(u.ue_index for u in view.ues if u.backlog_bytes)
        if cell==1 and candidates and global_obs.tti_index>=3:
            actions=torch.tensor([[0]+[-1]*17],dtype=torch.long)
            cell_used.append(cell)
            return DownlinkSchedulingDecision(CellAllocation(actions),candidates,None)
        return DownlinkSchedulingDecision(CellAllocation(torch.full((1,18),-1,dtype=torch.long)),(0,),None)
    def link(obs,decision):
        if int(torch.any(decision.allocation.candidate_by_user_slot>=0)):
            assert b.serving_cell(decision.queue_ue_by_candidate[0])==1
            return {0:WholeGrantLink(0,(0,),20,0.)}
        return {}
    def pack(obs,decision,tbs):
        return {(0,'critical'):min(tbs[0].payload_capacity_bytes,q.flow_snapshot(0,'critical').backlog_bytes)} if tbs else {}
    grants=NRGrantAssembler(bandwidth=NRBandwidthPart(18,1),region=PDSCHDataRegion(),
                            channel_num_prbs=18,tbs_backend=lambda specs:[8*64 for _ in specs])
    runner=ServiceAwareDownlinkCellRunner(queues=q,observer=observer,
         decoder=ServiceAwareTransmissionEngine(seed=2),grants=grants,
         policy=policy,link_provider=link,flow_packer=pack)
    for t in range(5):
        runner.step(tti_index=t,arrivals=(FlowPacketArrival(0,'critical',40),) if t==0 else ())
    report=runner.report()
    assert report.arrived_bytes==40 and report.delivered_bytes==40
    assert report.final_backlog_bytes==0 and cell_used==[1]
    assert report.on_time_ratio_of_arrived_packets==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires allocated CUDA GPU')
def test_cuda_geometry_and_routing():
    q,b=make_system(device='cuda:0')
    for t in range(4):
        q.begin_tti(tti_index=t, arrivals=())
        b.advance(t)
        q.apply_service(service_bytes_by_flow={})
    assert b.membership_generation==1
    assert b.serving_cell(0)==1


def test_transient_stronger_neighbor_does_not_immediately_handover():
    seq=[torch.tensor([[5.,1.],[5.,1.]]),
         torch.tensor([[1.,5.],[5.,1.]]),
         torch.tensor([[5.,1.],[5.,1.]])]
    q,b=make_system(measurement=lambda snapshot:seq[snapshot.tti_index])
    for t in range(3):
        q.begin_tti(tti_index=t,arrivals=())
        snap=b.advance(t)
        assert snap.membership_generation==0
        assert not snap.handovers
        q.apply_service(service_bytes_by_flow={})
    assert b.eligible_ue_ids(0,require_backlog=False)==(0,1)


def test_gbr_observation_history_survives_reassociation():
    q,b=make_system()
    obs=ServiceAwareQoSObserver(queues=q,gbr_window_ttis=5)
    for t in range(4):
        q.begin_tti(tti_index=t,arrivals=(FlowPacketArrival(0,'critical',50),) if t==0 else ())
        view=obs.observe()
        snap=b.advance(t)
        if t==3:
            projected=b.cell_observation(view,1)
            assert len(projected.flows)==1
            assert projected.flows[0].hol_delay_ttis==4
            assert projected.total_backlog_bytes==40
            assert snap.handovers[0].preserved_queue_bytes==40
        q.apply_service(service_bytes_by_flow={(0,'critical'):10} if t==1 else {})
    assert q.flow_snapshot(0,'critical').delivered_bytes==10
    assert q.flow_snapshot(0,'critical').backlog_bytes==40
