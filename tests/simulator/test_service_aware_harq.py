"""Opt-in Task-6 HARQ regression: feedback chronology, FIFO payload identity, QoS."""
from __future__ import annotations

import pytest
import torch

from oran_scheduler.schedulers.allocation import CellAllocation
from oran_scheduler.simulator.service_aware_harq import ServiceAwareHARQEngine
from oran_scheduler.simulator.service_aware_downlink import (
    DownlinkSchedulingDecision, ServiceAwareDownlinkCellRunner,
)
from oran_scheduler.simulator.service_aware_nr_grants import (
    NRBandwidthPart, NRGrantAssembler, PDSCHDataRegion, WholeGrantLink,
)
from oran_scheduler.simulator.service_aware_observation import ServiceAwareQoSObserver
from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival, QoSFlowSpec, ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_transmission import DownlinkTB


def build(*, flows=None, failures=(1., 0., 0.), feedback=2, max_tx=3,
          num_ues=1, capacity_bytes=64):
    if flows is None:
        flows = [QoSFlowSpec(0, 'critical', 1, 10, pdb_ttis=10)]
    q = ServiceAwareTrafficManager(num_ues=num_ues, flows=flows,
                                   tti_duration_s=0.0005)
    h = ServiceAwareHARQEngine(seed=7, feedback_delay_ttis=feedback,
                               max_transmissions=max_tx)
    allocations = []
    views = []
    def policy(view):
        views.append((view.tti_index, view.total_backlog_bytes,
                      tuple((s.ue_index,s.state) for s in h.statuses())))
        statuses = {s.ue_index:s for s in h.statuses()}
        ready = [ue for ue,s in statuses.items() if s.state == 'retransmit']
        if ready:
            ue = ready[0]
        else:
            ue = next((i for i in range(num_ues)
                       if view.ues[i].backlog_bytes > 0 and i not in statuses), None)
        if ue is None:
            candidate = -1
        else:
            candidate = ue
        allocations.append(candidate)
        # Mapping candidate->UE is identity in these tests.
        a = CellAllocation(torch.full((1,3), candidate, dtype=torch.long))
        return DownlinkSchedulingDecision(a, tuple(range(num_ues)), None)
    attempts = []
    def link(view, decision):
        if allocations[-1] == -1:
            return {}
        ue = allocations[-1]
        attempts.append((view.tti_index, ue))
        return {ue: WholeGrantLink(ue, (0,1,2), 20,
                  failures[min(len(attempts)-1,len(failures)-1)], 1)}
    def pack(view, decision, tbs):
        ready = {s.ue_index:s for s in h.statuses() if s.state=='retransmit'}
        plan = {}
        for tb in tbs:
            if tb.ue_index in ready:
                plan.update(dict(ready[tb.ue_index].planned_flow_bytes))
            else:
                remain = tb.payload_capacity_bytes
                for f in view.flows:
                    if f.ue_index != tb.ue_index: continue
                    n = min(remain, f.backlog_bytes)
                    if n:
                        plan[f.key] = n
                        remain -= n
        return plan
    runner = ServiceAwareDownlinkCellRunner(
        queues=q, observer=ServiceAwareQoSObserver(queues=q), decoder=h,
        grants=NRGrantAssembler(bandwidth=NRBandwidthPart(3,1),
            region=PDSCHDataRegion(),channel_num_prbs=3,
            tbs_backend=lambda specs: [capacity_bytes*8]*len(specs)),
        policy=policy, link_provider=link, flow_packer=pack,
    )
    return runner,h,views,attempts


def test_delayed_feedback_stop_and_wait_and_fifo_conservation():
    r,h,views,attempts = build(failures=(1.,0.,0.))
    for t,arrival in enumerate((
        (FlowPacketArrival(0,'critical',10),),
        (FlowPacketArrival(0,'critical',5),), (), (), (),
    )):
        r.step(tti_index=t,arrivals=arrival)
    assert attempts == [(0,0),(2,0),(4,0)]
    assert views[1][2] == ((0,'awaiting_feedback'),)
    assert views[2][2] == ((0,'retransmit'),)
    assert views[3][2] == ((0,'awaiting_feedback'),)
    assert views[4][2] == ()
    x = r.report()
    assert x.arrived_bytes == 15 and x.delivered_bytes == 15
    assert x.final_backlog_bytes == 0 and x.flows[0].completed_packets == 2
    c = h.counters()
    assert c.initial_transmissions == 2 and c.retransmissions == 1
    assert c.ack_received == 1 and c.nack_received == 1
    assert c.pending_processes == 1  # TTI4 successful TB, feedback pending


def test_expired_payload_cancels_retransmission_without_wrong_packet_credit():
    r,h,views,attempts = build(
        flows=[QoSFlowSpec(0,'critical',1,10,pdb_ttis=1,expiry_policy='drop')],
        failures=(1.,0.),feedback=2)
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',8),))
    r.step(tti_index=1,arrivals=(FlowPacketArrival(0,'critical',5),))
    x=r.report()
    assert x.expired_dropped_bytes==8 and x.delivered_bytes==5
    assert x.arrived_bytes==13 and x.final_backlog_bytes==0
    assert h.counters().cancelled_by_expiry==1
    assert attempts==[(0,0),(1,0)]  # no stale HARQ retry of the 8-byte packet


def test_retain_late_delivery_after_feedback():
    r,h,_,attempts = build(
        flows=[QoSFlowSpec(0,'critical',1,10,pdb_ttis=1,expiry_policy='retain')],
        failures=(1.,0.),feedback=2)
    for t in range(4):
        r.step(tti_index=t,arrivals=(FlowPacketArrival(0,'critical',7),) if t==0 else ())
    f=r.report().flows[0]
    assert attempts==[(0,0),(2,0)]
    assert f.deadline_misses==1 and f.late_completed_packets==1
    assert f.on_time_completed_packets==0 and f.delivered_bytes==7


def test_exhausted_process_releases_ue_but_keeps_rlc_queue():
    r,h,_,attempts = build(failures=(1.,1.,0.),max_tx=2,feedback=1)
    for t in range(3):
        r.step(tti_index=t, arrivals=(FlowPacketArrival(0,'critical',11),) if t==0 else ())
    assert attempts==[(0,0),(1,0),(2,0)]
    assert r.report().delivered_bytes==11
    assert h.counters().processes_exhausted==1
    assert h.counters().initial_transmissions==2
    assert h.counters().retransmissions==1


def test_waiting_ue_rejects_unsolicited_transmission_before_rng_or_service():
    q=ServiceAwareTrafficManager(num_ues=1,flows=[QoSFlowSpec(0,'f',1,10)],tti_duration_s=.0005)
    h=ServiceAwareHARQEngine(seed=5,feedback_delay_ttis=2)
    q.begin_tti(tti_index=0,arrivals=(FlowPacketArrival(0,'f',10),))
    h.prepare_tti(queues=q)
    first=h.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,64,1.0)],
                        planned_flow_bytes={(0,'f'):10})
    assert first.traffic_service.delivered_bytes==0
    q.begin_tti(tti_index=1,arrivals=())
    h.prepare_tti(queues=q)
    with pytest.raises(ValueError,match='waiting for HARQ feedback'):
        h.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,64,0.0)],
                      planned_flow_bytes={(0,'f'):10})
    assert q.has_active_tti and q.backlog_bytes==10
    h.execute_tti(queues=q,transport_blocks=[],planned_flow_bytes={})
    q.begin_tti(tti_index=2,arrivals=())
    h.prepare_tti(queues=q)
    with pytest.raises(ValueError,match='preserve TB size'):
        h.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,32,0.0)],
                      planned_flow_bytes={(0,'f'):10})
    with pytest.raises(ValueError,match='exact flow payload'):
        h.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,64,0.0)],
                      planned_flow_bytes={(0,'f'):9})
    assert q.backlog_bytes == 10
    res=h.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,64,0.0)],
                      planned_flow_bytes={(0,'f'):10})
    assert res.traffic_service.delivered_bytes==10


def test_no_early_ack_visibility_and_replay_reset():
    r,h,views,_=build(failures=(0.,))
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',3),))
    assert r.report().delivered_bytes==3
    assert h.statuses()[0].state=='awaiting_feedback'
    r.step(tti_index=1,arrivals=())
    assert views[1][2] == ((0,'awaiting_feedback'),)
    r.step(tti_index=2,arrivals=())
    assert views[2][2] == ()
    assert h.counters().ack_received == 1
    r.reset(seed=7)
    assert h.counters().pending_processes==0
    assert h.counters().initial_transmissions==0
    assert r.report().measured_ttis==0
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',3),))
    assert r.report().delivered_bytes==3


def test_harq_config_and_tti_lifecycle_guards():
    for key, val in [('feedback_delay_ttis',0),('max_transmissions',0),
                     ('feedback_delay_ttis',True),('max_transmissions',1.5)]:
        with pytest.raises(ValueError):ServiceAwareHARQEngine(**{key:val})
    q=ServiceAwareTrafficManager(num_ues=1,flows=[QoSFlowSpec(0,'f',1,10)],tti_duration_s=.0005)
    h=ServiceAwareHARQEngine()
    with pytest.raises(RuntimeError,match='requires begun'):
        h.prepare_tti(queues=q)
    q.begin_tti(tti_index=0,arrivals=())
    h.prepare_tti(queues=q)
    with pytest.raises(RuntimeError,match='out of order'):
        h.prepare_tti(queues=q)
    h.execute_tti(queues=q,transport_blocks=[],planned_flow_bytes={})
    q.begin_tti(tti_index=1,arrivals=())
    with pytest.raises(RuntimeError,match='prepare_tti'):
        h.execute_tti(queues=q,transport_blocks=[],planned_flow_bytes={})


def test_independent_ues_can_transmit_while_first_ue_awaits_harq():
    flows=[QoSFlowSpec(0,'a',1,10),QoSFlowSpec(1,'b',9,50)]
    q=ServiceAwareTrafficManager(num_ues=2,flows=flows,tti_duration_s=.0005)
    h=ServiceAwareHARQEngine(seed=2,feedback_delay_ttis=2)
    q.begin_tti(tti_index=0,arrivals=(FlowPacketArrival(0,'a',12),FlowPacketArrival(1,'b',9)))
    h.prepare_tti(queues=q)
    first=h.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,64,1.),DownlinkTB(1,64,0.)],
                        planned_flow_bytes={(0,'a'):12,(1,'b'):9})
    assert first.traffic_service.delivered_bytes==9
    q.begin_tti(tti_index=1,arrivals=(FlowPacketArrival(1,'b',6),))
    h.prepare_tti(queues=q)
    second=h.execute_tti(queues=q,transport_blocks=[],planned_flow_bytes={})
    assert second.traffic_service.delivered_bytes==0
    q.begin_tti(tti_index=2,arrivals=())
    h.prepare_tti(queues=q)
    assert h.statuses()[0].state=='retransmit'
    assert h.counters().ack_received==1
    third=h.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,64,0.),DownlinkTB(1,64,0.)],
                        planned_flow_bytes={(0,'a'):12,(1,'b'):6})
    assert third.traffic_service.delivered_bytes==18
    assert q.backlog_bytes==0
    assert h.counters().initial_transmissions==3
    assert h.counters().retransmissions==1


def test_no_process_for_padding_only_tb():
    q=ServiceAwareTrafficManager(num_ues=1,flows=[QoSFlowSpec(0,'a',1,10)],tti_duration_s=.0005)
    h=ServiceAwareHARQEngine(seed=2)
    q.begin_tti(tti_index=0,arrivals=())
    h.prepare_tti(queues=q)
    h.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,64,1.)],planned_flow_bytes={})
    assert h.counters().pending_processes==0


def test_engine_cannot_be_shared_between_two_cell_queue_owners():
    mk=lambda: ServiceAwareTrafficManager(num_ues=1,flows=[QoSFlowSpec(0,'a',1,10)],tti_duration_s=.0005)
    q0,q1=mk(),mk()
    h=ServiceAwareHARQEngine()
    q0.begin_tti(tti_index=0,arrivals=())
    h.prepare_tti(queues=q0)
    q1.begin_tti(tti_index=0,arrivals=())
    with pytest.raises(ValueError, match='shared across'):
        h.prepare_tti(queues=q1)
    with pytest.raises(ValueError, match='different queue'):
        h.execute_tti(queues=q1,transport_blocks=[],planned_flow_bytes={})
