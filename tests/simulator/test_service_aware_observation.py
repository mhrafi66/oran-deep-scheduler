"""Task 4: pre-schedule flow/UE QoS observation and GBR accounting."""

from dataclasses import FrozenInstanceError

import pytest

from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival,
    QoSFlowSpec,
    ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_observation import ServiceAwareQoSObserver
from oran_scheduler.simulator.service_aware_transmission import (
    DownlinkTB,
    ServiceAwareTransmissionEngine,
)


def manager(*, specs=None, num_ues=2, tti_s=0.0005):
    if specs is None:
        specs = (
            QoSFlowSpec(0, 'urgent', 7, 10, pdb_ttis=3, gbr_bps=80_000),
            QoSFlowSpec(0, 'bulk', 9, 90),
            QoSFlowSpec(1, 'video', 8, 30, pdb_ttis=2, gbr_bps=160_000),
        )
    return ServiceAwareTrafficManager(num_ues=num_ues, flows=specs, tti_duration_s=tti_s)


def begin(q, t, *arr):
    return q.begin_tti(tti_index=t, arrivals=tuple(FlowPacketArrival(*x) for x in arr))


def end(q, **named_grants):
    service = {k: v for k, v in named_grants.items()}
    return q.apply_service(service_bytes_by_flow=service)


def views(o):
    return {(f.ue_index, f.flow_id): f for f in o.flows}


def test_flow_and_ue_summary_start_before_scheduling():
    q=manager(num_ues=4)
    obs=ServiceAwareQoSObserver(queues=q)
    begin(q, 0, (0,'urgent',100), (0,'bulk',240), (1,'video',90))
    result=obs.observe()
    f=views(result)
    assert result.tti_index==0 and result.total_backlog_bytes==430
    assert [u.backlog_bytes for u in result.ues] == [340,90,0,0]
    assert [(x.flow_id,x.backlog_bytes) for x in result.flows]==[('urgent',100),('bulk',240),('video',90)]
    assert (result.ues[0].active_flows,result.ues[0].highest_priority)==(2,10)
    assert (result.ues[0].min_deadline_slack_ttis,result.ues[0].max_hol_delay_ttis)==(2,1)
    assert (result.ues[1].highest_priority,result.ues[1].min_deadline_slack_ttis)==(30,1)
    assert result.ues[2].active_flows==0 and result.ues[2].highest_priority is None
    assert result.ues[2].max_hol_delay_ttis is None and result.ues[2].min_deadline_slack_ttis is None
    assert (f[0,'urgent'].five_qi, f[0,'urgent'].gbr_bps, f[0,'urgent'].priority)==(7,80_000,10)
    assert f[0,'bulk'].deadline_slack_ttis is None
    assert f[0,'bulk'].gbr_bps is None
    assert q.backlog_bytes == 430


def test_one_tti_gbr_target_and_backlog_cap():
    q=manager()
    obs=ServiceAwareQoSObserver(queues=q,gbr_window_ttis=4)
    begin(q,0,(0,'urgent',3),(1,'video',100))
    got=obs.observe()
    a=views(got)[0,'urgent']; b=views(got)[1,'video']
    # 80,000 bit/s x .0005s /8 = 5 bytes per backlogged TTI.
    assert a.gbr_window_backlogged_ttis==1
    assert a.gbr_target_bytes_in_window==5.0
    assert a.gbr_deficit_bytes==3  # cap to actual remaining queue
    assert b.gbr_target_bytes_in_window==10.0
    assert b.gbr_deficit_bytes==10
    assert got.ues[0].gbr_deficit_bytes==3


def test_gbr_uses_actual_successfully_delivered_bytes_not_capacity():
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q,gbr_window_ttis=3)
    engine=ServiceAwareTransmissionEngine(seed=1)
    begin(q,0,(0,'urgent',100))
    assert views(ob.observe())[0,'urgent'].gbr_deficit_bytes==5
    # Physical capacity 99 bytes but decoder fails -> delivered 0.
    r=engine.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,99,1.0)],planned_flow_bytes={(0,'urgent'):20})
    assert r.traffic_service.delivered_bytes==0
    begin(q,1)
    s=views(ob.observe())[0,'urgent']
    assert (s.gbr_window_backlogged_ttis,s.gbr_window_delivered_bytes,s.gbr_deficit_bytes)==(2,0,10)
    engine.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,99,0.0)],planned_flow_bytes={(0,'urgent'):18})
    begin(q,2)
    s=views(ob.observe())[0,'urgent']
    assert (s.gbr_window_backlogged_ttis,s.gbr_window_delivered_bytes,s.gbr_deficit_bytes)==(3,18,0)
    q.apply_service(service_bytes_by_flow={})


def test_gbr_trailing_window_evicts_old_service_and_demand():
    # No PDB for this continuous-backlog window accounting test; otherwise
    # Task 1 correctly expires the packet at TTI 3.
    q=manager(specs=(QoSFlowSpec(0,'urgent',7,10,gbr_bps=80_000),),num_ues=1)
    ob=ServiceAwareQoSObserver(queues=q,gbr_window_ttis=3)
    begin(q,0,(0,'urgent',100))
    x=views(ob.observe())[0,'urgent']
    assert (x.gbr_window_backlogged_ttis,x.gbr_window_delivered_bytes,x.gbr_deficit_bytes)==(1,0,5)
    q.apply_service(service_bytes_by_flow={(0,'urgent'):2})
    begin(q,1)
    x=views(ob.observe())[0,'urgent']
    assert (x.gbr_window_backlogged_ttis,x.gbr_window_delivered_bytes,x.gbr_deficit_bytes)==(2,2,8)
    q.apply_service(service_bytes_by_flow={(0,'urgent'):4})
    begin(q,2)
    x=views(ob.observe())[0,'urgent']
    assert (x.gbr_window_backlogged_ttis,x.gbr_window_delivered_bytes,x.gbr_deficit_bytes)==(3,6,9)
    q.apply_service(service_bytes_by_flow={(0,'urgent'):1})
    begin(q,3)
    x=views(ob.observe())[0,'urgent']
    # Previous slots 1,2 and current slot 3 (old slot 0 has aged out).
    assert (x.gbr_window_backlogged_ttis,x.gbr_window_delivered_bytes,x.gbr_deficit_bytes)==(3,5,10)
    q.apply_service(service_bytes_by_flow={})


def test_no_gbr_debt_accumulation_while_idle_then_new_arrival():
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q,gbr_window_ttis=10)
    for t in range(20):
        begin(q,t)
        x=views(ob.observe())[0,'urgent']
        assert (x.gbr_window_backlogged_ttis,x.gbr_deficit_bytes)==(0,0)
        q.apply_service(service_bytes_by_flow={})
    begin(q,20,(0,'urgent',100))
    x=views(ob.observe())[0,'urgent']
    assert (x.gbr_window_backlogged_ttis,x.gbr_target_bytes_in_window,x.gbr_deficit_bytes)==(1,5.0,5)
    q.apply_service(service_bytes_by_flow={})


def test_window_one_simplest_gbr_proxy():
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q,gbr_window_ttis=1)
    begin(q,0,(0,'urgent',40))
    assert views(ob.observe())[0,'urgent'].gbr_deficit_bytes==5
    q.apply_service(service_bytes_by_flow={(0,'urgent'):1})
    begin(q,1)
    s=views(ob.observe())[0,'urgent']
    assert (s.gbr_window_delivered_bytes,s.gbr_window_backlogged_ttis,s.gbr_deficit_bytes)==(0,1,5)
    q.apply_service(service_bytes_by_flow={})


def test_deadline_slack_zero_at_last_eligible_slot_then_drop():
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0,(0,'urgent',100))
    assert views(ob.observe())[0,'urgent'].deadline_slack_ttis==2
    q.apply_service(service_bytes_by_flow={})
    for t, expected in ((1,1),(2,0)):
        begin(q,t)
        s=views(ob.observe())[0,'urgent']
        assert (s.deadline_slack_ttis,s.hol_delay_ttis)==(expected,t+1)
        q.apply_service(service_bytes_by_flow={})
    start=begin(q,3)
    assert start.expired_dropped_bytes==100
    x=views(ob.observe())[0,'urgent']
    assert x.deadline_slack_ttis is None and x.backlog_bytes==0
    q.apply_service(service_bytes_by_flow={})


def test_deadline_slack_negative_for_retained_late_head():
    specs=(QoSFlowSpec(0,'retain',7,7,pdb_ttis=1,expiry_policy='retain'),)
    q=manager(specs=specs,num_ues=1)
    ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0,(0,'retain',30)); assert views(ob.observe())[0,'retain'].deadline_slack_ttis==0
    q.apply_service(service_bytes_by_flow={})
    start=begin(q,1)
    assert start.new_deadline_misses==1 and start.expired_dropped_bytes==0
    snap=views(ob.observe())[0,'retain']
    assert (snap.deadline_slack_ttis,snap.hol_delay_ttis,snap.expiry_policy)==(-1,2,'retain')
    q.apply_service(service_bytes_by_flow={})


def test_fifo_head_delay_after_partial_service():
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0,(0,'urgent',100),(0,'urgent',70))
    assert views(ob.observe())[0,'urgent'].queued_packets==2
    q.apply_service(service_bytes_by_flow={(0,'urgent'):120})
    begin(q,1)
    snap=views(ob.observe())[0,'urgent']
    assert (snap.backlog_bytes,snap.queued_packets,snap.hol_delay_ttis,snap.deadline_slack_ttis)==(50,1,2,1)
    q.apply_service(service_bytes_by_flow={})


def test_highest_priority_and_min_slack_only_for_backlogged_flows():
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0,(0,'bulk',50))
    x=ob.observe().ues[0]
    assert x.highest_priority==90 and x.min_deadline_slack_ttis is None
    q.apply_service(service_bytes_by_flow={})
    begin(q,1,(0,'urgent',2))
    x=ob.observe().ues[0]
    assert x.highest_priority==10 and x.min_deadline_slack_ttis==2
    q.apply_service(service_bytes_by_flow={})


def test_snapshots_are_immutable_and_dont_change_as_queue_advances():
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0,(0,'urgent',100))
    prior=ob.observe()
    with pytest.raises(FrozenInstanceError):
        prior.flows[0].backlog_bytes=123
    with pytest.raises(FrozenInstanceError):
        prior.ues[0].backlog_bytes=123
    q.apply_service(service_bytes_by_flow={(0,'urgent'):60})
    begin(q,1)
    current=ob.observe()
    assert prior.flows[0].backlog_bytes==100
    assert current.flows[0].backlog_bytes==40
    q.apply_service(service_bytes_by_flow={})


def test_requires_active_queue_and_no_duplicate_observation():
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q)
    with pytest.raises(RuntimeError,match='begin_tti'):
        ob.observe()
    begin(q,0)
    ob.observe()
    with pytest.raises(RuntimeError,match='expected TTI'):
        ob.observe()
    q.apply_service(service_bytes_by_flow={})
    with pytest.raises(RuntimeError,match='begin_tti'):
        ob.observe()


def test_missing_tti_observation_refused_not_silently_assumed():
    q=manager(); ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0); ob.observe(); q.apply_service(service_bytes_by_flow={})
    begin(q,1); q.apply_service(service_bytes_by_flow={})  # deliberately skip
    begin(q,2)
    with pytest.raises(RuntimeError,match='expected TTI 1'):
        ob.observe()
    q.apply_service(service_bytes_by_flow={})


def test_observer_must_be_created_before_first_tti():
    q=manager()
    begin(q,0)
    with pytest.raises(ValueError,match='before TTI 0'):
        ServiceAwareQoSObserver(queues=q)
    q.apply_service(service_bytes_by_flow={})
    with pytest.raises(ValueError,match='before TTI 0'):
        ServiceAwareQoSObserver(queues=q)


def test_reset_episode_requires_queue_reset_first_then_replays():
    q=manager(); ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0,(0,'urgent',100)); a=ob.observe(); q.apply_service(service_bytes_by_flow={(0,'urgent'):1})
    with pytest.raises(RuntimeError,match='Reset queues first'):
        ob.reset()
    q.reset()
    ob.reset()
    begin(q,0,(0,'urgent',100)); b=ob.observe()
    assert a==b
    q.apply_service(service_bytes_by_flow={})


def test_if_queue_resets_without_observer_reset_refuse_replay():
    q=manager(); ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0); ob.observe(); q.apply_service(service_bytes_by_flow={})
    q.reset()
    begin(q,0)
    with pytest.raises(RuntimeError,match='expected TTI 1'):
        ob.observe()
    q.apply_service(service_bytes_by_flow={})


def test_queue_reads_are_semantically_non_mutating():
    q=manager(); ob=ServiceAwareQoSObserver(queues=q)
    start=begin(q,0,(0,'urgent',50),(0,'bulk',60))
    before=(q.backlog_bytes,q.ue_backlog_bytes(),q.flow_snapshot(0,'urgent'),q.flow_snapshot(0,'bulk'))
    got=ob.observe()
    after=(q.backlog_bytes,q.ue_backlog_bytes(),q.flow_snapshot(0,'urgent'),q.flow_snapshot(0,'bulk'))
    assert start.backlog_bytes==110 and before==after
    assert got.total_backlog_bytes==110
    q.apply_service(service_bytes_by_flow={})


def test_local_only_queue_observations_independent_between_cells():
    a=manager(); b=manager()
    oa=ServiceAwareQoSObserver(queues=a); ob=ServiceAwareQoSObserver(queues=b)
    begin(a,0,(0,'urgent',100))
    begin(b,0,(1,'video',50))
    got_a=oa.observe(); got_b=ob.observe()
    assert (got_a.ues[0].backlog_bytes,got_a.ues[1].backlog_bytes)==(100,0)
    assert (got_b.ues[0].backlog_bytes,got_b.ues[1].backlog_bytes)==(0,50)
    a.apply_service(service_bytes_by_flow={}); b.apply_service(service_bytes_by_flow={})


def test_cached_ue_totals_match_aggregate_for_multiple_flows():
    specs=tuple(QoSFlowSpec(ue, f'f{i}', 9, 40) for ue in range(7) for i in range(3))
    q=manager(specs=specs,num_ues=7)
    ob=ServiceAwareQoSObserver(queues=q)
    arrivals=tuple((ue,f'f{i}',ue+1+i) for ue in range(7) for i in range(3))
    begin(q,0,*arrivals)
    snap=ob.observe()
    assert len(snap.flows)==21 and len(snap.ues)==7
    assert sum(u.backlog_bytes for u in snap.ues)==q.backlog_bytes
    for ue in range(7):
        assert snap.ues[ue].backlog_bytes==sum(f.backlog_bytes for f in snap.flows if f.ue_index==ue)
    q.apply_service(service_bytes_by_flow={})


def test_one_snapshot_per_configured_flow_not_per_queued_packet(monkeypatch):
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q)
    orig=q.flow_snapshot
    calls=[]
    def traced(ue,flow):
        calls.append((ue,flow))
        return orig(ue,flow)
    monkeypatch.setattr(q,'flow_snapshot',traced)
    begin(q,0,*[(0,'bulk',1) for _ in range(500)])
    obs=ob.observe()
    assert obs.total_backlog_bytes==500
    assert calls==[(0,'urgent'),(0,'bulk'),(1,'video')]
    q.apply_service(service_bytes_by_flow={})


@pytest.mark.parametrize('bad', (0,-1,True,1.1,'5',None))
def test_invalid_gbr_window_rejected(bad):
    with pytest.raises(ValueError,match='gbr_window_ttis'):
        ServiceAwareQoSObserver(queues=manager(),gbr_window_ttis=bad)


def test_wrong_queue_type_rejected():
    with pytest.raises(TypeError,match='ServiceAwareTrafficManager'):
        ServiceAwareQoSObserver(queues=object())


def test_gbr_without_backlog_does_not_obscure_priority():
    q=manager(); ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0,(0,'bulk',10))
    result=ob.observe()
    u=result.ues[0]
    assert u.highest_priority==90 and u.gbr_deficit_bytes==0
    q.apply_service(service_bytes_by_flow={})


def test_observer_remains_consistent_on_queue_overflow_drops():
    specs=(QoSFlowSpec(0,'critical',7,10,pdb_ttis=2,gbr_bps=80_000,max_queue_bytes=20),)
    q=manager(specs=specs,num_ues=1)
    ob=ServiceAwareQoSObserver(queues=q)
    begin(q,0,(0,'critical',25),(0,'critical',17))
    f=ob.observe().flows[0]
    assert (f.backlog_bytes,f.queued_packets,f.gbr_deficit_bytes)==(17,1,5)
    q.apply_service(service_bytes_by_flow={})


def test_repeated_read_does_not_double_count_prior_delivery():
    q=manager(); ob=ServiceAwareQoSObserver(queues=q,gbr_window_ttis=4)
    begin(q,0,(0,'urgent',100)); ob.observe(); q.apply_service(service_bytes_by_flow={(0,'urgent'):3})
    begin(q,1)
    s=ob.observe()
    assert views(s)[0,'urgent'].gbr_window_delivered_bytes==3
    with pytest.raises(RuntimeError): ob.observe()
    assert views(s)[0,'urgent'].gbr_window_delivered_bytes==3
    q.apply_service(service_bytes_by_flow={})
    begin(q,2)
    assert views(ob.observe())[0,'urgent'].gbr_window_delivered_bytes==3
    q.apply_service(service_bytes_by_flow={})


def test_gbr_shortfall_is_capped_and_idle_slot_reclaims_window():
    q=manager()
    ob=ServiceAwareQoSObserver(queues=q,gbr_window_ttis=2)
    begin(q,0,(0,'urgent',3)); assert views(ob.observe())[0,'urgent'].gbr_deficit_bytes==3
    q.apply_service(service_bytes_by_flow={(0,'urgent'):3})
    begin(q,1); assert views(ob.observe())[0,'urgent'].gbr_deficit_bytes==0
    q.apply_service(service_bytes_by_flow={})
    begin(q,2); assert views(ob.observe())[0,'urgent'].gbr_window_backlogged_ttis==0
    q.apply_service(service_bytes_by_flow={})
    begin(q,3,(0,'urgent',100)); assert views(ob.observe())[0,'urgent'].gbr_deficit_bytes==5
    q.apply_service(service_bytes_by_flow={})


def test_active_tti_and_configured_specs_accessors():
    q=manager()
    assert q.active_tti_index is None
    assert [s.key for s in q.configured_flows()]==[(0,'urgent'),(0,'bulk'),(1,'video')]
    begin(q,0)
    assert q.active_tti_index==0
    q.apply_service(service_bytes_by_flow={})
    assert q.active_tti_index is None


def test_observe_precedes_attempted_transport_blocks_not_after():
    q=manager(); obs=ServiceAwareQoSObserver(queues=q)
    tb=ServiceAwareTransmissionEngine(seed=4)
    begin(q,0,(0,'urgent',30))
    before=obs.observe()
    tb.execute_tti(queues=q,transport_blocks=[DownlinkTB(0,30,0)], planned_flow_bytes={(0,'urgent'):30})
    assert before.flows[0].backlog_bytes==30
    begin(q,1)
    after=obs.observe()
    assert after.flows[0].backlog_bytes==0
    assert after.flows[0].gbr_window_delivered_bytes==30
    assert after.flows[0].gbr_deficit_bytes==0
    q.apply_service(service_bytes_by_flow={})


def test_gbr_history_storage_is_bounded_after_long_episode():
    specs=(QoSFlowSpec(0,'gbr',7,20,gbr_bps=80_000),)
    q=manager(specs=specs,num_ues=1)
    ob=ServiceAwareQoSObserver(queues=q,gbr_window_ttis=5)
    for t in range(150):
        begin(q,t,*([(0,'gbr',1)] if t % 2 == 0 else []))
        ob.observe()
        q.apply_service(service_bytes_by_flow={(0,'gbr'):1})
    hist=ob._gbr_histories[0,'gbr']
    assert len(hist.completed)==4
    assert 0<=hist.backlogged_slots<=4


def test_configured_flow_metadata_loaded_once(monkeypatch):
    q=manager()
    real=q.configured_flows
    called=[]
    def wrapped():
        called.append(True)
        return real()
    monkeypatch.setattr(q,'configured_flows',wrapped)
    obs=ServiceAwareQoSObserver(queues=q)
    for t in range(3):
        begin(q,t)
        obs.observe()
        q.apply_service(service_bytes_by_flow={})
    assert len(called)==1
