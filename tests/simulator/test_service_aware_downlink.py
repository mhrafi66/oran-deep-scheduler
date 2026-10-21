"""Task-5 integrated service accounting, regression guards and metrics."""
import math

import pytest
import torch

from oran_scheduler.schedulers.allocation import CellAllocation
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
from oran_scheduler.simulator.service_aware_transmission import ServiceAwareTransmissionEngine


def make_runner(
    *, flows=None, tbler_by_tti=None, capacity_bits=2048, link=None, policy=None,
    flow_packer=None, num_ues=1, max_backlog=None, bounded_packing=False,
):
    if flows is None:
        flows=[QoSFlowSpec(0,'critical',1,10,pdb_ttis=2,max_queue_bytes=max_backlog),
               QoSFlowSpec(0,'ftp',9,50,gbr_bps=8000.)]
    queues=ServiceAwareTrafficManager(num_ues=num_ues,flows=flows,tti_duration_s=0.0005)
    observer=ServiceAwareQoSObserver(queues=queues,gbr_window_ttis=3)
    dec=ServiceAwareTransmissionEngine(seed=2)
    grants=NRGrantAssembler(
        bandwidth=NRBandwidthPart(3,1),region=PDSCHDataRegion(),
        channel_num_prbs=3,tbs_backend=lambda specs: [capacity_bits]*len(specs),
    )
    allocation=CellAllocation(torch.tensor([[0,0,0]],dtype=torch.long))
    if policy is None:
        def policy(obs):
            return DownlinkSchedulingDecision(allocation,(0,),
                None if bounded_packing else {(f.ue_index,f.flow_id):f.backlog_bytes for f in obs.flows})
    if link is None:
        def link(obs,decision):
            p=(tbler_by_tti or {}).get(obs.tti_index,0.)
            return {0:WholeGrantLink(0,(0,1,2),20,p,1)}
    if bounded_packing:
        def flow_packer(obs,decision,blocks):
            budget={b.ue_index:b.payload_capacity_bytes for b in blocks}
            packed={}
            for f in obs.flows:
                amount=min(budget.get(f.ue_index,0),f.backlog_bytes)
                if amount:
                    packed[f.key]=amount
                    budget[f.ue_index]-=amount
            return packed
    return ServiceAwareDownlinkCellRunner(
        queues=queues,observer=observer,decoder=dec,grants=grants,
        policy=policy,link_provider=link,flow_packer=flow_packer,
    )


def test_decode_failure_retains_queue_and_later_success():
    r=make_runner(tbler_by_tti={0:1.,1:0.})
    first=r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',10),))
    assert first.queue_start.arrived_bytes==10
    assert first.pre_service_observation.total_backlog_bytes==10
    assert first.transmission.decoded_capacity_bytes==0
    assert first.backlog_after_bytes==10
    second=r.step(tti_index=1,arrivals=())
    assert second.transmission.traffic_service.delivered_bytes==10
    report=r.report()
    assert report.arrived_bytes==10 and report.delivered_bytes==10
    assert report.final_backlog_bytes==0
    assert report.flows[0].completed_delay_p50_slots==2
    assert report.flows[0].on_time_completed_packets==1
    assert report.on_time_ratio_of_arrived_packets==1.
    assert report.failed_capacity_bytes==256
    assert report.decoded_capacity_utilization==10/256


def test_empty_tti_and_zero_arrival_ratios_are_undefined():
    r=make_runner()
    r.step(tti_index=0,arrivals=())
    s=r.report()
    assert s.measured_ttis==1 and s.arrived_bytes==0
    assert s.jain_fairness_offered_ues is None
    assert s.on_time_ratio_of_arrived_packets is None
    assert all(x.completed_delay_p95_slots is None for x in s.flows)


def test_deadline_expiry_one_tti_after_final_opportunity():
    r=make_runner(tbler_by_tti={0:1.,1:1.})
    for idx,arrivals in enumerate(((FlowPacketArrival(0,'critical',12),),(),())):
        r.step(tti_index=idx,arrivals=arrivals)
    s=r.report()
    flow=s.flows[0]
    assert flow.deadline_misses==1
    assert flow.expired_dropped_bytes==12 and flow.expired_dropped_packets==1
    assert flow.queued_packets==0
    assert flow.on_time_ratio_of_arrivals==0.


def test_lookahead_is_forbidden_via_pre_service_observation():
    seen=[]
    def policy(obs):
        seen.append((obs.tti_index,obs.total_backlog_bytes,obs.flows[0].hol_delay_ttis))
        a=CellAllocation(torch.tensor([[0,0,0]],dtype=torch.long))
        return DownlinkSchedulingDecision(a,(0,),{(0,'critical'):obs.flows[0].backlog_bytes})
    r=make_runner(policy=policy,tbler_by_tti={0:1.})
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',30),))
    r.step(tti_index=1,arrivals=())
    assert seen==[(0,30,1),(1,30,2)]


def test_cannot_advance_before_correct_slot_or_duplicate_slot():
    r=make_runner()
    with pytest.raises(ValueError,match='Expected contiguous'):
        r.step(tti_index=1,arrivals=())
    r.step(tti_index=0,arrivals=())
    with pytest.raises(ValueError,match='Expected contiguous'):
        r.step(tti_index=0,arrivals=())


def test_invalid_link_fails_closed_and_reset_replays():
    def bad_link(obs,d): return {0:WholeGrantLink(0,(1,2),20,0.)}
    r=make_runner(link=bad_link)
    with pytest.raises(ValueError,match='stale or from a different'):
        r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',8),))
    assert r.queues.has_active_tti
    r.reset(seed=2)
    assert not r.queues.has_active_tti
    r.link_provider=lambda obs,d:{0:WholeGrantLink(0,(0,1,2),20,0.)}
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',8),))
    assert r.report().delivered_bytes==8


def test_reject_overcommitted_tb_before_decode_or_queue_mutation():
    r=make_runner(capacity_bits=8*10)
    with pytest.raises(ValueError,match='Planned flow bytes exceed TB capacity'):
        r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',12),))
    assert r.queues.backlog_bytes==12
    r.reset()
    assert r.report().measured_ttis==0


def test_flow_packer_receives_physical_tbs_and_multiplexes():
    invoked=[]
    def policy(obs):
        a=CellAllocation(torch.tensor([[0,0,0]],dtype=torch.long))
        return DownlinkSchedulingDecision(a,(0,),None)
    def pack(obs,d,tbs):
        invoked.append(tbs[0].payload_capacity_bytes)
        return {(0,'critical'):min(tbs[0].payload_capacity_bytes,obs.flows[0].backlog_bytes)}
    r=make_runner(policy=policy,flow_packer=pack,capacity_bits=8*6)
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',10),))
    assert invoked==[6]
    assert r.report().delivered_bytes==6 and r.queues.backlog_bytes==4


def test_requires_flow_packing_when_missing():
    def policy(obs):
        return DownlinkSchedulingDecision(CellAllocation(torch.tensor([[0,0,0]])),(0,),None)
    r=make_runner(policy=policy)
    with pytest.raises(ValueError,match='flow_packer'):
        r.step(tti_index=0,arrivals=())


def test_packet_delay_nearest_rank_quantiles_without_packet_vectors():
    # 4 packets completed with delays [1,1,2,3].
    r=make_runner(capacity_bits=8*10)
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'ftp',2),))
    r.step(tti_index=1,arrivals=(FlowPacketArrival(0,'ftp',2),FlowPacketArrival(0,'ftp',2)))
    r.step(tti_index=2,arrivals=(FlowPacketArrival(0,'ftp',2),))
    report=r.report().flows[1]
    assert report.completed_packets==4
    assert report.completed_delay_p50_slots==1
    assert report.completed_delay_p95_slots==1


def test_packet_delay_histograms_cover_partial_service():
    r=make_runner(capacity_bits=8*3,bounded_packing=True)
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'ftp',8),))
    r.step(tti_index=1,arrivals=())
    r.step(tti_index=2,arrivals=())
    report=r.report().flows[1]
    assert report.completed_delay_p50_slots==3
    assert report.delivered_bytes==8
    assert report.completed_packets==1


def test_retained_deadline_packet_counts_late_completion():
    r=make_runner(flows=[QoSFlowSpec(0,'critical',1,10,pdb_ttis=1,expiry_policy='retain')],
                  tbler_by_tti={0:1.})
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',10),))
    r.step(tti_index=1,arrivals=())
    x=r.report().flows[0]
    assert x.deadline_misses==1 and x.late_completed_packets==1
    assert x.completed_delay_p95_slots==2
    assert x.on_time_ratio_of_arrivals==0.


def test_bounded_buffer_overflow_is_tracked():
    r=make_runner(max_backlog=5,tbler_by_tti={0:1.})
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',6),FlowPacketArrival(0,'critical',4)))
    x=r.report().flows[0]
    assert x.arrived_bytes==10 and x.overflow_dropped_bytes==6
    assert x.backlog_bytes==4 and x.admitted_packets==1


def test_bounded_gbr_backlog_gate_only():
    r=make_runner()
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'ftp',10),))
    r.step(tti_index=1,arrivals=())
    x=r.report().flows[1]
    assert x.gbr_backlogged_slots==1
    assert x.backlogged_slot_delivery_to_gbr_ratio==10/(8000*0.0005/8)


def test_reset_restores_exact_packet_latency_statistics():
    r=make_runner(capacity_bits=8*4,bounded_packing=True)
    arrivals=(FlowPacketArrival(0,'ftp',7),)
    for round in range(2):
        r.step(tti_index=0,arrivals=arrivals)
        r.step(tti_index=1,arrivals=())
        result=r.report()
        assert result.flows[1].completed_delay_p50_slots==2
        assert result.arrived_bytes==7
        if round==0: r.reset()


def test_separate_cells_have_no_shared_queue_or_metrics():
    r0=make_runner()
    r1=make_runner()
    r0.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',10),))
    r1.step(tti_index=0,arrivals=())
    assert r0.report().delivered_bytes==10
    assert r1.report().delivered_bytes==0


def test_end_horizon_pending_deadline_is_not_automatically_counted_missed():
    r=make_runner(tbler_by_tti={0:1.})
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',10),))
    x=r.report().flows[0]
    assert x.queued_packets==1 and x.deadline_misses==0
    assert x.on_time_ratio_of_arrivals==0.


def test_conservation_randomized_paired_seed():
    import random
    for seed in range(3):
        rng=random.Random(seed)
        r=make_runner(capacity_bits=8*40,
                      tbler_by_tti={i:rng.choice([0.,1.]) for i in range(80)},bounded_packing=True)
        for t in range(80):
            events=[]
            if rng.random()<0.5: events.append(FlowPacketArrival(0,'critical',rng.randint(1,30)))
            if rng.random()<0.7: events.append(FlowPacketArrival(0,'ftp',rng.randint(1,30)))
            r.step(tti_index=t,arrivals=events)
            s=r.report()
            assert (s.arrived_bytes==s.delivered_bytes+s.overflow_dropped_bytes
                    +s.expired_dropped_bytes+s.final_backlog_bytes)


def test_no_silent_multi_rbg_bler_from_old_per_rbg_outputs():
    r=make_runner(link=lambda obs,d:{})
    with pytest.raises(ValueError,match='match exactly'):
        r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',5),))


def test_sionna_whole_grant_capture_smoke():
    pytest.importorskip('sionna')
    from oran_scheduler.simulator.service_aware_whole_grant_phy import FinalAllocationSINRCapture
    from oran_scheduler.phy.link_adaptation import LinkAdaptationConfig
    capture=FinalAllocationSINRCapture()
    selected=torch.tensor([0,1],dtype=torch.long)
    ranks=torch.tensor([1,1],dtype=torch.long)
    for rbg in (0,1):
        capture.capture_rbg(
            rbg_index=rbg,
            selected_candidate_indices=selected,
            selected_ranks=ranks,
            layer_sinr_linear=torch.ones(1,12,2,dtype=torch.float32)*5.0,
            layer_ue_indices=torch.tensor([0,1]),
            layer_index_within_ue=torch.tensor([0,0]),
        )
    actions=CellAllocation(torch.tensor([[0,0],[1,1]],dtype=torch.long))
    config=LinkAdaptationConfig(device='cpu',num_rbgs=2)
    links=capture.whole_links(allocation=actions,config=config)
    assert len(links)==2
    assert all(set(l.assigned_rbgs)=={0,1} and 0<=l.tbler<=1 for l in links.values())
    assert all(0<=l.mcs_index<=27 for l in links.values())
    assert capture._samples=={}


def test_capture_real_rbg_layers_without_recomputing_sinr():
    from oran_scheduler.simulator.service_aware_whole_grant_phy import FinalAllocationSINRCapture
    capture=FinalAllocationSINRCapture()
    channel_sinr=torch.ones(1,12,3)
    capture.capture_rbg(
        rbg_index=0,selected_candidate_indices=torch.tensor([7,4]),
        selected_ranks=torch.tensor([2,1]),layer_sinr_linear=channel_sinr,
        layer_ue_indices=torch.tensor([0,1,0]),
        layer_index_within_ue=torch.tensor([0,0,1]),
    )
    assert capture._rank=={7:2,4:1}
    assert capture._samples[7][0].shape==(1,12,2)
    assert capture._samples[4][0].shape==(1,12,1)
    assert capture._samples[7][0].data_ptr()!=channel_sinr.data_ptr()


def test_capture_rejects_repeated_rbg():
    from oran_scheduler.simulator.service_aware_whole_grant_phy import FinalAllocationSINRCapture
    capture=FinalAllocationSINRCapture()
    args=dict(
        rbg_index=0,selected_candidate_indices=torch.tensor([0]),
        selected_ranks=torch.tensor([1]),layer_sinr_linear=torch.ones(1,12,1),
        layer_ue_indices=torch.tensor([0]),layer_index_within_ue=torch.tensor([0]),
    )
    capture.capture_rbg(**args)
    with pytest.raises(ValueError,match='captured twice'):
        capture.capture_rbg(**args)
    capture.clear()
    assert capture._samples=={}


def test_capture_rejects_missing_or_duplicate_layer():
    from oran_scheduler.simulator.service_aware_whole_grant_phy import FinalAllocationSINRCapture
    capture=FinalAllocationSINRCapture()
    with pytest.raises(ValueError,match='Missing scheduled UE'):
        capture.capture_rbg(
            rbg_index=0,selected_candidate_indices=torch.tensor([0]),
            selected_ranks=torch.tensor([2]),layer_sinr_linear=torch.ones(1,12,2),
            layer_ue_indices=torch.tensor([0,1]),layer_index_within_ue=torch.tensor([0,1]),
        )


def test_failure_does_not_degrade_existing_queue_reset_default():
    r=make_runner(link=lambda o,d:{})
    with pytest.raises(ValueError):
        r.step(tti_index=0,arrivals=())
    with pytest.raises(RuntimeError,match='middle of a TTI'):
        r.queues.reset()
    r.reset()
    assert r.report().measured_ttis==0


def test_empty_initial_report_is_consistent():
    r=make_runner()
    s=r.report()
    assert s.measured_ttis==0 and s.duration_s==0
    assert s.arrived_bytes==s.delivered_bytes==0
    assert s.on_time_ratio_of_arrived_packets is None


def test_packet_completion_delay_histogram_is_exact_and_resettable():
    r=make_runner(capacity_bits=8*4,bounded_packing=True)
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'ftp',6),))
    r.step(tti_index=1,arrivals=())
    assert r.queues.completed_delay_histogram(0,'ftp')==((2,1),)
    with pytest.raises(ValueError,match='Unknown flow'):
        r.queues.completed_delay_histogram(0,'notthere')
    r.reset()
    assert r.queues.completed_delay_histogram(0,'ftp')==()


def test_fairness_only_counts_ues_with_offered_traffic():
    r=make_runner(num_ues=2)
    r.step(tti_index=0,arrivals=(FlowPacketArrival(0,'critical',10),))
    assert r.report().jain_fairness_offered_ues==1.0
