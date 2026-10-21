"""Task 7: causal report delay, actual-PHY separation and queue integration."""
from types import SimpleNamespace

import pytest
import torch

from oran_scheduler.schedulers.allocation import CellAllocation
from oran_scheduler.simulator.service_aware_downlink import (
    DownlinkSchedulingDecision, ServiceAwareDownlinkCellRunner,
)
from oran_scheduler.simulator.service_aware_nr_grants import (
    NRBandwidthPart, NRGrantAssembler, PDSCHDataRegion, WholeGrantLink,
)
from oran_scheduler.simulator.service_aware_observation import (
    QoSCellObservation, ServiceAwareQoSObserver,
)
from oran_scheduler.simulator.service_aware_temporal_csi import (
    CSIReport, ServiceAwareTemporalCSIAdapter, TemporalWindowFrameSource,
)
from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival, QoSFlowSpec, ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_transmission import ServiceAwareTransmissionEngine


def frame(t):
    return torch.full((1,1,1,1,1,1,12), complex(t + 1, 0.0), dtype=torch.complex64)


def observation(t):
    return QoSCellObservation(t, 0.0005, (), (), 0)


def mk_adapter(*, delay=0, period=1, builder=None, source=None):
    traces = {'reports': [], 'policy': [], 'physical': [], 'frames': []}
    allocation = CellAllocation(torch.tensor([[0,0,0]], dtype=torch.long))

    def source_fn(t):
        traces['frames'].append(t)
        return frame(t)

    def build(h, t):
        traces['reports'].append(t)
        return CSIReport(t, (17,), int(h.real[0,0,0,0,0,0,0].item()))

    def select(o, published):
        traces['policy'].append((o.tti_index, published.report_tti,
                                 published.age_ttis, published.report.payload,
                                 published.bootstrap))
        return DownlinkSchedulingDecision(allocation, (0,), {(0, 'f'): 5})

    def actual(o, decision, h, published):
        traces['physical'].append((o.tti_index, int(h.real[0,0,0,0,0,0,0].item()),
                                   published.report_tti))
        return {0: WholeGrantLink(0, (0,1,2), 20, 0.0)}

    adapter = ServiceAwareTemporalCSIAdapter(
        frame_source=source or source_fn, report_builder=builder or build,
        policy=select, physical_link_provider=actual,
        delay_ttis=delay, report_period_ttis=period,
    )
    return adapter, traces


def execute(adapter, t):
    o=observation(t)
    d=adapter.policy(o)
    return adapter.link_provider(o,d)


def test_zero_delay_current_report_and_current_phy():
    a,log=mk_adapter()
    for t in range(4):
        execute(a,t)
    assert [x[:4] for x in log['policy']]==[(t,t,0,t+1) for t in range(4)]
    assert log['physical']==[(t,t+1,t) for t in range(4)]
    assert a.frame_fetch_count==4 and a.report_build_count==4
    assert a.retained_reports <= 2


def test_two_tti_age_and_bootstrap_are_causal():
    a,log=mk_adapter(delay=2)
    for t in range(7):
        execute(a,t)
    assert [x[1] for x in log['policy']]==[0,0,0,1,2,3,4]
    assert [x[2] for x in log['policy']]==[0,1,2,2,2,2,2]
    assert [x[4] for x in log['policy']]==[True,True,False,False,False,False,False]
    assert [x[1] for x in log['physical']]==[1,2,3,4,5,6,7]
    assert a.retained_reports <= 4


def test_periodic_csi_update_is_computed_only_when_needed():
    a,log=mk_adapter(delay=2,period=3)
    for t in range(12):
        execute(a,t)
    assert log['reports']==[0,3,6,9]
    assert [x[1] for x in log['policy']]==[0,0,0,0,0,3,3,3,6,6,6,9]
    assert a.frame_fetch_count==12 and a.report_build_count==4
    assert a.retained_reports <= 3


def test_requires_policy_then_link_and_same_decision_identity():
    a,_=mk_adapter()
    with pytest.raises(RuntimeError,match='without preceding'):
        a.link_provider(observation(0), DownlinkSchedulingDecision(
            CellAllocation(torch.tensor([[0,0,0]])),(0,),{}))
    d=a.policy(observation(0))
    with pytest.raises(RuntimeError,match='Previous'):
        a.policy(observation(1))
    with pytest.raises(RuntimeError,match='Stale or mismatched'):
        a.link_provider(observation(0),DownlinkSchedulingDecision(d.allocation,d.queue_ue_by_candidate, d.planned_flow_bytes))
    with pytest.raises(RuntimeError,match='Stale or mismatched'):
        a.link_provider(observation(1),d)
    a.link_provider(observation(0),d)
    with pytest.raises(RuntimeError,match='without preceding'):
        a.link_provider(observation(0),d)


def test_rejects_skipping_or_replaying_ttis():
    a,_=mk_adapter()
    with pytest.raises(ValueError,match='contiguous'):
        a.policy(observation(1))
    execute(a,0)
    with pytest.raises(ValueError,match='contiguous'):
        a.policy(observation(0))
    with pytest.raises(ValueError,match='contiguous'):
        a.policy(observation(3))


@pytest.mark.parametrize('delay,period', [(-1,1),(True,1),(0,0),(0,True),(1,1.5)])
def test_reject_invalid_schedule_configuration(delay,period):
    with pytest.raises(ValueError):
        mk_adapter(delay=delay,period=period)


def test_membership_changes_refused_before_using_stale_csi():
    def build(h,t): return CSIReport(t,(4,) if t==0 else (5,),float(t))
    a,_=mk_adapter(delay=2,builder=build)
    execute(a,0)
    with pytest.raises(RuntimeError,match='membership'):
        a.policy(observation(1))


def test_malformed_report_and_physical_channel_fail_closed():
    for bad in [lambda h,t: CSIReport(t+1,(0,),t),
                lambda h,t: CSIReport(t,(),t),
                lambda h,t: CSIReport(t,(0,0),t)]:
        a,_=mk_adapter(builder=bad)
        with pytest.raises(ValueError):
            a.policy(observation(0))
    for bad_frame in [lambda t: torch.zeros((1,1,1)),
                      lambda t: torch.zeros((1,1,1,1,1,1,12))]:
        a,_=mk_adapter(source=bad_frame)
        with pytest.raises(ValueError):
            a.policy(observation(0))


def test_physical_layout_changed_fail_closed():
    def source(t):
        return frame(0) if t == 0 else torch.ones((1,1,1,1,1,1,24),dtype=torch.complex64)
    a,_=mk_adapter(source=source)
    execute(a,0)
    with pytest.raises(RuntimeError,match='dimensions/device/dtype'):
        a.policy(observation(1))


def test_no_autograd_graph_retained_for_frames_or_reports():
    def source(t):
        return torch.complex(torch.ones((1,1,1,1,1,1,12),requires_grad=True),
                             torch.zeros((1,1,1,1,1,1,12)))
    def build(h,t):
        assert not torch.is_grad_enabled()
        return CSIReport(t,(0,),h.real.sum())
    a,logs=mk_adapter(builder=build,source=source)
    execute(a,0)
    assert not a._history[0].payload.requires_grad
    assert a._physical_frame is None


def test_reset_replay_and_bounded_history():
    a,log=mk_adapter(delay=5)
    for t in range(80):
        execute(a,t)
        assert a.retained_reports <= 7
    a.reset()
    assert (a.frame_fetch_count,a.report_build_count,a.retained_reports)==(0,0,0)
    execute(a,0)
    assert log['policy'][-1][1:4]==(0,0,1)


def test_temporal_window_source_is_no_copy_and_range_checked():
    win=SimpleNamespace(num_ttis=3,tti_slice=lambda t: window_tensor[...,t:t+1,:])
    window_tensor=torch.ones((1,1,1,1,1,3,12),dtype=torch.complex64)
    source=TemporalWindowFrameSource(win)
    assert source(2).data_ptr()==window_tensor[...,2:3,:].data_ptr()
    with pytest.raises(ValueError,match='outside'):
        source(3)


def test_runner_integration_causal_qos_and_current_phy():
    adapter, traces = mk_adapter(delay=1)
    # Recreate with a link that fails using ACTUAL channel index 0 and succeeds
    # in index 1; stale report MUST NOT decide the PHY decode probability.
    def actual(o, d, h, report):
        current=int(h.real[0,0,0,0,0,0,0].item())
        return {0:WholeGrantLink(0,(0,1,2),20,1.0 if current==1 else 0.0)}
    observed_phy=[]
    def actual_with_record(o,d,h,report):
        observed_phy.append((o.tti_index,int(h.real[0,0,0,0,0,0,0].item())))
        return actual(o,d,h,report)
    adapter._link=actual_with_record
    queues=ServiceAwareTrafficManager(num_ues=1,tti_duration_s=0.0005,
                                      flows=(QoSFlowSpec(0,'f',1,10,pdb_ttis=4),))
    observer=ServiceAwareQoSObserver(queues=queues)
    decoder=ServiceAwareTransmissionEngine(seed=3)
    grants=NRGrantAssembler(bandwidth=NRBandwidthPart(3,1),
                             region=PDSCHDataRegion(),channel_num_prbs=3,
                             tbs_backend=lambda specs:[8*100]*len(specs))
    runner=ServiceAwareDownlinkCellRunner(
        queues=queues,observer=observer,decoder=decoder,grants=grants,
        policy=adapter.policy,link_provider=adapter.link_provider,
    )
    r0=runner.step(tti_index=0,arrivals=(FlowPacketArrival(0,'f',5),))
    r1=runner.step(tti_index=1,arrivals=())
    assert r0.backlog_after_bytes==5
    assert r1.backlog_after_bytes==0
    assert runner.report().delivered_bytes==5
    assert traces['policy'][1][1]==0
    assert observed_phy==[(0,1),(1,2)]


def test_real_sionna_window_source_smoke_if_installed():
    pytest.importorskip('sionna')
    from oran_scheduler.simulator.temporal_channel import TemporalChannelWindow
    h=torch.ones((1,1,1,1,1,3,12),dtype=torch.complex64)
    w=TemporalChannelWindow(h,seed=4,tti_duration_s=0.0005)
    a,_=mk_adapter(source=TemporalWindowFrameSource(w), delay=1)
    for t in range(3):
        execute(a,t)
    assert a.frame_fetch_count==3
