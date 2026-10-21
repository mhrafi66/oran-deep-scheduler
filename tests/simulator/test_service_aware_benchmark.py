"""CPU-compatible tests for the diagnostic GPU benchmark wiring.

The real multi-cell Sionna GPU workload must be run in CHPC Slurm; these tests
exercise decisions, flow packing, timing helper, and Task-5 integration with
explicit synthetic whole-TB results (not a replacement for PHY validation).
"""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    'service_aware_runtime_benchmark',
    _ROOT / 'scripts' / 'benchmark_service_aware_runtime.py',
)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

from oran_scheduler.simulator.service_aware_downlink import ServiceAwareDownlinkCellRunner
from oran_scheduler.simulator.service_aware_nr_grants import NRBandwidthPart, NRGrantAssembler, PDSCHDataRegion, WholeGrantLink
from oran_scheduler.simulator.service_aware_traffic import QoSFlowSpec, ServiceAwareTrafficManager
from oran_scheduler.simulator.service_aware_observation import ServiceAwareQoSObserver
from oran_scheduler.simulator.service_aware_transmission import ServiceAwareTransmissionEngine


def _view(*backlogs):
    return SimpleNamespace(ues=[SimpleNamespace(backlog_bytes=n) for n in backlogs])


def test_policy_deallocates_empty_queue():
    policy = _MOD.DiagnosticPolicy(rbg_score=torch.ones(3, 4), user_slots=2)
    result = policy(_view(0, 0, 0))
    assert result.allocation.candidate_by_user_slot.tolist() == [[-1] * 4] * 2
    assert result.queue_ue_by_candidate == (0, 1, 2)


def test_policy_dynamic_queue_and_radio_quality():
    policy = _MOD.DiagnosticPolicy(
        rbg_score=torch.tensor([[9., 1., 9.], [1., 9., 1.]]), user_slots=1,
    )
    assert policy(_view(20, 20)).allocation.candidate_by_user_slot.tolist() == [[0, 1, 0]]
    assert policy(_view(0, 20)).allocation.candidate_by_user_slot.tolist() == [[1, 1, 1]]
    policy.update_scores(torch.tensor([[1., 8., 1.], [9., 1., 9.]]))
    assert policy(_view(20, 20)).allocation.candidate_by_user_slot.tolist() == [[1, 0, 1]]


def test_two_user_slots_unique_or_padded():
    policy = _MOD.DiagnosticPolicy(rbg_score=torch.tensor([[4., 1.], [2., 3.]]), user_slots=2)
    scheduled = policy(_view(30, 30)).allocation.candidate_by_user_slot
    assert scheduled.shape == (2, 2)
    for col in range(2):
        assert sorted(scheduled[:, col].tolist()) == [0, 1]
    idle_one = policy(_view(0, 30)).allocation.candidate_by_user_slot
    assert idle_one[0].tolist() == [1, 1]
    assert idle_one[1].tolist() == [-1, -1]


def test_bad_policy_config_or_score_refresh():
    with pytest.raises(ValueError):
        _MOD.DiagnosticPolicy(rbg_score=torch.ones(0, 2), user_slots=1)
    with pytest.raises(ValueError):
        _MOD.DiagnosticPolicy(rbg_score=torch.ones(2, 2), user_slots=3)
    policy = _MOD.DiagnosticPolicy(rbg_score=torch.ones(2, 2), user_slots=1)
    with pytest.raises(ValueError):
        policy.update_scores(torch.ones(2, 3))


def test_arrivals_and_post_tbs_packing():
    arrivals = _MOD.deterministic_arrivals(0, 2)
    assert [(a.ue_index, a.flow_id, a.size_bytes) for a in arrivals] == [
        (0, 'control', 256), (0, 'bulk', 1400), (1, 'bulk', 1400),
    ]
    flows = [
        SimpleNamespace(ue_index=0, flow_id='bulk', key=(0, 'bulk'), priority=50, backlog_bytes=100),
        SimpleNamespace(ue_index=0, flow_id='control', key=(0, 'control'), priority=10, backlog_bytes=80),
    ]
    planned = _MOD.pack_by_priority(SimpleNamespace(flows=flows), None, [
        SimpleNamespace(ue_index=0, payload_capacity_bytes=130),
    ])
    assert planned == {(0, 'control'): 80, (0, 'bulk'): 50}


def test_stage_recorder_cpu_profile_and_helper():
    record = _MOD.StageRecorder('cpu', profile=True)
    obj = SimpleNamespace(f=lambda x: x + 2)
    record.wrap_method(obj, 'f', 'addition')
    assert obj.f(1) == 3
    assert 'addition' not in record.samples
    record.enabled = True
    assert obj.f(8) == 10
    assert record.samples['addition'][0] >= 0
    assert _MOD.summarize([])['p95_ms'] is None
    assert _MOD.summarize([1., 2., 3.])['p95_ms'] == 3


def test_two_tti_task5_runner_with_diagnostic_scheduling():
    queues = ServiceAwareTrafficManager(
        num_ues=2, tti_duration_s=0.0005,
        flows=(QoSFlowSpec(0,'control',1,10,pdb_ttis=5),
               QoSFlowSpec(0,'bulk',9,50),
               QoSFlowSpec(1,'control',1,10,pdb_ttis=5),
               QoSFlowSpec(1,'bulk',9,50)),
    )
    policy = _MOD.DiagnosticPolicy(
        rbg_score=torch.tensor([[2., 8., 2.],[8., 2., 8.]]),user_slots=1,
    )
    observer = ServiceAwareQoSObserver(queues=queues)
    decoder = ServiceAwareTransmissionEngine(seed=1)
    grants = NRGrantAssembler(
        bandwidth=NRBandwidthPart(3,1),
        region=PDSCHDataRegion(num_pdsch_symbols=14,dmrs_re_per_prb=12),
        channel_num_prbs=3, tbs_backend=lambda specs: [2048]*len(specs),
    )
    def link_source(view, decision):
        entries = decision.allocation.candidate_by_user_slot.tolist()[0]
        result = {}
        for u in range(2):
            used = tuple(i for i, candidate in enumerate(entries) if candidate == u)
            if used:
                result[u] = WholeGrantLink(u, used, mcs_index=10, tbler=0.0)
        return result
    runner = ServiceAwareDownlinkCellRunner(
        queues=queues,observer=observer,decoder=decoder,grants=grants,
        policy=policy,link_provider=link_source,flow_packer=_MOD.pack_by_priority,
    )
    runner.step(tti_index=0, arrivals=_MOD.deterministic_arrivals(0,2))
    runner.step(tti_index=1, arrivals=_MOD.deterministic_arrivals(1,2))
    report = runner.report()
    assert report.measured_ttis == 2
    assert report.delivered_bytes > 0
    assert report.arrived_bytes == (report.delivered_bytes + report.overflow_dropped_bytes
                                    + report.expired_dropped_bytes + report.final_backlog_bytes)


def test_validate_options_rejects_no_cuda_without_escape():
    args=SimpleNamespace(ttis=1,warmup=0,ut_per_sector=1,candidates=1,user_slots=1,
                         refresh_every=0,log_every=0,device='cuda:0',allow_cpu=False)
    with pytest.raises(ValueError,match='Require ttis'):
        _MOD.validate_options(args)
    args.ttis=2
    args.device='cpu'
    with pytest.raises(ValueError,match='GPU-first'):
        _MOD.validate_options(args)
    args.allow_cpu=True
    _MOD.validate_options(args)
