"""Task 3 strict whole-grant geometry, capacity, and integration tests."""

from dataclasses import replace
import math

import pytest
import torch

from oran_scheduler.schedulers.allocation import CellAllocation
from oran_scheduler.simulator.service_aware_nr_grants import (
    NRBandwidthPart,
    PDSCHDataRegion,
    WholeGrantLink,
    NRGrantAssembler,
    sionna_nr_tb_size_bits,
)
from oran_scheduler.simulator.service_aware_traffic import (
    QoSFlowSpec, FlowPacketArrival, ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_transmission import ServiceAwareTransmissionEngine


class FakeSizer:
    def __init__(self):
        self.calls = []

    def __call__(self, specs):
        self.calls.append(tuple(specs))
        # Plausible byte-aligned but intentionally NOT NR-calibrated fake;
        # used ONLY for testing the adapter, geometry, caching and integration.
        return [8 * max(1, int(re * layers // (16 + mcs))) for mcs, re, layers in specs]


def assembler(num_prbs=18, rbg_size=1, **kwargs):
    fake = FakeSizer()
    a = NRGrantAssembler(
        bandwidth=NRBandwidthPart(num_prbs, rbg_size),
        region=PDSCHDataRegion(**kwargs),
        channel_num_prbs=num_prbs,
        tbs_backend=fake,
    )
    return a, fake


def allocation(*slots):
    return CellAllocation(torch.tensor(slots, dtype=torch.long))


def link(candidate=0, rbgs=(0,), mcs=10, tbler=0.1, layers=1):
    return WholeGrantLink(candidate, rbgs, mcs, tbler, layers)


def test_full_band_273_prbs_partition_17_full_and_one_partial():
    b = NRBandwidthPart(273, 16)
    assert b.num_rbgs == 18
    assert b.prbs_per_rbg == (16,) * 17 + (1,)
    assert b.rbg_prb_spans[0] == (0, 16)
    assert b.rbg_prb_spans[-1] == (272, 273)
    assert sum(b.prbs_per_rbg) == 273
    assert all(b.rbg_prb_spans[i][1] == b.rbg_prb_spans[i+1][0] for i in range(17))


@pytest.mark.parametrize("start", [0, 1, 3, 15, 16, 17, 31])
def test_nonzero_bwp_start_partition_exact_no_double_count(start):
    b = NRBandwidthPart(273, 16, bwp_start_prb=start)
    assert sum(b.prbs_per_rbg) == 273
    assert b.rbg_prb_spans[0][0] == 0
    assert b.rbg_prb_spans[-1][1] == 273
    assert all(b.rbg_prb_spans[i][1] == b.rbg_prb_spans[i+1][0] for i in range(b.num_rbgs - 1))


def test_legacy_reduced_band_18x1_preserved_as_distinct_geometry():
    b = NRBandwidthPart(18, 1)
    assert b.num_rbgs == 18
    assert b.prbs_per_rbg == (1,) * 18


def test_channel_bandwidth_mismatch_guard():
    with pytest.raises(ValueError, match="match"):
        NRGrantAssembler(bandwidth=NRBandwidthPart(273,16), region=PDSCHDataRegion(), channel_num_prbs=18)


@pytest.mark.parametrize("data", [(-1,1,0), (0,1,0), (18,0,0), (18,-1,0), (18,1,-1), (True,1,0)])
def test_invalid_bandwidth_config(data):
    with pytest.raises(ValueError):
        NRBandwidthPart(*data)


def test_data_region_example_has_156_re_per_prb():
    assert PDSCHDataRegion().data_re_per_prb == 156
    assert PDSCHDataRegion(num_pdsch_symbols=12, dmrs_re_per_prb=12).data_re_per_prb == 132
    assert PDSCHDataRegion(num_pdsch_symbols=14, dmrs_re_per_prb=12, x_overhead_re_per_prb=6).data_re_per_prb == 150


@pytest.mark.parametrize("options", [
    {"num_pdsch_symbols": 0}, {"num_pdsch_symbols": 15},
    {"dmrs_re_per_prb": 168}, {"x_overhead_re_per_prb": 5},
    {"higher_layer_header_bytes": -1}, {"mcs_table_index": 1},
    {"max_layers_per_rbg": 0},
])
def test_invalid_resource_region(options):
    with pytest.raises(ValueError):
        PDSCHDataRegion(**options)


def test_nr_multi_rbg_is_one_tb_and_no_per_rbg_rounding():
    a, fake = assembler()
    alloc = allocation([0, 0, 0, -1] + [-1] * 14)
    sized = a.size(a.plan(allocation=alloc, queue_ue_by_candidate=[7], whole_links={0: link(rbgs=(0,1,2))}))
    assert len(sized) == 1
    assert sized[0].plan.num_prbs == 3
    assert sized[0].plan.data_re == 3 * 156
    assert fake.calls == [((10, 3 * 156, 1),)]
    assert sized[0].to_downlink_tb().ue_index == 7


def test_last_partial_rbg_is_one_prb_not_sixteen():
    fake = FakeSizer()
    a = NRGrantAssembler(bandwidth=NRBandwidthPart(273,16),region=PDSCHDataRegion(),channel_num_prbs=273,tbs_backend=fake)
    alloc = allocation([-1]*17 + [0])
    plan = a.plan(allocation=alloc, queue_ue_by_candidate=[5], whole_links={0: link(rbgs=(17,))})
    assert plan[0].num_prbs == 1
    assert plan[0].data_re == 156
    a.size(plan)
    assert fake.calls == [((10,156,1),)]


def test_mu_mimo_two_ues_same_rbg_unique_tb_and_layers():
    a, fake = assembler()
    alloc = allocation([0]+[-1]*17, [1]+[-1]*17)
    grants = a.build(allocation=alloc,queue_ue_by_candidate=[2,3],whole_links={0: link(0,(0,),layers=2),1: link(1,(0,),layers=1,mcs=12,tbler=0.2)})
    assert len(grants) == 2
    assert [t.ue_index for t in grants] == [2,3]
    assert len(fake.calls) == 1
    assert set(fake.calls[0]) == {(10,156,2),(12,156,1)}


def test_rank2_size_uses_layer_count_once():
    a, fake = assembler()
    alloc = allocation([0]+[-1]*17)
    a.build(allocation=alloc,queue_ue_by_candidate=[2],whole_links={0: link(layers=2)})
    assert fake.calls == [((10,156,2),)]


def test_repeat_configuration_uses_cached_sionna_size_without_second_call():
    a, fake = assembler()
    alloc = allocation([0]+[-1]*17)
    for _ in range(10):
        tb = a.build(allocation=alloc, queue_ue_by_candidate=[0], whole_links={0: link()})
        assert len(tb) == 1
    assert len(fake.calls) == 1
    assert a.cache_entries == 1


def test_bounded_cache_handles_more_unique_tbs_than_capacity_in_one_tti():
    fake = FakeSizer()
    a = NRGrantAssembler(bandwidth=NRBandwidthPart(18,1),region=PDSCHDataRegion(),channel_num_prbs=18,tbs_backend=fake,max_cache_entries=1)
    alloc = allocation([0, 1, 2] + [-1]*15)
    grants = a.build(allocation=alloc, queue_ue_by_candidate=[0,1,2], whole_links={0:link(0,(0,),mcs=3),1:link(1,(1,),mcs=4),2:link(2,(2,),mcs=5)})
    assert len(grants) == 3
    assert a.cache_entries == 1
    assert len(fake.calls) == 1


def test_cache_change_in_mcs_or_rbgs_causes_new_calculation():
    a, fake = assembler()
    a.build(allocation=allocation([0]+[-1]*17), queue_ue_by_candidate=[0], whole_links={0:link()})
    a.build(allocation=allocation([0]+[-1]*17), queue_ue_by_candidate=[0], whole_links={0:link(mcs=11)})
    a.build(allocation=allocation([0,0]+[-1]*16), queue_ue_by_candidate=[0], whole_links={0:link(rbgs=(0,1))})
    assert len(fake.calls) == 3


def test_header_overhead_subtracted_once_after_tb_sizing():
    a, fake = assembler(higher_layer_header_bytes=5)
    block = a.build(allocation=allocation([0]+[-1]*17),queue_ue_by_candidate=[1],whole_links={0:link()})[0]
    # Sizer returned 6 bytes; 5 bytes headers => 1 payload byte.
    assert block.payload_capacity_bytes == 1


def test_invalid_header_too_large_rejected():
    a, fake = assembler(higher_layer_header_bytes=100)
    with pytest.raises(ValueError, match="headers"):
        a.build(allocation=allocation([0]+[-1]*17),queue_ue_by_candidate=[0],whole_links={0:link()})


def test_duplicate_ue_on_same_rbg_rejected():
    a, _ = assembler()
    with pytest.raises(ValueError, match="twice"):
        a.plan(allocation=allocation([0]+[-1]*17,[0]+[-1]*17),queue_ue_by_candidate=[0],whole_links={0:link()})


def test_different_candidates_same_queue_ue_rejected():
    a, _ = assembler()
    with pytest.raises(ValueError, match="distinct"):
        a.plan(allocation=allocation([0,1]+[-1]*16),queue_ue_by_candidate=[5,5],whole_links={0:link(0,(0,)),1:link(1,(1,))})


@pytest.mark.parametrize("n,whole_links,match", [
    (1, {}, "exactly"),
    (1, {0:link(rbgs=(1,))}, "stale"),
    (1, {0:link(), 1:link(1,(1,))}, "exactly"),
])
def test_missing_stale_or_extra_whole_tb_link_rejected(n,whole_links,match):
    a,_=assembler()
    with pytest.raises(ValueError,match=match):
        a.plan(allocation=allocation([0]+[-1]*17),queue_ue_by_candidate=[0],whole_links=whole_links)


def test_malformed_or_negative_candidate_rejected():
    a,_=assembler()
    with pytest.raises(ValueError,match="invalid candidate"):
        a.plan(allocation=allocation([1]+[-1]*17),queue_ue_by_candidate=[0],whole_links={})
    with pytest.raises(ValueError,match="real queue UE"):
        a.plan(allocation=allocation([0]+[-1]*17),queue_ue_by_candidate=[-1],whole_links={0:link()})


def test_no_allocation_no_tbs_no_calls():
    a,fake=assembler()
    assert a.build(allocation=allocation([-1]*18),queue_ue_by_candidate=[0],whole_links={}) == ()
    assert fake.calls == []


def test_layer_overflow_rejected():
    a, _ = assembler(max_layers_per_rbg=2)
    with pytest.raises(ValueError,match="layers"):
        a.build(allocation=allocation([0]+[-1]*17,[1]+[-1]*17),queue_ue_by_candidate=[0,1],whole_links={0:link(0,(0,),layers=2),1:link(1,(0,),layers=2)})


@pytest.mark.parametrize("bad", [
    {"tbler": float('nan')},{"tbler": 1.1},{"mcs_index": 28},
    {"num_layers": 3},{"assigned_rbgs": (1,0)}, {"assigned_rbgs": (0,0)},
])
def test_whole_grant_malformed_rejected(bad):
    with pytest.raises(ValueError):
        WholeGrantLink(**{**dict(candidate_index=0,assigned_rbgs=(0,),mcs_index=0,tbler=0.1,num_layers=1),**bad})


def test_backend_failure_is_transactional_cache_unchanged():
    def broken(specs):
        return [7] * len(specs)
    a=NRGrantAssembler(bandwidth=NRBandwidthPart(18,1),region=PDSCHDataRegion(),channel_num_prbs=18,tbs_backend=broken)
    with pytest.raises(ValueError,match="byte-aligned"):
        a.build(allocation=allocation([0]+[-1]*17),queue_ue_by_candidate=[0],whole_links={0:link()})
    assert a.cache_entries == 0


def test_cpu_array_one_batch_and_mapping_can_be_pytorch_tensor():
    a,_=assembler()
    block=a.build(allocation=allocation([0]+[-1]*17),queue_ue_by_candidate=torch.tensor([3],dtype=torch.long),whole_links={0:link()})
    assert block[0].ue_index==3


def test_end_to_end_two_flows_and_failed_tb_preserves_backlog():
    a, fake = assembler()
    queues=ServiceAwareTrafficManager(num_ues=1,tti_duration_s=0.0005,flows=[
        QoSFlowSpec(ue_index=0,flow_id='urgent',five_qi=1,priority=10,pdb_ttis=3),
        QoSFlowSpec(ue_index=0,flow_id='best',five_qi=9,priority=50),
    ])
    decoder=ServiceAwareTransmissionEngine(seed=10)
    blocks=a.build(allocation=allocation([0,0]+[-1]*16),queue_ue_by_candidate=[0],whole_links={0:link(rbgs=(0,1),tbler=1.0)})
    assert len(blocks)==1
    queues.begin_tti(tti_index=0,arrivals=[FlowPacketArrival(0,'urgent',5),FlowPacketArrival(0,'best',5)])
    result=decoder.execute_tti(queues=queues,transport_blocks=blocks,planned_flow_bytes={(0,'urgent'):5,(0,'best'):5})
    assert result.traffic_service.delivered_bytes==0
    assert queues.backlog_bytes==10
    # Reuse same sized TB in next TTI; only link reliability changes.
    blocks=a.build(allocation=allocation([0,0]+[-1]*16),queue_ue_by_candidate=[0],whole_links={0:link(rbgs=(0,1),tbler=0.0)})
    queues.begin_tti(tti_index=1,arrivals=[])
    result=decoder.execute_tti(queues=queues,transport_blocks=blocks,planned_flow_bytes={(0,'urgent'):5,(0,'best'):5})
    assert result.traffic_service.delivered_bytes==10
    assert queues.backlog_bytes==0
    assert len(fake.calls)==1


@pytest.mark.skipif(__import__('importlib').util.find_spec('sionna') is None,reason='Requires Sionna in CHPC conda environment')
def test_real_sionna_tbs_consistency_with_rate_api():
    # Cross-check SAME Sionna NR functions as existing PHY code.
    from sionna.phy.nr.utils import calculate_tb_size, decode_mcs_index
    mcs = torch.tensor([0,10,20],dtype=torch.int32)
    qm,rate=decode_mcs_index(mcs_index=mcs,table_index=2,is_pusch=False,device='cpu')
    re=torch.tensor([156,312,624],dtype=torch.int32)
    layers=torch.tensor([1,2,1],dtype=torch.int32)
    expected,_,_,_,_=calculate_tb_size(modulation_order=qm,target_coderate=rate,num_coded_bits=qm*re*layers,return_cw_length=False,device='cpu')
    actual=sionna_nr_tb_size_bits([(0,156,1),(10,312,2),(20,624,1)])
    assert actual == [int(x) for x in expected.tolist()]
    assert all(x>0 and x%8==0 for x in actual)
