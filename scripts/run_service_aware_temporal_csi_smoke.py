#!/usr/bin/env python3
"""Task 7 causality smoke; --real-sionna runs opt-in correlated GPU PHY test.

Finite 2-TTI short-window diagnostic; does not certify full 21-cell or long-run
continuous mobility. Uses real Sionna Doppler window and final PHY exactly once.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.allocation import CellAllocation
from oran_scheduler.simulator.service_aware_downlink import (
    DownlinkSchedulingDecision, ServiceAwareDownlinkCellRunner,
)
from oran_scheduler.simulator.service_aware_nr_grants import (
    NRBandwidthPart, NRGrantAssembler, PDSCHDataRegion, WholeGrantLink,
)
from oran_scheduler.simulator.service_aware_observation import ServiceAwareQoSObserver
from oran_scheduler.simulator.service_aware_temporal_csi import (
    CSIReport, ServiceAwareTemporalCSIAdapter, TemporalWindowFrameSource,
)
from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival, QoSFlowSpec, ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_transmission import ServiceAwareTransmissionEngine


def pack_flows(view, decision, transport_blocks):
    available={tb.ue_index:tb.payload_capacity_bytes for tb in transport_blocks}
    selected={}
    for f in sorted(view.flows,key=lambda x:x.priority):
        size=min(f.backlog_bytes,available.get(f.ue_index,0))
        if size:
            selected[f.key]=size
            available[f.ue_index]-=size
    return selected


def make_runner(adapter, *, real_sionna):
    queues=ServiceAwareTrafficManager(
        num_ues=1,tti_duration_s=0.0005,
        flows=(QoSFlowSpec(0,'critical',1,10,pdb_ttis=4),),
    )
    observer=ServiceAwareQoSObserver(queues=queues)
    decoder=ServiceAwareTransmissionEngine(seed=14)
    grants=NRGrantAssembler(
        bandwidth=NRBandwidthPart(18,1),
        region=PDSCHDataRegion(num_pdsch_symbols=14,dmrs_re_per_prb=12),
        channel_num_prbs=18,
        tbs_backend=None if real_sionna else lambda specs:[8*100 for _ in specs],
    )
    return ServiceAwareDownlinkCellRunner(
        queues=queues,observer=observer,decoder=decoder,grants=grants,
        policy=adapter.policy,link_provider=adapter.link_provider,flow_packer=pack_flows,
    )


def run_component():
    # Raw deterministic current H_t sequence; it is not a physical channel.
    physical_seen=[]
    report_seen=[]
    def source(t):
        return torch.full((1,1,1,1,1,1,12),complex(t+1,0.0),dtype=torch.complex64)
    def builder(h,t):
        return CSIReport(t,(0,),int(h.real[0,0,0,0,0,0,0].item()))
    def policy(qos,published):
        report_seen.append((published.decision_tti,published.report_tti,published.age_ttis))
        row=torch.tensor([[0,0,0]+[-1]*15],dtype=torch.long)
        return DownlinkSchedulingDecision(CellAllocation(row),(0,),None)
    def link(qos,decision,h,published):
        physical_seen.append((qos.tti_index,int(h.real[0,0,0,0,0,0,0].item())))
        return {0:WholeGrantLink(0,(0,1,2),20,1.0 if qos.tti_index==0 else 0.0)}
    adapter=ServiceAwareTemporalCSIAdapter(
        frame_source=source,report_builder=builder,policy=policy,
        physical_link_provider=link,delay_ttis=1,
    )
    runner=make_runner(adapter,real_sionna=False)
    for t in range(3):
        runner.step(tti_index=t,arrivals=(FlowPacketArrival(0,'critical',20),) if t==0 else ())
    r=runner.report()
    assert report_seen==[(0,0,0),(1,0,1),(2,1,1)]
    assert physical_seen==[(0,1),(1,2),(2,3)]
    assert (r.arrived_bytes==20 and r.delivered_bytes==20 and r.final_backlog_bytes==0)
    print('CAUSAL_REPORT_TIMELINE:',report_seen)
    print('CURRENT_PHY_TIMELINE:',physical_seen)
    print('SERVICE_AWARE_TASK7_COMPONENT_PASS')


@dataclass(frozen=True)
class RadioPayload:
    csi: object
    scores: torch.Tensor


def run_real_sionna(device):
    if not torch.cuda.is_available() and str(device).startswith('cuda'):
        raise RuntimeError('This mode must run on an allocated GPU compute node')
    from sionna.phy.constants import BOLTZMANN_CONSTANT
    from sionna.phy.utils import dbm_to_watt
    from oran_scheduler.simulator.channel import ChannelConfig
    from oran_scheduler.simulator.topology import TopologyConfig,generate_topology,subset_topology_ues
    from oran_scheduler.simulator.temporal_channel import VelocityWindowFrequencyChannelRuntime
    from oran_scheduler.simulator.cell_association import build_cell_association
    from oran_scheduler.phy.rank_diagnostic import compute_ideal_svd_rank_diagnostic
    from oran_scheduler.phy.csi import compute_ideal_svd_csi,extract_serving_mimo_rbg_channel
    from oran_scheduler.phy.link_adaptation import LinkAdaptationConfig
    from oran_scheduler.phy.rate import RateConfig
    from oran_scheduler.phy.schedule_evaluator import evaluate_cell_allocation
    from oran_scheduler.simulator.service_aware_whole_grant_phy import FinalAllocationSINRCapture

    with torch.no_grad():
        topo_config=TopologyConfig(batch_size=1,num_rings=1,num_ut_per_sector=1,
                                   seed=321,device=device)
        topology=generate_topology(topo_config)
        # Two served UEs and all 21 BSs: keeps paper MIMO while avoiding
        # an unnecessary full temporal frequency tensor for 21 UEs.
        topology=subset_topology_ues(
            topology=topology,
            global_ue_indices=torch.tensor([0,1],dtype=torch.long,device=device),
        )
        channel_config=ChannelConfig(
            carrier_frequency_hz=4.0e9,subcarrier_spacing_hz=30e3,
            num_rbs=18,subcarriers_per_rb=12,num_ofdm_symbols=1,
            antenna_mode='paper',device=device,seed=141,
        )
        runtime=VelocityWindowFrequencyChannelRuntime(config=channel_config)
        window=runtime.generate_window(
            topology=topology,seed=987,batch_size=1,num_ttis=2,tti_duration_s=0.0005,
        )
        source=TemporalWindowFrameSource(window)
        initial=source(0)
        association=build_cell_association(h_freq=initial,precision=channel_config.precision)
        serving_bs=association.serving_bs
        cell=int(torch.argmax(association.num_ues_per_cell[0]).item())
        candidate=torch.nonzero(serving_bs[0]==cell,as_tuple=False).flatten()[:1]
        if candidate.numel()!=1:
            raise RuntimeError('No candidate UE in serving cell')
        membership=(int(candidate[0].item()),)
        power=dbm_to_watt(44.0,precision=channel_config.precision,device=device)/channel_config.num_subcarriers
        noise=torch.as_tensor(BOLTZMANN_CONSTANT*294.0*channel_config.subcarrier_spacing_hz,
                              dtype=torch.float32,device=device)
        la=LinkAdaptationConfig(device=device)
        rate=RateConfig(device=device)
        capture=FinalAllocationSINRCapture()
        reports=[]
        phy_seen=[]

        def report_builder(h,t):
            rank=compute_ideal_svd_rank_diagnostic(
                h_freq=h,serving_bs=serving_bs,num_rbgs=18,
                tx_power_per_subcarrier_w=power,noise_power_per_subcarrier_w=noise,
            )
            extracted=extract_serving_mimo_rbg_channel(
                h_freq=h,serving_bs=serving_bs,num_rbgs=18,
            )
            csi=compute_ideal_svd_csi(
                h_serving_rbg=extracted.h_serving_rbg,
                rank1_rbg_score=rank.rank1_rbg_spectral_efficiency,
                rank2_rbg_score=rank.rank2_rbg_spectral_efficiency,
            )
            score=rank.rank1_rbg_spectral_efficiency[0,candidate[0]].detach()
            return CSIReport(t,membership,RadioPayload(csi,score))

        def policy(qos,published):
            reports.append((qos.tti_index,published.report_tti,published.age_ttis))
            # Choose three RBGs using ONLY the reported frequency-dependent CSI.
            good=torch.topk(published.report.payload.scores,k=3).indices.tolist()
            row=torch.full((1,18),-1,dtype=torch.long,device=device)
            row[0,good]=0
            return DownlinkSchedulingDecision(CellAllocation(row), (0,), None)

        def current_link(qos,decision,h,published):
            phy_seen.append(qos.tti_index)
            csi=published.report.payload.csi
            evaluate_cell_allocation(
                allocation=decision.allocation,
                candidate_global_ue_indices=candidate,
                h_freq=h,serving_cell_index=cell,
                recommended_rank=csi.recommended_rank,rx_combiners=csi.rx_combiners,
                csi_subcarrier_index=csi.csi_subcarrier_index,subcarriers_per_rbg=12,
                tx_power_per_subcarrier_w=power,noise_power_per_subcarrier_w=noise,
                link_adaptation_config=la,rate_config=rate,
                capture_post_precoder_sinr=capture.capture_rbg,capture_only=True,
            )
            return capture.whole_links(allocation=decision.allocation,config=la)

        adapter=ServiceAwareTemporalCSIAdapter(
            frame_source=source,report_builder=report_builder,
            policy=policy,physical_link_provider=current_link,delay_ttis=1,
        )
        runner=make_runner(adapter,real_sionna=True)
        for t in range(2):
            runner.step(tti_index=t,arrivals=(FlowPacketArrival(0,'critical',20),) if t==0 else ())
        report=runner.report()
        assert reports==[(0,0,0),(1,0,1)]
        assert phy_seen==[0,1]
        assert adapter.frame_fetch_count==2 and adapter.report_build_count==2
        assert report.arrived_bytes==(report.delivered_bytes+
            report.overflow_dropped_bytes+report.expired_dropped_bytes+report.final_backlog_bytes)
        change=torch.max(torch.abs(source(0)-source(1))).item()
        assert change>0.0, 'Real temporal channel did not evolve'
        print('REAL_SIONNA_CHANNEL_MAX_CHANGE:',change)
        print('REAL_SIONNA_CAUSAL_REPORT_TIMELINE:',reports)
        print('DELIVERED_BYTES:',report.delivered_bytes,'REMAINING:',report.final_backlog_bytes)
        print('SERVICE_AWARE_TASK7_REAL_SIONNA_PASS')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--real-sionna',action='store_true')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.real_sionna:
        run_real_sionna(args.device)
    else:
        run_component()


if __name__=='__main__':
    main()
