#!/usr/bin/env python3
"""Task-5 opt-in end-to-end component smoke or real Sionna PHY smoke.

Default: deterministic synthetic WHOLE-GRANT link data, exercises full packet
lifecycle but does NOT claim physical BLER validation.
--real-sionna: generate a real Sionna multicell channel, use the *existing*
final MU-MIMO evaluator once and capture its post-SINR for one combined-TB
Sionna link prediction. Requires CHPC compute GPU for practical performance.
"""
from __future__ import annotations

import argparse

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-sionna', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    queue = ServiceAwareTrafficManager(
        num_ues=1, tti_duration_s=0.0005,
        flows=(QoSFlowSpec(0,'control',1,10,pdb_ttis=2),
               QoSFlowSpec(0,'bulk',9,50,gbr_bps=8000.0)),
    )
    obs = ServiceAwareQoSObserver(queues=queue, gbr_window_ttis=3)
    decode = ServiceAwareTransmissionEngine(seed=42)
    asm = NRGrantAssembler(
        bandwidth=NRBandwidthPart(18,1),
        region=PDSCHDataRegion(num_pdsch_symbols=14,dmrs_re_per_prb=12),
        channel_num_prbs=18,
        tbs_backend=None if args.real_sionna else lambda keys: [2048 for _ in keys],
    )
    allocation = CellAllocation(torch.tensor([[0,0,0]+[-1]*15],dtype=torch.long))

    def policy(view):
        assert view.total_backlog_bytes == sum(f.backlog_bytes for f in view.flows)
        return DownlinkSchedulingDecision(
            allocation=allocation,
            queue_ue_by_candidate=(0,),
            planned_flow_bytes=None,
        )

    if args.real_sionna:
        from sionna.phy.constants import BOLTZMANN_CONSTANT
        from sionna.phy.utils import dbm_to_watt
        from oran_scheduler.simulator.channel import ChannelConfig, generate_frequency_channel
        from oran_scheduler.simulator.topology import TopologyConfig, generate_topology
        from oran_scheduler.simulator.cell_association import build_cell_association
        from oran_scheduler.phy.csi import compute_ideal_svd_csi, extract_serving_mimo_rbg_channel
        from oran_scheduler.phy.rank_diagnostic import compute_ideal_svd_rank_diagnostic
        from oran_scheduler.phy.link_adaptation import LinkAdaptationConfig
        from oran_scheduler.phy.rate import RateConfig
        from oran_scheduler.phy.schedule_evaluator import evaluate_cell_allocation
        from oran_scheduler.simulator.service_aware_whole_grant_phy import FinalAllocationSINRCapture

        device = args.device
        topology_cfg = TopologyConfig(batch_size=1,num_rings=1,num_ut_per_sector=1,device=device)
        topology = generate_topology(topology_cfg)
        channel_cfg = ChannelConfig(
            carrier_frequency_hz=4.0e9,subcarrier_spacing_hz=30e3,
            num_rbs=18,subcarriers_per_rb=12,num_ofdm_symbols=1,
            antenna_mode='paper',device=device,
        )
        ch = generate_frequency_channel(
            topology=topology,topology_config=topology_cfg,channel_config=channel_cfg,
        )
        association = build_cell_association(h_freq=ch.h_freq,precision=channel_cfg.precision)
        active = torch.nonzero(association.num_ues_per_cell[0] > 0, as_tuple=False).flatten()
        cell = int(active[0].item())
        candidate = torch.nonzero(association.serving_bs[0]==cell,as_tuple=False).flatten()[:1]
        power=dbm_to_watt(44.0,precision=channel_cfg.precision,device=device)/channel_cfg.num_subcarriers
        noise=torch.as_tensor(BOLTZMANN_CONSTANT * 294.0 * channel_cfg.subcarrier_spacing_hz,
                              dtype=torch.float32,device=device)
        rank_data=compute_ideal_svd_rank_diagnostic(
            h_freq=ch.h_freq,serving_bs=association.serving_bs,num_rbgs=18,
            tx_power_per_subcarrier_w=power,noise_power_per_subcarrier_w=noise,
        )
        mimo=extract_serving_mimo_rbg_channel(
            h_freq=ch.h_freq,serving_bs=association.serving_bs,num_rbgs=18,
        )
        csi=compute_ideal_svd_csi(
            h_serving_rbg=mimo.h_serving_rbg,
            rank1_rbg_score=rank_data.rank1_rbg_spectral_efficiency,
            rank2_rbg_score=rank_data.rank2_rbg_spectral_efficiency,
        )
        phy_config=LinkAdaptationConfig(device=device)
        rate_config=RateConfig(device=device)
        capture=FinalAllocationSINRCapture()

        def link_source(view, decision):
            # Reuse existing final PHY evaluator once, with only a passive hook.
            evaluate_cell_allocation(
                allocation=CellAllocation(decision.allocation.candidate_by_user_slot.to(device)),
                candidate_global_ue_indices=candidate,
                h_freq=ch.h_freq,
                serving_cell_index=cell,
                recommended_rank=csi.recommended_rank,
                rx_combiners=csi.rx_combiners,
                csi_subcarrier_index=csi.csi_subcarrier_index,
                subcarriers_per_rbg=12,
                tx_power_per_subcarrier_w=power,
                noise_power_per_subcarrier_w=noise,
                link_adaptation_config=phy_config,
                rate_config=rate_config,
                capture_post_precoder_sinr=capture.capture_rbg,
                capture_only=True,
            )
            return capture.whole_links(allocation=decision.allocation,config=phy_config)
    else:
        def link_source(view, decision):
            # Deliberate synthetic oracle for deterministic queue integration.
            return {0: WholeGrantLink(0,(0,1,2),mcs_index=20,tbler=1.0 if view.tti_index==0 else 0.0)}

    def pack_flows(view, decision, tbs):
        # Demonstration only: priority-ordered fill after actual TBS is known.
        # This is NOT a new scheduling baseline; arbitrary user packers may plug in.
        remaining = {tb.ue_index:tb.payload_capacity_bytes for tb in tbs}
        packed = {}
        for f in sorted(view.flows,key=lambda f: f.priority):
            n=min(f.backlog_bytes,remaining.get(f.ue_index,0))
            if n:
                packed[f.key]=n
                remaining[f.ue_index]-=n
        return packed

    runner=ServiceAwareDownlinkCellRunner(
        queues=queue,observer=obs,decoder=decode,grants=asm,
        policy=policy,link_provider=link_source,flow_packer=pack_flows,
    )
    tti0=runner.step(tti_index=0,arrivals=(
        FlowPacketArrival(0,'control',30),FlowPacketArrival(0,'bulk',25),
    ))
    tti1=runner.step(tti_index=1,arrivals=())
    report=runner.report()
    assert report.arrived_bytes == (report.delivered_bytes + report.overflow_dropped_bytes
                                    + report.expired_dropped_bytes + report.final_backlog_bytes)
    assert report.measured_ttis==2
    print('TTI0 delivered:',tti0.transmission.traffic_service.delivered_bytes,
          'TTI1 delivered:',tti1.transmission.traffic_service.delivered_bytes)
    print('On-time fraction:',report.on_time_ratio_of_arrived_packets,
          'Throughput bps:',report.throughput_bps,
          'Remaining bytes:',report.final_backlog_bytes)
    print('SERVICE_AWARE_TASK5_REAL_SIONNA_PASS' if args.real_sionna else 'SERVICE_AWARE_TASK5_COMPONENT_PASS')


if __name__=='__main__':
    main()
