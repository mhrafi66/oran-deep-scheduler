#!/usr/bin/env python3
"""GPU-first *diagnostic* benchmark for the Task-5 service-aware cell loop.

Real Sionna 21-sector interferer topology; ONE serving cell scheduled in
this initial benchmark. Existing final-allocation PHY and whole-grant Sionna
link abstraction are reused. The queue-aware best-channel policy here is
only a diagnostic driver, NOT a publication-quality scheduler baseline.

--profile synchronizes the GPU at every stage boundary and measures stage
latency. Without --profile, CUDA is synchronized only at window boundaries
so total throughput is not inflated by profiling synchronization overhead.

--refresh-every 0 reuses a single channel/CSI snapshot across TTIs;
--refresh-every N resamples fading on the same topology every N slots.
These are independent channel draws, NOT time-correlated mobility.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import torch


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    i = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, i)]


def summarize(samples: list[float]) -> dict[str, float | int | None]:
    return {
        'count': len(samples),
        'mean_ms': statistics.mean(samples) if samples else None,
        'p50_ms': percentile(samples, 0.50),
        'p95_ms': percentile(samples, 0.95),
        'max_ms': max(samples) if samples else None,
    }


class StageRecorder:
    """Stage timer; explicit synchronize is necessary for honest CUDA timing.

    Disabled during warm-up and unprofiled throughput mode. This intentionally
    makes profiled TTIs slower; compare throughput ONLY in unprofiled mode.
    """
    def __init__(self, device: str, *, profile: bool) -> None:
        self.device = torch.device(device)
        self.profile = profile
        self.enabled = False
        self.samples: dict[str, list[float]] = defaultdict(list)

    def synchronize(self) -> None:
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def measure(self, name: str, callback: Callable[[], Any]) -> Any:
        if not self.profile or not self.enabled:
            return callback()
        self.synchronize()
        start = time.perf_counter()
        try:
            return callback()
        finally:
            self.synchronize()
            self.samples[name].append((time.perf_counter() - start) * 1000)

    def wrap_method(self, obj: object, method_name: str, stage: str) -> None:
        original = getattr(obj, method_name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self.measure(stage, lambda: original(*args, **kwargs))

        setattr(obj, method_name, wrapped)


def deterministic_arrivals(tti: int, num_ues: int):
    """Fixed trace across devices, independent of the PHY and scheduler RNG."""
    from oran_scheduler.simulator.service_aware_traffic import FlowPacketArrival

    packets = []
    for ue in range(num_ues):
        if (tti + ue) % 3 == 0:
            packets.append(FlowPacketArrival(ue, 'control', 256))
        if (tti + 2 * ue) % 2 == 0:
            packets.append(FlowPacketArrival(ue, 'bulk', 1400))
    return tuple(packets)


def pack_by_priority(view, decision, tbs):
    """Fill *after* whole-TB sizing, without overcommitting queue bytes."""
    remaining = {tb.ue_index: tb.payload_capacity_bytes for tb in tbs}
    planned = {}
    for flow in sorted(view.flows, key=lambda f: (f.priority, f.ue_index, f.flow_id)):
        amount = min(flow.backlog_bytes, remaining.get(flow.ue_index, 0))
        if amount:
            planned[flow.key] = amount
            remaining[flow.ue_index] -= amount
    return planned


class DiagnosticPolicy:
    """One/two-UE-slot queue-aware greedy over *precomputed* CSI scores.

    Uses actual pre-service queue observations and cached RBG channel scores;
    never redoes CSI inside the policy. This is an integration workload, not
    an optimized PF or QoS scheduler, nor a credible MU-MIMO pairing baseline.
    """
    def __init__(self, *, rbg_score: torch.Tensor, user_slots: int) -> None:
        from oran_scheduler.schedulers.allocation import CellAllocation
        if rbg_score.ndim != 2 or rbg_score.shape[0] < 1:
            raise ValueError('Expected [candidate, RBG] CSI scores')
        if user_slots not in (1, 2):
            raise ValueError('user_slots must be 1 or 2')
        self.scores = rbg_score.detach()
        self.user_slots = user_slots
        self.num_ues = int(rbg_score.shape[0])
        self.empty = CellAllocation(torch.full(
            (user_slots, int(rbg_score.shape[1])), -1,
            dtype=torch.long, device=rbg_score.device,
        ))
        self.queue_ue_by_candidate = tuple(range(self.num_ues))

    def update_scores(self, rbg_score: torch.Tensor) -> None:
        if rbg_score.shape != self.scores.shape or rbg_score.device != self.scores.device:
            raise ValueError('Channel refresh changed candidate/RBG geometry')
        self.scores = rbg_score.detach()

    def __call__(self, view):
        from oran_scheduler.schedulers.allocation import CellAllocation
        from oran_scheduler.simulator.service_aware_downlink import DownlinkSchedulingDecision

        if len(view.ues) != self.num_ues:
            raise ValueError('QoS queue population differs from PHY candidate list')
        backlogs = [u.backlog_bytes for u in view.ues]
        if not any(backlogs):
            return DownlinkSchedulingDecision(self.empty, self.queue_ue_by_candidate)
        # CPU queue metadata is small. This one vector transfer is unavoidable
        # when the application queue runs on CPU and CSI stays on CUDA.
        weights = torch.tensor(
            [min(4.0, 1.0 + math.log1p(q) / 7.0) if q else 0.0 for q in backlogs],
            dtype=self.scores.dtype, device=self.scores.device,
        )
        valid = weights > 0
        metrics = self.scores * weights[:, None]
        metrics = metrics.masked_fill(~valid[:, None], float('-inf'))
        slots = min(self.user_slots, sum(1 for q in backlogs if q))
        selected = torch.topk(metrics, k=slots, dim=0).indices.to(dtype=torch.long)
        if slots < self.user_slots:
            padding = torch.full(
                (self.user_slots - slots, metrics.shape[1]), -1,
                dtype=torch.long, device=self.scores.device,
            )
            selected = torch.cat((selected, padding), dim=0)
        return DownlinkSchedulingDecision(
            CellAllocation(selected), self.queue_ue_by_candidate,
        )


def validate_options(args) -> None:
    if args.ttis < 2 or args.warmup < 0 or args.warmup >= args.ttis:
        raise ValueError('Require ttis >= 2 and 0 <= warmup < ttis')
    if args.ut_per_sector < 1 or args.candidates < 1:
        raise ValueError('UEs per sector and candidates must be positive')
    if args.user_slots not in (1, 2):
        raise ValueError('user-slots must be 1 or 2')
    if args.log_every < 0:
        raise ValueError('log-every must be nonnegative')
    if args.refresh_every < 0:
        raise ValueError('refresh-every must be nonnegative')
    if args.device != 'cuda:0' and not args.allow_cpu:
        raise ValueError('GPU-first run requires --device cuda:0; --allow-cpu is for debugging only')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; execute via Slurm on a GPU compute node')


def run(args) -> dict[str, Any]:
    validate_options(args)
    from sionna.phy.constants import BOLTZMANN_CONSTANT
    from sionna.phy.utils import dbm_to_watt
    from oran_scheduler.simulator.topology import TopologyConfig, generate_topology
    from oran_scheduler.simulator.channel import ChannelConfig, generate_frequency_channel
    from oran_scheduler.simulator.cell_association import build_cell_association
    from oran_scheduler.phy.rank_diagnostic import compute_ideal_svd_rank_diagnostic
    from oran_scheduler.phy.csi import compute_ideal_svd_csi, extract_serving_mimo_rbg_channel
    from oran_scheduler.phy.link_adaptation import LinkAdaptationConfig
    from oran_scheduler.phy.rate import RateConfig
    from oran_scheduler.phy.schedule_evaluator import evaluate_cell_allocation
    from oran_scheduler.simulator.service_aware_whole_grant_phy import FinalAllocationSINRCapture
    from oran_scheduler.simulator.service_aware_downlink import ServiceAwareDownlinkCellRunner
    from oran_scheduler.simulator.service_aware_nr_grants import NRGrantAssembler, NRBandwidthPart, PDSCHDataRegion
    from oran_scheduler.simulator.service_aware_traffic import QoSFlowSpec, ServiceAwareTrafficManager
    from oran_scheduler.simulator.service_aware_observation import ServiceAwareQoSObserver
    from oran_scheduler.simulator.service_aware_transmission import ServiceAwareTransmissionEngine
    from oran_scheduler.schedulers.allocation import CellAllocation

    # No autograd tracking or training graphs in this environment benchmark.
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    recorder = StageRecorder(args.device, profile=args.profile)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    timings: dict[str, float] = {}

    def timed_setup(name, fn):
        recorder.synchronize()
        start = time.perf_counter()
        result = fn()
        recorder.synchronize()
        timings[name] = (time.perf_counter() - start) * 1000
        return result

    topo_cfg = TopologyConfig(
        batch_size=1, num_rings=1, num_ut_per_sector=args.ut_per_sector,
        seed=args.seed, device=args.device,
    )
    channel_cfg = ChannelConfig(
        carrier_frequency_hz=4.0e9,
        subcarrier_spacing_hz=30e3,
        num_rbs=18, subcarriers_per_rb=12,
        num_ofdm_symbols=1, antenna_mode='paper', device=args.device,
    )
    topo = timed_setup('topology', lambda: generate_topology(topo_cfg))
    state: dict[str, Any] = {}

    def generate_channel():
        return generate_frequency_channel(
            topology=topo, topology_config=topo_cfg, channel_config=channel_cfg,
        )

    ch = timed_setup('initial_channel', generate_channel)
    association = timed_setup('association', lambda: build_cell_association(
        h_freq=ch.h_freq, precision=channel_cfg.precision,
    ))
    serving_bs = association.serving_bs
    cell = int(torch.argmax(association.num_ues_per_cell[0]).item())
    candidate_global = torch.nonzero(serving_bs[0] == cell, as_tuple=False).flatten()[:args.candidates]
    if candidate_global.numel() < 1:
        raise RuntimeError('No UEs associated with selected serving cell')
    num_ues = int(candidate_global.numel())
    power = dbm_to_watt(44.0, precision=channel_cfg.precision, device=args.device) / channel_cfg.num_subcarriers
    noise = torch.as_tensor(
        BOLTZMANN_CONSTANT * 294.0 * channel_cfg.subcarrier_spacing_hz,
        dtype=torch.float32, device=args.device,
    )
    phy_config = LinkAdaptationConfig(device=args.device)
    rate_config = RateConfig(device=args.device)

    def build_phy(channel):
        # Compute these once per channel snapshot; no recomputation per grant.
        rank = compute_ideal_svd_rank_diagnostic(
            h_freq=channel.h_freq, serving_bs=serving_bs, num_rbgs=18,
            tx_power_per_subcarrier_w=power, noise_power_per_subcarrier_w=noise,
        )
        serving = extract_serving_mimo_rbg_channel(
            h_freq=channel.h_freq, serving_bs=serving_bs, num_rbgs=18,
        )
        csi = compute_ideal_svd_csi(
            h_serving_rbg=serving.h_serving_rbg,
            rank1_rbg_score=rank.rank1_rbg_spectral_efficiency,
            rank2_rbg_score=rank.rank2_rbg_spectral_efficiency,
        )
        return csi, rank.rank1_rbg_spectral_efficiency[0, candidate_global].detach()

    csi, rbg_score = timed_setup('initial_rank_csi', lambda: build_phy(ch))
    state['channel'] = ch
    state['csi'] = csi
    policy = DiagnosticPolicy(rbg_score=rbg_score, user_slots=args.user_slots)
    del ch, csi, rbg_score

    queue = ServiceAwareTrafficManager(
        num_ues=num_ues, tti_duration_s=0.0005,
        flows=tuple(
            spec for ue in range(num_ues) for spec in (
                QoSFlowSpec(ue, 'control', 1, 10, pdb_ttis=5,
                            max_queue_bytes=8192, expiry_policy='drop'),
                QoSFlowSpec(ue, 'bulk', 9, 50, gbr_bps=1_000_000.0,
                            max_queue_bytes=65536),
            )
        ),
    )
    observer = ServiceAwareQoSObserver(queues=queue, gbr_window_ttis=20)
    decoder = ServiceAwareTransmissionEngine(seed=args.seed)
    grants = NRGrantAssembler(
        bandwidth=NRBandwidthPart(18, 1),
        region=PDSCHDataRegion(num_pdsch_symbols=14, dmrs_re_per_prb=12),
        channel_num_prbs=18,
    )
    capture = FinalAllocationSINRCapture()

    def link_source(view, decision):
        # Reuse the existing final-allocation PHY (capture-only, no redundant
        # per-RBG link adaptation), then assemble TB-wide Sionna estimates.
        recorder.measure('final_mu_mimo_sinr', lambda: evaluate_cell_allocation(
            allocation=decision.allocation,
            candidate_global_ue_indices=candidate_global,
            h_freq=state['channel'].h_freq,
            serving_cell_index=cell,
            recommended_rank=state['csi'].recommended_rank,
            rx_combiners=state['csi'].rx_combiners,
            csi_subcarrier_index=state['csi'].csi_subcarrier_index,
            subcarriers_per_rbg=12,
            tx_power_per_subcarrier_w=power,
            noise_power_per_subcarrier_w=noise,
            link_adaptation_config=phy_config,
            rate_config=rate_config,
            capture_post_precoder_sinr=capture.capture_rbg,
            capture_only=True,
        ))
        return recorder.measure('whole_grant_link_adaptation', lambda: capture.whole_links(
            allocation=decision.allocation, config=phy_config,
        ))

    def scheduled_policy(view):
        return recorder.measure('queue_csi_diagnostic_policy', lambda: policy(view))

    def scheduled_packer(view, decision, tbs):
        return recorder.measure('flow_payload_packing', lambda: pack_by_priority(view, decision, tbs))

    runner = ServiceAwareDownlinkCellRunner(
        queues=queue, observer=observer, decoder=decoder, grants=grants,
        policy=scheduled_policy, link_provider=link_source,
        flow_packer=scheduled_packer,
    )
    for obj, method, stage in (
        (queue, 'begin_tti', 'packet_arrivals_and_expiry'),
        (observer, 'observe', 'qos_observation'),
        (grants, 'build', 'nr_whole_grant_tbs'),
        (decoder, 'execute_tti', 'tb_decode_and_queue_service'),
        (runner.metrics, 'accept', 'metrics_conservation'),
    ):
        recorder.wrap_method(obj, method, stage)

    initial_peak = (torch.cuda.max_memory_allocated(args.device)
                    if torch.device(args.device).type == 'cuda' else 0)
    if torch.device(args.device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(args.device)
    total_ms_samples: list[float] = []
    live_memory_samples: list[dict[str, int]] = []
    window_start = None
    refreshed = 0
    recorder.synchronize()
    with torch.no_grad():
        for tti in range(args.ttis):
            if tti == args.warmup:
                recorder.synchronize()
                window_start = time.perf_counter()
                recorder.enabled = True
            start_ensure = time.perf_counter() if recorder.profile and recorder.enabled else 0.0
            # Do not keep old channel / CSI references when resampling.
            # Association and candidates stay fixed for this finite trace.
            if args.refresh_every and tti and tti % args.refresh_every == 0:
                state.clear()
                if recorder.profile:
                    channel = recorder.measure('channel_resampling', generate_channel)
                    new_csi, new_score = recorder.measure('rank_csi_refresh', lambda: build_phy(channel))
                else:
                    channel = generate_channel()
                    new_csi, new_score = build_phy(channel)
                state['channel'] = channel
                state['csi'] = new_csi
                policy.update_scores(new_score)
                refreshed += 1
            result = runner.step(
                tti_index=tti, arrivals=deterministic_arrivals(tti, num_ues),
            )
            if result.tti_index != tti:
                raise AssertionError('TTI index mismatch')
            if recorder.profile and recorder.enabled:
                recorder.synchronize()
                total_ms_samples.append((time.perf_counter() - start_ensure) * 1000)
            # Allocation accounting is a CPU-side allocator query, not a
            # cuda synchronize; track post-slot *live* bytes to flag leaks.
            if tti >= args.warmup and (tti == args.warmup or tti == args.ttis - 1
                                       or (tti - args.warmup) % max(1, (args.ttis-args.warmup)//5) == 0):
                if torch.device(args.device).type == 'cuda':
                    live_memory_samples.append({
                        'tti': tti,
                        'live_allocated_bytes': int(torch.cuda.memory_allocated(args.device)),
                    })
            if args.log_every and (tti + 1) % args.log_every == 0:
                print(f'PROGRESS_TTI {tti + 1}/{args.ttis}', flush=True)
    recorder.synchronize()
    if window_start is None:
        raise AssertionError('No benchmark window')
    elapsed_s = time.perf_counter() - window_start
    report = runner.report()
    measured = args.ttis - args.warmup
    accounted = (report.delivered_bytes + report.overflow_dropped_bytes
                 + report.expired_dropped_bytes + report.final_backlog_bytes)
    if report.arrived_bytes != accounted:
        raise AssertionError('Payload conservation violated')
    summary = {
        'benchmark_mode': 'instrumented_stage_profile' if args.profile else 'uninstrumented_throughput',
        'timing_warning': ('In profile mode synchronization occurs after every stage; '
                           'TTIs/s is NOT comparable to uninstrumented throughput.'
                           if args.profile else 'One CUDA synchronization at each timing window boundary.'),
        'channel_mode': 'fixed_snapshot' if args.refresh_every == 0 else 'independent_fading_draws',
        'refresh_interval_ttis': args.refresh_every,
        'device': str(args.device),
        'gpu_name': (torch.cuda.get_device_name(args.device)
                     if torch.device(args.device).type == 'cuda' else None),
        'seed': args.seed,
        'git_commit': args.git_commit,
        'full_network_sectors_in_channel': topo_cfg.num_cells,
        'num_network_ues': topo_cfg.num_ues,
        'scheduled_cells': 1,
        'scheduled_cell_id': cell,
        'scheduled_ues': num_ues,
        'rbg_count': 18,
        'bwp_prbs': 18,
        'user_slots': args.user_slots,
        'total_ttis': args.ttis,
        'warmup_ttis': args.warmup,
        'timed_ttis': measured,
        'timed_wall_seconds': elapsed_s,
        'timed_ttis_per_second': measured / elapsed_s,
        'channel_refreshes': refreshed,
        'one_time_initialization_ms': timings,
        'stage_statistics': {k: summarize(v) for k, v in sorted(recorder.samples.items())},
        'profiled_total_slot_latency': summarize(total_ms_samples),
        'peak_cuda_bytes_during_initialization': int(initial_peak),
        'peak_cuda_bytes_during_run': (int(torch.cuda.max_memory_allocated(args.device))
                                       if torch.device(args.device).type == 'cuda' else 0),
        'max_cuda_reserved_bytes_during_run': (int(torch.cuda.max_memory_reserved(args.device))
                                              if torch.device(args.device).type == 'cuda' else 0),
        'live_cuda_memory_samples': live_memory_samples,
        'live_cuda_memory_change_bytes': (
            live_memory_samples[-1]['live_allocated_bytes'] - live_memory_samples[0]['live_allocated_bytes']
            if len(live_memory_samples) >= 2 else None
        ),
        'queue_report': asdict(report),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--ttis', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--ut-per-sector', type=int, default=2)
    parser.add_argument('--candidates', type=int, default=4)
    parser.add_argument('--user-slots', type=int, choices=(1, 2), default=1)
    parser.add_argument('--refresh-every', type=int, default=0)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--seed', type=int, default=20261021)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    import subprocess
    args.git_commit = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
        check=False,
    ).stdout.strip() or 'unknown'
    output = run(args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
        print('JSON_RESULT:', args.json)
    print('BENCHMARK_MODE:', output['benchmark_mode'])
    print('GPU:', output['gpu_name'])
    print('CHANNEL_MODE:', output['channel_mode'])
    print('NETWORK:', output['full_network_sectors_in_channel'], 'sectors,',
          output['num_network_ues'], 'UEs; scheduled 1 cell with',
          output['scheduled_ues'], 'UEs')
    print('TIMED_TTIS_PER_SECOND:', f"{output['timed_ttis_per_second']:.3f}")
    print('PEAK_CUDA_MEMORY_GIB:', f"{output['peak_cuda_bytes_during_run']/1024**3:.3f}")
    for name, value in output['stage_statistics'].items():
        print('STAGE:', name, value)
    print('DELIVERED_BYTES:', output['queue_report']['delivered_bytes'],
          'ARRIVED_BYTES:', output['queue_report']['arrived_bytes'])
    print('SERVICE_AWARE_GPU_BENCH_PASS')


if __name__ == '__main__':
    main()
