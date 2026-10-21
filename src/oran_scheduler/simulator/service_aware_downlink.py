"""Task 5: one-cell, opt-in downlink TTI integration and QoS accounting.

Ordered lifecycle: arrivals/expiry -> PRE-service QoS observation -> externally
selected RBG/MU-MIMO allocation -> WHOLE-GRANT link prediction -> one NR TB
per scheduled UE -> sampled decoding -> authoritative per-flow queue service ->
online conservation/metrics.

This is a per-cell component designed to plug into an existing Sionna PHY /
scheduler; it does not create a second PHY evaluation or choose a QoS policy.
A whole-grant link predictor MUST be supplied. In particular a per-RBG BLER
cannot be silently reinterpreted as multi-RBG TB BLER.

A policy/provider exception leaves the queue TTI open. Call reset() and replay
the episode; rollback of packet arrivals or randomness is deliberately not
attempted. Legacy Nokia scheduling remains untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable, Mapping, Sequence

from oran_scheduler.schedulers.allocation import CellAllocation
from oran_scheduler.simulator.service_aware_nr_grants import NRGrantAssembler, WholeGrantLink
from oran_scheduler.simulator.service_aware_observation import (
    QoSCellObservation, ServiceAwareQoSObserver,
)
from oran_scheduler.simulator.service_aware_traffic import (
    FlowKey, FlowPacketArrival, FlowTTIStart, QoSFlowSpec,
    ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_transmission import (
    DownlinkTB, ServiceAwareTransmissionEngine, TransmissionTTIResult,
)


@dataclass(frozen=True)
class DownlinkSchedulingDecision:
    """Final schedule: action matrix, candidate/queue UE mapping and flow payload.

    `planned_flow_bytes` is the explicit in-TB payload placement; the
    simulator does NOT choose priority/flow order on the policy's behalf.
    """
    allocation: CellAllocation
    queue_ue_by_candidate: tuple[int, ...]
    # Optional. The provider may instead supply a flow packer AFTER the PHY
    # determines actual whole-TB payload sizes; never assume TB size in advance.
    planned_flow_bytes: Mapping[FlowKey, int] | None = None


Policy = Callable[[QoSCellObservation], DownlinkSchedulingDecision]
WholeLinkProvider = Callable[
    [QoSCellObservation, DownlinkSchedulingDecision], Mapping[int, WholeGrantLink]
]
FlowPacker = Callable[
    [QoSCellObservation, DownlinkSchedulingDecision, tuple[DownlinkTB, ...]], Mapping[FlowKey, int]
]


@dataclass(frozen=True)
class DownlinkTTIResult:
    tti_index: int
    queue_start: FlowTTIStart
    pre_service_observation: QoSCellObservation
    transmission: TransmissionTTIResult
    backlog_after_bytes: int


@dataclass(frozen=True)
class FlowQoSReport:
    ue_index: int
    flow_id: str
    five_qi: int
    arrived_packets: int
    arrived_bytes: int
    admitted_packets: int
    delivered_bytes: int
    completed_packets: int
    on_time_completed_packets: int
    late_completed_packets: int
    deadline_misses: int
    overflow_dropped_packets: int
    overflow_dropped_bytes: int
    expired_dropped_packets: int
    expired_dropped_bytes: int
    queued_packets: int
    backlog_bytes: int
    on_time_ratio_of_arrivals: float | None
    completed_delay_p50_slots: int | None
    completed_delay_p95_slots: int | None
    completed_delay_p99_slots: int | None
    gbr_backlogged_slots: int
    backlogged_slot_delivery_to_gbr_ratio: float | None


@dataclass(frozen=True)
class DownlinkQoSReport:
    measured_ttis: int
    duration_s: float
    arrived_bytes: int
    delivered_bytes: int
    overflow_dropped_bytes: int
    expired_dropped_bytes: int
    final_backlog_bytes: int
    scheduled_capacity_bytes: int
    decoded_capacity_bytes: int
    failed_capacity_bytes: int
    throughput_bps: float
    decoded_capacity_utilization: float | None
    on_time_ratio_of_arrived_packets: float | None
    jain_fairness_offered_ues: float | None
    per_ue_throughput_bps: tuple[float, ...]
    flows: tuple[FlowQoSReport, ...]


def _percentile_histogram(hist: Sequence[tuple[int, int]], percentile: int) -> int | None:
    """Exact nearest-rank quantile; no expanding samples into packet arrays."""
    size = sum(count for _, count in hist)
    if size == 0:
        return None
    rank = max(1, ceil(percentile * size / 100))
    prefix = 0
    for delay, count in hist:
        prefix += count
        if prefix >= rank:
            return delay
    raise AssertionError("Malformed packet delay histogram")


class ServiceAwareDownlinkMetrics:
    """Streaming scalar statistics with final exact queue/packet conservation.

    No per-TTI rows are retained by default. Exact completion-delay histograms
    are maintained incrementally inside Task 1 only when a packet completes.
    Memory is O(flows + UEs + distinct observed packet delays), not O(TTIs).
    """

    def __init__(self, queues: ServiceAwareTrafficManager) -> None:
        self.queues = queues
        self.specs = queues.configured_flows()
        self.reset()

    def reset(self) -> None:
        if self.queues.has_active_tti or self.queues.last_completed_tti_index is not None:
            raise RuntimeError("Reset queues before resetting metrics")
        self._n = 0
        self._arrived = 0
        self._delivered = 0
        self._overflow = 0
        self._expired = 0
        self._scheduled = 0
        self._decoded = 0
        self._failed = 0
        self._gbr_backlogged_slots = {s.key: 0 for s in self.specs if s.gbr_bps is not None}

    def accept(
        self,
        *,
        started: FlowTTIStart,
        observed: QoSCellObservation,
        result: TransmissionTTIResult,
    ) -> None:
        if started.tti_index != self._n or observed.tti_index != self._n or result.tti_index != self._n:
            raise ValueError("Missing, repeated, or out-of-order TTI")
        if self.queues.has_active_tti or self.queues.last_completed_tti_index != self._n:
            raise RuntimeError("Metrics must be updated after finalized queue service")
        if started.backlog_bytes != observed.total_backlog_bytes:
            raise AssertionError("Observation differs from after-arrival queue")
        if observed.total_backlog_bytes - result.traffic_service.delivered_bytes != self.queues.backlog_bytes:
            raise AssertionError("TTI queue conservation failed")
        if result.traffic_service.delivered_bytes != result.decoded_flow_payload_bytes:
            raise AssertionError("Counted failed/padded TB bytes as delivered")
        if result.decoded_capacity_bytes != result.traffic_service.delivered_bytes + result.unused_decoded_capacity_bytes:
            raise AssertionError("Decoded TB capacity is not conserved")
        if result.scheduled_capacity_bytes != result.decoded_capacity_bytes + result.failed_capacity_bytes:
            raise AssertionError("Scheduled TB capacity is not conserved")
        self._n += 1
        self._arrived += started.arrived_bytes
        self._overflow += started.overflow_dropped_bytes
        self._expired += started.expired_dropped_bytes
        self._delivered += result.traffic_service.delivered_bytes
        self._scheduled += result.scheduled_capacity_bytes
        self._decoded += result.decoded_capacity_bytes
        self._failed += result.failed_capacity_bytes
        for flow in observed.flows:
            if flow.backlog_bytes and flow.key in self._gbr_backlogged_slots:
                self._gbr_backlogged_slots[flow.key] += 1
        if self._arrived != self._delivered + self._overflow + self._expired + self.queues.backlog_bytes:
            raise AssertionError("Cumulative payload-byte conservation failed")

    def report(self) -> DownlinkQoSReport:
        """Produce full summary after a finalized TTI; exact, not an SLA oracle.

        Deadline misses are counted by Task 1 when expiry is processed at the
        START of the next slot; unresolved packets at the finite horizon stay
        in final_backlog (do not silently label them successful/missed).
        """
        if self.queues.has_active_tti:
            raise RuntimeError("Cannot report during a TTI")
        if self.queues.last_completed_tti_index != (None if self._n == 0 else self._n - 1):
            raise RuntimeError("Metrics episode and queue timeline do not match")
        duration = self._n * self.queues.tti_duration_s
        flows: list[FlowQoSReport] = []
        flow_snapshots = [self.queues.flow_snapshot(s.ue_index, s.flow_id) for s in self.specs]
        arrival_packets = 0
        on_time_packets = 0
        ue_arrivals = [0] * self.queues.num_ues
        ue_delivered = [0] * self.queues.num_ues
        for spec, snap in zip(self.specs, flow_snapshots):
            if (snap.arrived_bytes != snap.delivered_bytes + snap.overflow_dropped_bytes
                    + snap.expired_dropped_bytes + snap.backlog_bytes):
                raise AssertionError("Per-flow byte conservation failed")
            if snap.admitted_packets != (snap.completed_packets + snap.expired_dropped_packets
                                         + snap.queued_packets):
                raise AssertionError("Per-flow packet conservation failed")
            if snap.completed_packets != snap.on_time_completed_packets + snap.late_completed_packets:
                raise AssertionError("Completed packet categories not conserved")
            hist = self.queues.completed_delay_histogram(spec.ue_index, spec.flow_id)
            if sum(v for _, v in hist) != snap.completed_packets:
                raise AssertionError("Completed delay histogram lost packet observations")
            arrival_packets += snap.arrived_packets
            on_time_packets += snap.on_time_completed_packets
            ue_arrivals[spec.ue_index] += snap.arrived_bytes
            ue_delivered[spec.ue_index] += snap.delivered_bytes
            backlogged = self._gbr_backlogged_slots.get(spec.key, 0)
            gbr_target = (spec.gbr_bps * backlogged * self.queues.tti_duration_s / 8.0
                          if spec.gbr_bps is not None else 0.0)
            flows.append(FlowQoSReport(
                ue_index=spec.ue_index,
                flow_id=spec.flow_id,
                five_qi=spec.five_qi,
                arrived_packets=snap.arrived_packets,
                arrived_bytes=snap.arrived_bytes,
                admitted_packets=snap.admitted_packets,
                delivered_bytes=snap.delivered_bytes,
                completed_packets=snap.completed_packets,
                on_time_completed_packets=snap.on_time_completed_packets,
                late_completed_packets=snap.late_completed_packets,
                deadline_misses=snap.deadline_misses,
                overflow_dropped_packets=snap.overflow_dropped_packets,
                overflow_dropped_bytes=snap.overflow_dropped_bytes,
                expired_dropped_packets=snap.expired_dropped_packets,
                expired_dropped_bytes=snap.expired_dropped_bytes,
                queued_packets=snap.queued_packets,
                backlog_bytes=snap.backlog_bytes,
                on_time_ratio_of_arrivals=(snap.on_time_completed_packets / snap.arrived_packets
                                           if snap.arrived_packets else None),
                completed_delay_p50_slots=_percentile_histogram(hist, 50),
                completed_delay_p95_slots=_percentile_histogram(hist, 95),
                completed_delay_p99_slots=_percentile_histogram(hist, 99),
                gbr_backlogged_slots=backlogged,
                backlogged_slot_delivery_to_gbr_ratio=(snap.delivered_bytes / gbr_target
                                                       if gbr_target > 0 else None),
            ))
        if sum(f.arrived_bytes for f in flows) != self._arrived:
            raise AssertionError("Flow totals do not equal streaming arrival metrics")
        if sum(f.delivered_bytes for f in flows) != self._delivered:
            raise AssertionError("Flow totals do not equal streaming delivered metrics")
        rates = tuple(v * 8.0 / duration if duration else 0.0 for v in ue_delivered)
        offered_rates = [rates[u] for u, arrival in enumerate(ue_arrivals) if arrival > 0]
        fairness = None
        if offered_rates:
            power = sum(rate * rate for rate in offered_rates)
            fairness = ((sum(offered_rates) ** 2) / (len(offered_rates) * power)
                        if power else None)
        return DownlinkQoSReport(
            measured_ttis=self._n,
            duration_s=duration,
            arrived_bytes=self._arrived,
            delivered_bytes=self._delivered,
            overflow_dropped_bytes=self._overflow,
            expired_dropped_bytes=self._expired,
            final_backlog_bytes=self.queues.backlog_bytes,
            scheduled_capacity_bytes=self._scheduled,
            decoded_capacity_bytes=self._decoded,
            failed_capacity_bytes=self._failed,
            throughput_bps=self._delivered * 8.0 / duration if duration else 0.0,
            decoded_capacity_utilization=(self._delivered / self._decoded if self._decoded else None),
            on_time_ratio_of_arrived_packets=(on_time_packets / arrival_packets if arrival_packets else None),
            jain_fairness_offered_ues=fairness,
            per_ue_throughput_bps=rates,
            flows=tuple(flows),
        )


class ServiceAwareDownlinkCellRunner:
    """Coordinates all four existing Task modules, leaving PHY/policy pluggable.

    Every TTI calls the supplied policy ONCE with only pre-service local QoS.
    Then calls link_provider ONCE with the actual final cell allocation. That
    provider must obtain correctly estimated WHOLE-GRANT MCS/BLER from a PHY
    model (Task 3 rejects stale/mismatched links). No PHY/channel is recomputed
    by this runner. Complete `step` before advancing to the next TTI.
    """

    def __init__(
        self,
        *,
        queues: ServiceAwareTrafficManager,
        observer: ServiceAwareQoSObserver,
        decoder: ServiceAwareTransmissionEngine,
        grants: NRGrantAssembler,
        policy: Policy,
        link_provider: WholeLinkProvider,
        flow_packer: FlowPacker | None = None,
    ) -> None:
        if not isinstance(queues, ServiceAwareTrafficManager):
            raise TypeError("queues must be ServiceAwareTrafficManager")
        if not isinstance(observer, ServiceAwareQoSObserver):
            raise TypeError("observer must be ServiceAwareQoSObserver")
        if observer._queues is not queues:  # bound observer must use exactly this queue owner
            raise ValueError("Observer must belong to this cell's queue manager")
        if not isinstance(decoder, ServiceAwareTransmissionEngine):
            raise TypeError("decoder must be ServiceAwareTransmissionEngine")
        if not isinstance(grants, NRGrantAssembler):
            raise TypeError("grants must be NRGrantAssembler")
        if not callable(policy) or not callable(link_provider):
            raise TypeError("policy and link_provider must be callable")
        if flow_packer is not None and not callable(flow_packer):
            raise TypeError("flow_packer must be callable")
        if queues.has_active_tti or queues.last_completed_tti_index is not None:
            raise ValueError("Initialize runner before TTI 0")
        self.queues = queues
        self.observer = observer
        self.decoder = decoder
        self.grants = grants
        self.policy = policy
        self.link_provider = link_provider
        self.flow_packer = flow_packer
        self.metrics = ServiceAwareDownlinkMetrics(queues)
        self._next_tti = 0

    def step(
        self,
        *,
        tti_index: int,
        arrivals: Sequence[FlowPacketArrival] | None = None,
    ) -> DownlinkTTIResult:
        if type(tti_index) is not int or tti_index != self._next_tti:
            raise ValueError(f"Expected contiguous TTI {self._next_tti}")
        started = self.queues.begin_tti(tti_index=tti_index, arrivals=arrivals)
        observed = self.observer.observe()
        decision = self.policy(observed)
        if not isinstance(decision, DownlinkSchedulingDecision):
            raise TypeError("Policy must return DownlinkSchedulingDecision")
        whole_links = self.link_provider(observed, decision)
        tbs = self.grants.build(
            allocation=decision.allocation,
            queue_ue_by_candidate=decision.queue_ue_by_candidate,
            whole_links=whole_links,
        )
        if decision.planned_flow_bytes is not None:
            planned = decision.planned_flow_bytes
        elif self.flow_packer is not None:
            planned = self.flow_packer(observed, decision, tbs)
        else:
            raise ValueError("Policy must supply planned bytes or runner a flow_packer")
        transmission = self.decoder.execute_tti(
            queues=self.queues,
            transport_blocks=tbs,
            planned_flow_bytes=planned,
        )
        self.metrics.accept(started=started, observed=observed, result=transmission)
        self._next_tti += 1
        return DownlinkTTIResult(
            tti_index=tti_index,
            queue_start=started,
            pre_service_observation=observed,
            transmission=transmission,
            backlog_after_bytes=self.queues.backlog_bytes,
        )

    def report(self) -> DownlinkQoSReport:
        return self.metrics.report()

    def reset(self, *, seed: int | None = None) -> None:
        """Restart a run, including after failed policy/provider validation."""
        self.queues.reset(seed=seed, discard_active_tti=True)
        self.observer.reset()
        self.decoder.reset(seed=seed)
        self.metrics.reset()
        self._next_tti = 0
