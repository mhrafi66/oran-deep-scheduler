"""Task 4: local, pre-scheduling QoS observations for service-aware DL.

Opt-in interface: no policy, radio calculation, GPU copy, second packet queue,
or mutation of packet service. Call *once* after ``queues.begin_tti`` and
before the scheduler chooses any transmissions for that TTI.

All queue information comes from Task 1's O(1) per-flow snapshot and cached
per-UE backlog. A separate bounded, O(1)/GBR-flow delivered-byte ledger
accounts only for *past* completed TTIs, never the slot under decision.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math

from oran_scheduler.simulator.service_aware_traffic import (
    ExpiryPolicy,
    FlowKey,
    FlowSnapshot,
    QoSFlowSpec,
    ServiceAwareTrafficManager,
)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class QoSFlowObservation:
    """Per-flow information available to a local scheduler *before* service.

    ``deadline_slack_ttis`` is the HOL packet's number of FUTURE slots after
    this slot in which it can still complete on time. 0 means THIS slot is
    its final opportunity; negative means a late HOL packet retained under
    ``expiry_policy='retain'``. None means no pending deadline-bearing HOL.

    ``gbr_deficit_bytes`` is a bounded scheduling-pressure proxy, NOT a 3GPP
    GBR admission-control or compliance verdict. It is capped by backlog.
    """

    ue_index: int
    flow_id: str
    five_qi: int
    priority: int
    expiry_policy: ExpiryPolicy
    backlog_bytes: int
    queued_packets: int
    hol_delay_ttis: int | None
    deadline_slack_ttis: int | None
    gbr_bps: float | None
    gbr_window_backlogged_ttis: int
    gbr_window_delivered_bytes: int
    gbr_target_bytes_in_window: float
    gbr_deficit_bytes: int

    @property
    def key(self) -> FlowKey:
        return (self.ue_index, self.flow_id)


@dataclass(frozen=True, slots=True)
class QoSUEObservation:
    """Aggregate of local flows belonging to one UE; no radio data duplicated."""

    ue_index: int
    backlog_bytes: int
    active_flows: int
    highest_priority: int | None  # Lower number = higher priority
    max_hol_delay_ttis: int | None
    min_deadline_slack_ttis: int | None
    gbr_deficit_bytes: int


@dataclass(frozen=True, slots=True)
class QoSCellObservation:
    """Immutable, deterministic ordering: configured flows and UE 0..N-1."""

    tti_index: int
    tti_duration_s: float
    flows: tuple[QoSFlowObservation, ...]
    ues: tuple[QoSUEObservation, ...]
    total_backlog_bytes: int


@dataclass(slots=True)
class _GBRHistory:
    # Only retain the last (window_ttis-1) completed slots. The current slot
    # is backlogged if it has pending bytes, but its delivered bytes are not
    # known until NEXT observation, so they must never enter this snapshot.
    completed: deque[tuple[bool, int]] = field(default_factory=deque)
    prior_delivered_bytes: int = 0
    prior_backlogged: bool = False
    backlogged_slots: int = 0
    delivered_bytes: int = 0

    def incorporate_last_slot(self, *, delivered_delta: int, horizon: int) -> None:
        previous = (self.prior_backlogged, delivered_delta)
        self.completed.append(previous)
        self.backlogged_slots += int(previous[0])
        self.delivered_bytes += previous[1]
        while len(self.completed) > horizon:
            old_backlogged, old_bytes = self.completed.popleft()
            self.backlogged_slots -= int(old_backlogged)
            self.delivered_bytes -= old_bytes


class ServiceAwareQoSObserver:
    """Read-only queue adapter and bounded GBR observation accounting.

    Each observer is bound to ONE per-cell traffic manager and must run once
    per TTI: ``queues.begin_tti(...)`` -> ``observer.observe()`` -> external
    scheduler / Task 2 transmission -> next TTI. Construct before TTI 0.

    GBR pressure uses prior successfully delivered bytes and the number of
    *backlogged* slots in a configurable trailing window INCLUDING the
    current slot. Idle slots earn no target service and do not accumulate
    fictional long-term debt. The unserved target is capped at the current
    backlog. No HARQ, flow selection, or QoS policy is implemented here.

    CPU complexity: O(configured flows + UEs) per observation; backlog and
    head-of-line operations are O(1)/flow; history updates are amortized
    O(1)/GBR flow. No scan of queued packets, PHY or GPU data.
    """

    def __init__(
        self,
        *,
        queues: ServiceAwareTrafficManager,
        gbr_window_ttis: int = 100,
    ) -> None:
        if not isinstance(queues, ServiceAwareTrafficManager):
            raise TypeError("queues must be a ServiceAwareTrafficManager")
        _positive_int(gbr_window_ttis, "gbr_window_ttis")
        if queues.has_active_tti or queues.last_completed_tti_index is not None:
            raise ValueError("Construct QoS observer before TTI 0 (or reset queues)")

        self._queues = queues
        self._specs: tuple[QoSFlowSpec, ...] = queues.configured_flows()
        self._window_ttis = gbr_window_ttis
        self._gbr_histories: dict[FlowKey, _GBRHistory] = {}
        for spec in self._specs:
            if spec.gbr_bps is not None:
                # Fail before running if expected per-window float would overflow.
                per_tti_target = spec.gbr_bps * queues.tti_duration_s / 8.0
                if not math.isfinite(per_tti_target * gbr_window_ttis):
                    raise ValueError("GBR target cannot be represented as finite bytes")
                self._gbr_histories[spec.key] = _GBRHistory()
        self._last_observed_tti: int | None = None

    @property
    def gbr_window_ttis(self) -> int:
        return self._window_ttis

    def reset(self) -> None:
        """Reset ONLY observer history; caller separately resets the queues.

        Do not reset while either an observer TTI or queue TTI is active.
        Reset queues before starting the next episode.
        """
        if self._queues.has_active_tti or self._queues.last_completed_tti_index is not None:
            raise RuntimeError("Reset queues first; then call observer.reset()")
        self._gbr_histories = {key: _GBRHistory() for key in self._gbr_histories}
        self._last_observed_tti = None

    def observe(self) -> QoSCellObservation:
        """Generate a PRE-SERVICE snapshot; exactly once for each active TTI.

        No changes to packet service or counters, and no future outcomes.
        Task 1's flow_snapshot may lazily compact inactive deque heads.
        Cached per-flow backlog and HOL are read once, and existing cumulative
        delivered-byte counters are differenced to account for last slot.
        """
        tti = self._queues.active_tti_index
        if tti is None:
            raise RuntimeError("Call queues.begin_tti() before observe()")
        expected = 0 if self._last_observed_tti is None else self._last_observed_tti + 1
        if tti != expected:
            raise RuntimeError(f"QoS observation expected TTI {expected}; got TTI {tti}")
        if self._queues.last_completed_tti_index != (None if tti == 0 else tti - 1):
            raise RuntimeError("Queue service history is not contiguous")

        # Pre-validate ALL readings before advancing the GBR ledger.
        snaps: tuple[FlowSnapshot, ...] = tuple(
            self._queues.flow_snapshot(spec.ue_index, spec.flow_id)
            for spec in self._specs
        )
        cached_per_ue = self._queues.ue_backlog_bytes()
        observed_per_ue = [0] * self._queues.num_ues
        for spec, snap in zip(self._specs, snaps):
            if snap.spec != spec:
                raise AssertionError("QoS flow configuration changed during episode")
            observed_per_ue[spec.ue_index] += snap.backlog_bytes
            history = self._gbr_histories.get(spec.key)
            if history is not None:
                if snap.delivered_bytes < history.prior_delivered_bytes:
                    raise RuntimeError("Delivered counters were reset; reset the observer as well")
                if tti == 0 and snap.delivered_bytes != 0:
                    raise RuntimeError("QoS observer must start with empty service history")
        if tuple(observed_per_ue) != cached_per_ue or sum(cached_per_ue) != self._queues.backlog_bytes:
            raise AssertionError("Observed per-flow and cached UE backlogs disagree")

        gbr_flow_views: dict[FlowKey, tuple[int, int, float, int]] = {}
        for spec, snap in zip(self._specs, snaps):
            hist = self._gbr_histories.get(spec.key)
            if hist is None:
                continue
            if tti > 0:
                hist.incorporate_last_slot(
                    delivered_delta=snap.delivered_bytes - hist.prior_delivered_bytes,
                    horizon=self._window_ttis - 1,
                )
            backlogged = snap.backlog_bytes > 0
            active_count = hist.backlogged_slots + int(backlogged)
            target = spec.gbr_bps * self._queues.tti_duration_s * active_count / 8.0
            # Compare successful prior service with current-window target.
            # Avoid reporting more currently actionable deficit than can fit
            # in the flow's authoritative remaining packet backlog.
            shortfall = max(0.0, target - hist.delivered_bytes)
            deficit = min(snap.backlog_bytes, math.ceil(max(0.0, shortfall - 1e-10)))
            gbr_flow_views[spec.key] = (active_count, hist.delivered_bytes, target, deficit)
            hist.prior_delivered_bytes = snap.delivered_bytes
            hist.prior_backlogged = backlogged

        flow_views: list[QoSFlowObservation] = []
        ue_active = [0] * self._queues.num_ues
        ue_priority: list[int | None] = [None] * self._queues.num_ues
        ue_hol: list[int | None] = [None] * self._queues.num_ues
        ue_slack: list[int | None] = [None] * self._queues.num_ues
        ue_gbr_deficit = [0] * self._queues.num_ues

        for spec, snap in zip(self._specs, snaps):
            delay = snap.hol_delay_ttis
            slack = None if delay is None or spec.pdb_ttis is None else spec.pdb_ttis - delay
            gbr_slots, gbr_served, gbr_target, deficit = gbr_flow_views.get(
                spec.key, (0, 0, 0.0, 0)
            )
            flow_views.append(QoSFlowObservation(
                ue_index=spec.ue_index,
                flow_id=spec.flow_id,
                five_qi=spec.five_qi,
                priority=spec.priority,
                expiry_policy=spec.expiry_policy,
                backlog_bytes=snap.backlog_bytes,
                queued_packets=snap.queued_packets,
                hol_delay_ttis=delay,
                deadline_slack_ttis=slack,
                gbr_bps=spec.gbr_bps,
                gbr_window_backlogged_ttis=gbr_slots,
                gbr_window_delivered_bytes=gbr_served,
                gbr_target_bytes_in_window=gbr_target,
                gbr_deficit_bytes=deficit,
            ))
            if snap.backlog_bytes:
                ue = spec.ue_index
                ue_active[ue] += 1
                ue_priority[ue] = spec.priority if ue_priority[ue] is None else min(ue_priority[ue], spec.priority)
                if delay is not None:
                    ue_hol[ue] = delay if ue_hol[ue] is None else max(ue_hol[ue], delay)
                if slack is not None:
                    ue_slack[ue] = slack if ue_slack[ue] is None else min(ue_slack[ue], slack)
                ue_gbr_deficit[ue] += deficit

        ue_views = tuple(
            QoSUEObservation(
                ue_index=ue,
                backlog_bytes=cached_per_ue[ue],
                active_flows=ue_active[ue],
                highest_priority=ue_priority[ue],
                max_hol_delay_ttis=ue_hol[ue],
                min_deadline_slack_ttis=ue_slack[ue],
                gbr_deficit_bytes=ue_gbr_deficit[ue],
            )
            for ue in range(self._queues.num_ues)
        )
        self._last_observed_tti = tti
        return QoSCellObservation(
            tti_index=tti,
            tti_duration_s=self._queues.tti_duration_s,
            flows=tuple(flow_views),
            ues=ue_views,
            total_backlog_bytes=self._queues.backlog_bytes,
        )
