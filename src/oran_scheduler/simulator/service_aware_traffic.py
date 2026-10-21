"""Opt-in authoritative per-flow downlink queues (research Task 1).

This module does NOT alter the legacy Nokia TrafficBufferManager or shadow
packet-QoS diagnostics. It accepts *already successful* service in bytes;
transmission failures, HARQ, and scheduling policy are deliberately out of
scope. A normal-slot call sequence is ``begin_tti`` -> ``apply_service``.

Hot-path choices:
* Integer payload bytes, not repeated floating-point queue summation.
* Cached per-flow, per-UE, and total backlogs (O(1) queries).
* FIFO ``deque`` with an ordered expiry ``deque`` (amortized O(1)/packet).
* One batched Poisson call per TTI when FTP3 generation is requested; an
  externally supplied arrival trace bypasses sampling altogether.
* No GPU transfers, PHY recomputation, or per-TTI packet-list scans.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Iterable, Literal, Mapping

from oran_scheduler.simulator.traffic import FTP3TrafficConfig


FlowKey = tuple[int, str]
ExpiryPolicy = Literal["drop", "retain"]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


@dataclass(frozen=True)
class QoSFlowSpec:
    """Local UE/flow identity and explicit QoS characteristics.

    ``priority`` and ``five_qi`` are metadata; they do NOT select a scheduler
    or imply a fixed 5QI->priority/PDB mapping. ``pdb_ttis`` is the number of
    slots INCLUDING the arrival slot in which a packet can finish on time.
    ``gbr_bps`` is metadata until a future scheduler uses it.
    """

    ue_index: int
    flow_id: str
    five_qi: int
    priority: int
    pdb_ttis: int | None = None
    gbr_bps: float | None = None
    max_queue_bytes: int | None = None
    expiry_policy: ExpiryPolicy = "drop"
    ftp3: FTP3TrafficConfig | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.ue_index, "ue_index")
        if not isinstance(self.flow_id, str) or not self.flow_id.strip():
            raise ValueError("flow_id must be a nonempty string.")
        if not isinstance(self.five_qi, int) or isinstance(self.five_qi, bool) or not 1 <= self.five_qi <= 255:
            raise ValueError("five_qi must be an integer from 1 to 255.")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) or not 1 <= self.priority <= 127:
            raise ValueError("priority must be an integer from 1 to 127.")
        if self.pdb_ttis is not None:
            _positive_int(self.pdb_ttis, "pdb_ttis")
        if self.max_queue_bytes is not None:
            _positive_int(self.max_queue_bytes, "max_queue_bytes")
        if self.gbr_bps is not None:
            if not isinstance(self.gbr_bps, (int, float)) or isinstance(self.gbr_bps, bool) or not math.isfinite(self.gbr_bps) or self.gbr_bps <= 0:
                raise ValueError("gbr_bps must be finite and positive.")
        if self.expiry_policy not in ("drop", "retain"):
            raise ValueError("expiry_policy must be 'drop' or 'retain'.")
        if self.ftp3 is not None and not isinstance(self.ftp3, FTP3TrafficConfig):
            raise TypeError("ftp3 must be an FTP3TrafficConfig.")

    @property
    def key(self) -> FlowKey:
        return (self.ue_index, self.flow_id)


@dataclass(frozen=True)
class FlowPacketArrival:
    ue_index: int
    flow_id: str
    size_bytes: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.ue_index, "ue_index")
        if not isinstance(self.flow_id, str) or not self.flow_id.strip():
            raise ValueError("flow_id must be a nonempty string.")
        _positive_int(self.size_bytes, "size_bytes")

    @property
    def key(self) -> FlowKey:
        return (self.ue_index, self.flow_id)


@dataclass
class _QueuedPacket:
    arrival_tti: int
    size_bytes: int
    remaining_bytes: int
    deadline_tti: int | None
    active: bool = True
    deadline_missed: bool = False


@dataclass
class _FlowState:
    spec: QoSFlowSpec
    packets: deque[_QueuedPacket]
    pending_deadlines: deque[_QueuedPacket]
    backlog_bytes: int = 0
    packet_count: int = 0
    arrived_packets: int = 0
    arrived_bytes: int = 0
    admitted_packets: int = 0
    delivered_bytes: int = 0
    completed_packets: int = 0
    on_time_completed_packets: int = 0
    late_completed_packets: int = 0
    overflow_dropped_packets: int = 0
    overflow_dropped_bytes: int = 0
    expired_dropped_packets: int = 0
    expired_dropped_bytes: int = 0
    deadline_misses: int = 0
    # Exact histogram updated once per completed packet, no per-TTI scan.
    completed_delay_slots: dict[int, int] = field(default_factory=dict)

    def compact_heads(self) -> None:
        while self.packets and not self.packets[0].active:
            self.packets.popleft()
        while self.pending_deadlines and not self.pending_deadlines[0].active:
            self.pending_deadlines.popleft()


@dataclass(frozen=True)
class FlowSnapshot:
    spec: QoSFlowSpec
    backlog_bytes: int
    queued_packets: int
    hol_delay_ttis: int | None
    arrived_packets: int
    arrived_bytes: int
    admitted_packets: int
    delivered_bytes: int
    completed_packets: int
    on_time_completed_packets: int
    late_completed_packets: int
    overflow_dropped_packets: int
    overflow_dropped_bytes: int
    expired_dropped_packets: int
    expired_dropped_bytes: int
    deadline_misses: int


@dataclass(frozen=True)
class FlowTTIStart:
    tti_index: int
    arrived_packets: int
    arrived_bytes: int
    admitted_packets: int
    admitted_bytes: int
    overflow_dropped_packets: int
    overflow_dropped_bytes: int
    expired_dropped_packets: int
    expired_dropped_bytes: int
    new_deadline_misses: int
    backlog_bytes: int


@dataclass(frozen=True)
class FlowTTIService:
    tti_index: int
    requested_service_bytes: int
    delivered_bytes: int
    unused_requested_service_bytes: int
    completed_packets: int
    on_time_completed_packets: int
    late_completed_packets: int
    backlog_bytes: int
    delivered_bytes_by_ue: tuple[int, ...]


class ServiceAwareTrafficManager:
    """One authoritative owner of queued packets for each local UE/flow.

    Per-flow grants in ``apply_service`` explicitly determine which flow
    receives bytes; THIS CLASS DOES NOT CHOOSE A SCHEDULING POLICY.
    The optional per-UE cap prevents multiple flows claiming the same PHY
    service. A PHY success/failure model must provide the service budgets
    in the future (Task 2); nothing is inferred from TBLER in Task 1.
    """

    def __init__(
        self,
        *,
        num_ues: int,
        flows: Iterable[QoSFlowSpec],
        tti_duration_s: float,
        seed: int = 0,
    ) -> None:
        _positive_int(num_ues, "num_ues")
        if not isinstance(tti_duration_s, (int, float)) or isinstance(tti_duration_s, bool) or not math.isfinite(tti_duration_s) or tti_duration_s <= 0:
            raise ValueError("tti_duration_s must be finite and positive.")
        _nonnegative_int(seed, "seed")
        self.num_ues = num_ues
        self.tti_duration_s = float(tti_duration_s)
        self._seed = seed
        specs = tuple(flows)
        if not specs:
            raise ValueError("At least one QoS flow is required.")
        self._states: dict[FlowKey, _FlowState] = {}
        for spec in specs:
            if not isinstance(spec, QoSFlowSpec):
                raise TypeError("flows must contain QoSFlowSpec objects.")
            if spec.ue_index >= num_ues:
                raise ValueError("Flow UE index exceeds num_ues.")
            if spec.key in self._states:
                raise ValueError(f"Duplicate QoS flow: {spec.key!r}.")
            if spec.ftp3 is not None and not math.isclose(
                spec.ftp3.tti_duration_s, self.tti_duration_s, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("FTP3 TTI duration must match the manager.")
            self._states[spec.key] = _FlowState(spec, deque(), deque())

        self._keys = tuple(self._states)
        self._per_ue_backlog = [0] * num_ues
        self._backlog_bytes = 0
        self._active_tti: int | None = None
        self._last_completed_tti: int | None = None

        # Traffic sampling is independent from PHY RNG and is skipped when
        # the caller provides an arrival trace, preventing double sampling.
        self._auto_specs = tuple(s for s in specs if s.ftp3 is not None)
        self._poisson_generator = None
        self._poisson_means = None
        if self._auto_specs:
            import torch
            self._poisson_means = torch.tensor(
                [s.ftp3.expected_packets_per_tti for s in self._auto_specs],
                dtype=torch.float64,
                device="cpu",
            )
            self._poisson_generator = torch.Generator(device="cpu")
            self._poisson_generator.manual_seed(seed)

    @property
    def has_active_tti(self) -> bool:
        return self._active_tti is not None

    @property
    def backlog_bytes(self) -> int:
        return self._backlog_bytes

    @property
    def last_completed_tti_index(self) -> int | None:
        return self._last_completed_tti

    @property
    def active_tti_index(self) -> int | None:
        """TTI whose arrivals have been admitted but service is not finalized."""
        return self._active_tti

    def configured_flows(self) -> tuple[QoSFlowSpec, ...]:
        """Immutable configured flow descriptions, in stable insertion order.

        This is O(number of configured flows), with no queued-packet scan.
        Intended for constructing a long-lived per-cell QoS observer once.
        """
        return tuple(state.spec for state in self._states.values())

    def ue_backlog_bytes(self) -> tuple[int, ...]:
        """Cheap snapshot of cached aggregate UE backlog (no packet scan)."""
        return tuple(self._per_ue_backlog)

    def flow_snapshot(self, ue_index: int, flow_id: str) -> FlowSnapshot:
        state = self._states[(ue_index, flow_id)]
        state.compact_heads()
        head = state.packets[0] if state.packets else None
        clock = self._active_tti if self._active_tti is not None else self._last_completed_tti
        delay = None if head is None or clock is None else clock - head.arrival_tti + 1
        return FlowSnapshot(
            spec=state.spec,
            backlog_bytes=state.backlog_bytes,
            queued_packets=state.packet_count,
            hol_delay_ttis=delay,
            arrived_packets=state.arrived_packets,
            arrived_bytes=state.arrived_bytes,
            admitted_packets=state.admitted_packets,
            delivered_bytes=state.delivered_bytes,
            completed_packets=state.completed_packets,
            on_time_completed_packets=state.on_time_completed_packets,
            late_completed_packets=state.late_completed_packets,
            overflow_dropped_packets=state.overflow_dropped_packets,
            overflow_dropped_bytes=state.overflow_dropped_bytes,
            expired_dropped_packets=state.expired_dropped_packets,
            expired_dropped_bytes=state.expired_dropped_bytes,
            deadline_misses=state.deadline_misses,
        )

    def completed_delay_histogram(self, ue_index: int, flow_id: str) -> tuple[tuple[int, int], ...]:
        """Exact completed-packet sojourn times, in slots including arrival.

        For reporting only: O(distinct completed delay values); NOT used in
        scheduling observations or the per-TTI hot path.
        """
        key = (ue_index, flow_id)
        if key not in self._states:
            raise ValueError(f"Unknown flow: {key!r}.")
        return tuple(sorted(self._states[key].completed_delay_slots.items()))

    def _adjust_backlog(self, state: _FlowState, delta: int) -> None:
        state.backlog_bytes += delta
        self._per_ue_backlog[state.spec.ue_index] += delta
        self._backlog_bytes += delta
        if min(state.backlog_bytes, self._per_ue_backlog[state.spec.ue_index], self._backlog_bytes) < 0:
            raise AssertionError("Queue backlog conservation violated.")

    def sample_ftp3_arrivals(self) -> tuple[FlowPacketArrival, ...]:
        """Batched CPU Poisson sampler; call only when external trace absent."""
        if self._poisson_means is None:
            return ()
        import torch
        counts = torch.poisson(self._poisson_means, generator=self._poisson_generator)
        arrivals: list[FlowPacketArrival] = []
        for spec, count in zip(self._auto_specs, counts.tolist()):
            if count:
                arrivals.extend(
                    FlowPacketArrival(spec.ue_index, spec.flow_id, spec.ftp3.packet_size_bytes)
                    for _ in range(int(count))
                )
        return tuple(arrivals)

    def begin_tti(
        self,
        *,
        tti_index: int,
        arrivals: Iterable[FlowPacketArrival] | None = None,
    ) -> FlowTTIStart:
        _nonnegative_int(tti_index, "tti_index")
        if self._active_tti is not None:
            raise RuntimeError("Previous TTI must be served first.")
        if self._last_completed_tti is not None and tti_index != self._last_completed_tti + 1:
            raise ValueError("TTI indices must be contiguous.")
        if arrivals is None:
            arrivals = self.sample_ftp3_arrivals()
        # Validate everything BEFORE changing a queue; a malformed event must
        # not cause partial mutation of a TTI.
        events = tuple(arrivals)
        for event in events:
            if not isinstance(event, FlowPacketArrival):
                raise TypeError("arrivals must contain FlowPacketArrival objects.")
            if event.key not in self._states:
                raise ValueError(f"Unknown flow: {event.key!r}.")

        expired_n = expired_b = missed = 0
        for state in self._states.values():
            state.compact_heads()
            deadlines = state.pending_deadlines
            while deadlines and deadlines[0].deadline_tti < tti_index:
                packet = deadlines.popleft()
                if not packet.active:
                    continue
                if packet.deadline_missed:
                    raise AssertionError("Deadline was counted twice.")
                packet.deadline_missed = True
                state.deadline_misses += 1
                missed += 1
                if state.spec.expiry_policy == "drop":
                    packet.active = False
                    state.packet_count -= 1
                    state.expired_dropped_packets += 1
                    state.expired_dropped_bytes += packet.remaining_bytes
                    expired_n += 1
                    expired_b += packet.remaining_bytes
                    self._adjust_backlog(state, -packet.remaining_bytes)
            state.compact_heads()

        admitted_n = admitted_b = overflow_n = overflow_b = arrived_b = 0
        for event in events:
            state = self._states[event.key]
            size = event.size_bytes
            arrived_b += size
            state.arrived_packets += 1
            state.arrived_bytes += size
            if state.spec.max_queue_bytes is not None and state.backlog_bytes + size > state.spec.max_queue_bytes:
                state.overflow_dropped_packets += 1
                state.overflow_dropped_bytes += size
                overflow_n += 1
                overflow_b += size
                continue
            deadline = (None if state.spec.pdb_ttis is None else tti_index + state.spec.pdb_ttis - 1)
            packet = _QueuedPacket(tti_index, size, size, deadline)
            state.packets.append(packet)
            if deadline is not None:
                state.pending_deadlines.append(packet)
            state.packet_count += 1
            state.admitted_packets += 1
            self._adjust_backlog(state, size)
            admitted_n += 1
            admitted_b += size

        self._active_tti = tti_index
        return FlowTTIStart(
            tti_index=tti_index,
            arrived_packets=len(events), arrived_bytes=arrived_b,
            admitted_packets=admitted_n, admitted_bytes=admitted_b,
            overflow_dropped_packets=overflow_n, overflow_dropped_bytes=overflow_b,
            expired_dropped_packets=expired_n, expired_dropped_bytes=expired_b,
            new_deadline_misses=missed, backlog_bytes=self._backlog_bytes,
        )

    def apply_service(
        self,
        *,
        service_bytes_by_flow: Mapping[FlowKey, int],
        ue_success_capacity_bytes: Iterable[int] | None = None,
    ) -> FlowTTIService:
        """Consume only explicitly supplied successful per-flow payload bytes.

        Requested budgets can exceed actual queue size; actual delivery is
        clamped to queued bytes. When ``ue_success_capacity_bytes`` is given,
        the sum of requested grants for a UE cannot exceed its decoded payload
        budget. No HARQ/BLER success is inferred by this Task-1 queue manager.
        """
        if self._active_tti is None:
            raise RuntimeError("Call begin_tti() first.")
        if not isinstance(service_bytes_by_flow, Mapping):
            raise TypeError("service_bytes_by_flow must be a mapping.")
        requested_by_ue = [0] * self.num_ues
        for key, amount in service_bytes_by_flow.items():
            if key not in self._states:
                raise ValueError(f"Unknown service flow: {key!r}.")
            _nonnegative_int(amount, "service_bytes_by_flow amount")
            requested_by_ue[key[0]] += amount
        if ue_success_capacity_bytes is not None:
            caps = tuple(ue_success_capacity_bytes)
            if len(caps) != self.num_ues:
                raise ValueError("ue_success_capacity_bytes must have one entry per UE.")
            for ue, capacity in enumerate(caps):
                _nonnegative_int(capacity, "ue_success_capacity_bytes entry")
                if requested_by_ue[ue] > capacity:
                    raise ValueError("Sum of per-flow grants exceeds a UE's successful capacity.")

        tti_index = self._active_tti
        delivered_by_ue = [0] * self.num_ues
        completed = on_time = late = delivered = 0
        requested = sum(requested_by_ue)
        for key, budget in service_bytes_by_flow.items():
            state = self._states[key]
            state.compact_heads()
            while budget > 0 and state.packets:
                packet = state.packets[0]
                take = min(budget, packet.remaining_bytes)
                packet.remaining_bytes -= take
                budget -= take
                delivered += take
                delivered_by_ue[key[0]] += take
                state.delivered_bytes += take
                self._adjust_backlog(state, -take)
                if packet.remaining_bytes == 0:
                    state.packets.popleft()
                    packet.active = False
                    state.packet_count -= 1
                    state.completed_packets += 1
                    delay_slots = tti_index - packet.arrival_tti + 1
                    state.completed_delay_slots[delay_slots] = (
                        state.completed_delay_slots.get(delay_slots, 0) + 1
                    )
                    completed += 1
                    if packet.deadline_missed:
                        state.late_completed_packets += 1
                        late += 1
                    else:
                        state.on_time_completed_packets += 1
                        on_time += 1
                    state.compact_heads()
            state.compact_heads()

        self._last_completed_tti = tti_index
        self._active_tti = None
        return FlowTTIService(
            tti_index=tti_index,
            requested_service_bytes=requested,
            delivered_bytes=delivered,
            unused_requested_service_bytes=requested - delivered,
            completed_packets=completed,
            on_time_completed_packets=on_time,
            late_completed_packets=late,
            backlog_bytes=self._backlog_bytes,
            delivered_bytes_by_ue=tuple(delivered_by_ue),
        )

    def reset(self, *, seed: int | None = None, discard_active_tti: bool = False) -> None:
        """Clear episode; explicit opt-in permits aborting a failed active TTI.

        discard_active_tti=True is intended ONLY for abandoning/replaying the
        entire episode after an invalid policy/link-provider call. It must not
        be used as an ordinary queue rollback or per-TTI skip.
        """
        if type(discard_active_tti) is not bool:
            raise TypeError("discard_active_tti must be a bool")
        if self._active_tti is not None and not discard_active_tti:
            raise RuntimeError("Cannot reset in the middle of a TTI.")
        if seed is not None:
            _nonnegative_int(seed, "seed")
            self._seed = seed
        for state in self._states.values():
            state.packets.clear()
            state.pending_deadlines.clear()
            state.completed_delay_slots.clear()
            for name in (
                "backlog_bytes", "packet_count", "arrived_packets", "arrived_bytes",
                "admitted_packets", "delivered_bytes", "completed_packets",
                "on_time_completed_packets", "late_completed_packets",
                "overflow_dropped_packets", "overflow_dropped_bytes",
                "expired_dropped_packets", "expired_dropped_bytes", "deadline_misses",
            ):
                setattr(state, name, 0)
        self._per_ue_backlog = [0] * self.num_ues
        self._backlog_bytes = 0
        self._active_tti = self._last_completed_tti = None
        if self._poisson_generator is not None:
            self._poisson_generator.manual_seed(self._seed)
