"""Task 6: opt-in stop-and-wait downlink HARQ abstraction (one process / UE).

This component replaces ONLY the Task-2 decoder in a Task-5 cell runner.
It retains ONE copy of packet bytes in Task-1 authoritative queues until a
successful physical TB decode. There is no duplicate HARQ data queue.

HARQ is intentionally SYSTEM LEVEL, not NR bit-exact:
* Delayed HARQ process release / NACK availability (slot-based); at most one
  outstanding process per UE (NOT full NR parallel HARQ).
* Retransmissions reuse exactly the same per-flow payload and TB payload size.
* Each attempt uses its newly supplied whole-TB TBLER; no invented combining.
* The same UE cannot transmit other data while its TB awaits feedback.
* Exhausted HARQ process releases UE, but keeps undecoded data in the
  authoritative queue (RLC/recovery policy is beyond this abstraction).
* Deadline-drop flows can cancel an outstanding failed process. A drop in
  ANY reserved flow conservatively invalidates that process; this may lose a
  usable retransmission opportunity, but never delivers a different packet.
* A SUCCESSFULLY decoded TB releases its bytes immediately, but HARQ process
  admission remains blocked until ACK feedback becomes available. Because
  aggregate queue backlog is visible in the QoS observation, an observer
  could infer a decode outcome before ACK. Do NOT claim realistic feedback
  information hiding or throughput fidelity from this model.
* Multiple parallel HARQ processes per UE, soft combining, and RLC retransmit
  semantics are NOT supported. Use only as an isolated HARQ diagnostic; do
  not compare against a full NR MAC throughput baseline without upgrading it.

No channel/CSI/GPU work occurs here. A local scheduler can query `statuses()`
AFTER `queues.begin_tti` / the runner's prepare hook to reserve resources for
ready retransmissions. The engine refuses a new TB on a blocked UE. All
validation precedes RNG draws and queue mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from oran_scheduler.simulator.service_aware_traffic import (
    FlowKey, ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_transmission import (
    DownlinkTB, ServiceAwareTransmissionEngine, TransmissionTTIResult,
)


@dataclass(frozen=True)
class HARQStatus:
    ue_index: int
    attempts: int
    max_transmissions: int
    feedback_ready_tti: int
    state: str  # "awaiting_feedback" or "retransmit"
    payload_capacity_bytes: int
    planned_flow_bytes: tuple[tuple[FlowKey, int], ...]


@dataclass(frozen=True)
class HARQCounters:
    initial_transmissions: int
    retransmissions: int
    ack_received: int
    nack_received: int
    processes_exhausted: int
    cancelled_by_expiry: int
    pending_processes: int


@dataclass
class _Process:
    ue: int
    cap: int
    flow_plan: tuple[tuple[FlowKey, int], ...]
    expiry_baseline: tuple[tuple[FlowKey, int], ...]
    attempts: int
    feedback_ready_tti: int
    decoded: bool


class ServiceAwareHARQEngine(ServiceAwareTransmissionEngine):
    """Drop-in Task-2 decoder with conservative stop-and-wait semantics.

    `prepare_tti` is called by the opt-in Task-5 hook after arrivals/expiry
    and before scheduling. Policy can then read `statuses()` to select an
    exactly matching retransmission. No policy is embedded in this engine.
    """

    def __init__(self, *, seed: int = 0, feedback_delay_ttis: int = 2,
                 max_transmissions: int = 3) -> None:
        if type(feedback_delay_ttis) is not int or feedback_delay_ttis < 1:
            raise ValueError("feedback_delay_ttis must be >= 1")
        if type(max_transmissions) is not int or max_transmissions < 1:
            raise ValueError("max_transmissions must be >= 1")
        super().__init__(seed=seed)
        self.feedback_delay_ttis = feedback_delay_ttis
        self.max_transmissions = max_transmissions
        self._pending: dict[int, _Process] = {}
        self._bound_queues: ServiceAwareTrafficManager | None = None
        self._prepared_tti: int | None = None
        self._initial = self._retransmissions = 0
        self._acks = self._nacks = self._exhausted = self._cancelled = 0

    def reset(self, *, seed: int | None = None) -> None:
        super().reset(seed=seed)
        self._pending.clear()
        self._bound_queues = None
        self._prepared_tti = None
        self._initial = self._retransmissions = 0
        self._acks = self._nacks = self._exhausted = self._cancelled = 0

    def counters(self) -> HARQCounters:
        return HARQCounters(self._initial, self._retransmissions, self._acks,
                            self._nacks, self._exhausted, self._cancelled,
                            len(self._pending))

    def statuses(self) -> tuple[HARQStatus, ...]:
        """Locally available feedback only: no early ACK/NACK disclosure."""
        return tuple(
            HARQStatus(ue_index=p.ue, attempts=p.attempts,
                       max_transmissions=self.max_transmissions,
                       feedback_ready_tti=p.feedback_ready_tti,
                       state="retransmit" if self._prepared_tti is not None
                           and self._prepared_tti >= p.feedback_ready_tti
                           and not p.decoded else "awaiting_feedback",
                       payload_capacity_bytes=p.cap,
                       planned_flow_bytes=p.flow_plan)
            for _, p in sorted(self._pending.items())
        )

    def prepare_tti(self, *, queues: ServiceAwareTrafficManager) -> None:
        if not isinstance(queues, ServiceAwareTrafficManager):
            raise TypeError("queues must be a ServiceAwareTrafficManager")
        tti = queues.active_tti_index
        if tti is None:
            raise RuntimeError("prepare_tti requires begun queue slot")
        if self._bound_queues is not None and queues is not self._bound_queues:
            raise ValueError("HARQ engine cannot be shared across queue managers")
        if self._prepared_tti is not None and tti != self._prepared_tti + 1:
            raise RuntimeError("HARQ prepare_tti called out of order or twice")
        if queues.last_completed_tti_index != (None if tti == 0 else tti - 1):
            raise RuntimeError("Queue HARQ lifecycle not contiguous")
        for ue, p in tuple(self._pending.items()):
            if not p.decoded:
                # No service to this UE while awaiting feedback. A local flow
                # expiry invalidates the original payload identity; conservatively
                # cancel rather than retrying a newer packet with the same byte count.
                cancelled = False
                reserved = dict(p.flow_plan)
                for key, base_expired in p.expiry_baseline:
                    snapshot = queues.flow_snapshot(*key)
                    if (snapshot.expired_dropped_bytes != base_expired or
                            snapshot.backlog_bytes < reserved[key]):
                        cancelled = True
                        break
                if cancelled:
                    del self._pending[ue]
                    self._cancelled += 1
                    continue
            if tti >= p.feedback_ready_tti:
                if p.decoded:
                    self._acks += 1
                    del self._pending[ue]
                elif p.attempts >= self.max_transmissions:
                    self._nacks += 1
                    self._exhausted += 1
                    del self._pending[ue]
                elif tti == p.feedback_ready_tti:
                    self._nacks += 1
        self._prepared_tti = tti
        self._bound_queues = queues

    def execute_tti(
        self, *, queues: ServiceAwareTrafficManager,
        transport_blocks: Sequence[DownlinkTB],
        planned_flow_bytes: Mapping[FlowKey, int],
    ) -> TransmissionTTIResult:
        if queues is not self._bound_queues:
            raise ValueError("HARQ engine is bound to a different queue manager")
        if self._prepared_tti != queues.active_tti_index or not queues.has_active_tti:
            raise RuntimeError("Call prepare_tti after queue arrivals, before HARQ transmission")
        if not isinstance(planned_flow_bytes, Mapping):
            raise TypeError("planned_flow_bytes must be a mapping")
        if not isinstance(transport_blocks, (tuple, list)):
            raise TypeError("transport_blocks must be list or tuple")
        by_ue: dict[int, DownlinkTB] = {}
        plans: dict[int, dict[FlowKey, int]] = {}
        for tb in transport_blocks:
            if not isinstance(tb, DownlinkTB):
                raise TypeError("transport_blocks must contain DownlinkTB")
            if tb.ue_index in by_ue:
                raise ValueError("At most one TB per UE per slot")
            by_ue[tb.ue_index] = tb
        for key, amount in planned_flow_bytes.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("Invalid per-flow payload key")
            if type(amount) is not int or amount < 0:
                raise ValueError("Planned payload must be a nonnegative integer")
            if amount:
                plans.setdefault(key[0], {})[key] = amount
        for ue, tb in by_ue.items():
            p = self._pending.get(ue)
            if p is None:
                continue
            if (self._prepared_tti < p.feedback_ready_tti or p.decoded):
                raise ValueError("UE is waiting for HARQ feedback; new TB prohibited")
            if (tb.payload_capacity_bytes != p.cap or
                    tuple(sorted(plans.get(ue, {}).items())) != p.flow_plan):
                raise ValueError("HARQ retransmission must preserve TB size and exact flow payload")
        # Base engine handles the full TB/queue validation, consumes exactly
        # one RNG draw per TB and applies successful bytes only once.
        result = super().execute_tti(
            queues=queues, transport_blocks=transport_blocks,
            planned_flow_bytes=planned_flow_bytes,
        )
        for outcome in result.tb_outcomes:
            ue = outcome.ue_index
            p = self._pending.get(ue)
            plan = tuple(sorted(plans.get(ue, {}).items()))
            # Padding-only TBs carry no queued payload and need no HARQ process.
            if not plan:
                continue
            if p is None:
                self._initial += 1
                baseline = tuple((key, queues.flow_snapshot(*key).expired_dropped_bytes)
                                 for key, _ in plan)
                self._pending[ue] = _Process(
                    ue, outcome.scheduled_capacity_bytes, plan, baseline,
                    attempts=1,
                    feedback_ready_tti=self._prepared_tti + self.feedback_delay_ttis,
                    decoded=outcome.decoded,
                )
            else:
                self._retransmissions += 1
                p.attempts += 1
                p.decoded = outcome.decoded
                p.feedback_ready_tti = self._prepared_tti + self.feedback_delay_ttis
        return result
