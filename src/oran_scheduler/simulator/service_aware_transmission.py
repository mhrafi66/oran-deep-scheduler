"""Task 2: opt-in downlink TB decoding and successful packet service.

The authoritative per-flow packet queue is in ``service_aware_traffic.py``.
This module takes *precomputed TB-level* payload capacities and error
probabilities, samples one decode outcome per scheduled UE/TB, and hands only
successfully decoded, explicitly assigned flow bytes to that queue.

No Sionna calls, no channel reevaluation, no hidden flow scheduler, no HARQ.
The optional existing-PHY adapter is valid only when an allocated UE has
exactly ONE RBG: per-RBG TBLER is not whole-TB BLER for multi-RBG grants.

Local RNG is independent of the traffic/PHY RNG. One uniform is drawn for each
scheduled TB (including BLER=0 or 1), allowing paired-seed comparisons to
consume a stable number of samples under different channel qualities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence, TYPE_CHECKING

from oran_scheduler.simulator.service_aware_traffic import (
    FlowKey,
    FlowTTIService,
    ServiceAwareTrafficManager,
)

if TYPE_CHECKING:
    from oran_scheduler.phy.schedule_evaluator import CellAllocationEvaluationData


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _probability(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{label} must be a finite probability in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class DownlinkTB:
    """One downlink transport block destined to a local UE this slot.

    ``payload_capacity_bytes`` is a nonnegative integer payload capacity for
    THIS COMPLETE TB, after rate matching and overhead; it is not per-RBG
    capacity and must not include duplicate codeword/layer bytes.

    ``tbler`` is the probability that THIS COMPLETE TB fails to decode.
    It must come from an explicit whole-TB link abstraction. It cannot in
    general be obtained by averaging/multiplying per-RBG BLER values.

    Current first-stage contract: at most one TB per UE per TTI. A future
    two-codeword model must extend this contract explicitly rather than
    placing duplicate UE entries here.
    """

    ue_index: int
    payload_capacity_bytes: int
    tbler: float

    def __post_init__(self) -> None:
        _nonnegative_integer(self.ue_index, "ue_index")
        if _nonnegative_integer(self.payload_capacity_bytes, "payload_capacity_bytes") == 0:
            raise ValueError("Scheduled TB payload capacity must be positive")
        _probability(self.tbler, "tbler")


@dataclass(frozen=True)
class TBDecodeOutcome:
    ue_index: int
    scheduled_capacity_bytes: int
    tbler: float
    decoded: bool
    decoded_capacity_bytes: int


@dataclass(frozen=True)
class TransmissionTTIResult:
    tti_index: int
    tb_outcomes: tuple[TBDecodeOutcome, ...]
    scheduled_capacity_bytes: int
    decoded_capacity_bytes: int
    failed_capacity_bytes: int
    planned_flow_payload_bytes: int
    decoded_flow_payload_bytes: int
    traffic_service: FlowTTIService
    unused_decoded_capacity_bytes: int


class ServiceAwareTransmissionEngine:
    """A locally-seeded, lightweight TB decode -> per-flow delivery adapter.

    Protocol per TTI:
      1) ``queues.begin_tti`` (arrivals, expirations);
      2) external scheduler/PHY specifies TBs and which queued bytes they
         attempt to transmit (``planned_flow_bytes``);
      3) ``execute_tti`` samples each TB ONCE; on a failure none of that UE's
         planned flow bytes are delivered; on success queues apply service.

    Planned per-flow payload can be below TB capacity (remaining TB capacity
    is padding/unfilled resources). The engine does not choose QoS priority.

    Repeated/invalid calls are checked before RNG consumption. The actual
    flow queue is touched only once, after all TB decisions are available.
    """

    def __init__(self, *, seed: int = 0) -> None:
        _nonnegative_integer(seed, "seed")
        self._rng = random.Random(seed)
        self._seed = seed

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            _nonnegative_integer(seed, "seed")
            self._seed = seed
        self._rng.seed(self._seed)

    def execute_tti(
        self,
        *,
        queues: ServiceAwareTrafficManager,
        transport_blocks: Sequence[DownlinkTB],
        planned_flow_bytes: Mapping[FlowKey, int],
    ) -> TransmissionTTIResult:
        """Sample decode results and apply only successful per-flow bytes.

        ``planned_flow_bytes`` are bytes placed in scheduled TBs, NOT bytes
        guaranteed to arrive at the receiver. An undelivered TB leaves all
        its queued bytes untouched (without HARQ state in this task).
        """
        if not isinstance(queues, ServiceAwareTrafficManager):
            raise TypeError("queues must be ServiceAwareTrafficManager")
        if not queues.has_active_tti:
            raise RuntimeError("Call queues.begin_tti() before execute_tti()")
        if not isinstance(transport_blocks, (list, tuple)):
            raise TypeError("transport_blocks must be a list or tuple")
        if not isinstance(planned_flow_bytes, Mapping):
            raise TypeError("planned_flow_bytes must be a mapping")

        # Validate the entire plan BEFORE drawing randomness or mutating queues.
        tb_by_ue: dict[int, DownlinkTB] = {}
        for tb in transport_blocks:
            if not isinstance(tb, DownlinkTB):
                raise TypeError("transport_blocks must contain DownlinkTB")
            if tb.ue_index >= queues.num_ues:
                raise ValueError("TB references a UE outside the local queue")
            if tb.ue_index in tb_by_ue:
                raise ValueError("At most one TB per UE per TTI is supported")
            tb_by_ue[tb.ue_index] = tb

        planned_per_ue = [0] * queues.num_ues
        for key, amount in planned_flow_bytes.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("Flow key must be (local_ue_index, flow_id)")
            _nonnegative_integer(amount, "planned_flow_bytes")
            try:
                snap = queues.flow_snapshot(*key)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Unknown local QoS flow {key!r}") from exc
            ue = snap.spec.ue_index
            if amount > snap.backlog_bytes:
                raise ValueError("Planned flow payload exceeds current queued bytes")
            planned_per_ue[ue] += amount

        for ue, amount in enumerate(planned_per_ue):
            cap = tb_by_ue[ue].payload_capacity_bytes if ue in tb_by_ue else 0
            if amount > cap:
                raise ValueError("Planned flow bytes exceed TB capacity (or no TB exists)")

        successful_by_ue = [0] * queues.num_ues
        outcomes: list[TBDecodeOutcome] = []
        scheduled = decoded = 0
        for tb in transport_blocks:
            # One draw per scheduled TB, even at BLER endpoints.
            success = self._rng.random() >= tb.tbler
            cap = tb.payload_capacity_bytes
            scheduled += cap
            if success:
                decoded += cap
                successful_by_ue[tb.ue_index] = cap
            outcomes.append(
                TBDecodeOutcome(tb.ue_index, cap, tb.tbler, success, cap if success else 0)
            )

        # No PHY/queue recomputation. Existing Task 1 logic performs FIFO
        # partial service, packet completion, and exact backlog accounting.
        successful_flow_plan = {
            key: amount
            for key, amount in planned_flow_bytes.items()
            if amount and successful_by_ue[key[0]] > 0
        }
        service = queues.apply_service(
            service_bytes_by_flow=successful_flow_plan,
            ue_success_capacity_bytes=successful_by_ue,
        )
        decoded_planned = sum(successful_flow_plan.values())
        if service.delivered_bytes != decoded_planned:
            # This should be impossible after the prevalidated backlog checks.
            raise AssertionError("Queue service changed from the validated TB plan")
        return TransmissionTTIResult(
            tti_index=service.tti_index,
            tb_outcomes=tuple(outcomes),
            scheduled_capacity_bytes=scheduled,
            decoded_capacity_bytes=decoded,
            failed_capacity_bytes=scheduled - decoded,
            planned_flow_payload_bytes=sum(planned_per_ue),
            decoded_flow_payload_bytes=decoded_planned,
            traffic_service=service,
            unused_decoded_capacity_bytes=decoded - service.delivered_bytes,
        )


def single_rbg_transport_blocks_from_evaluation(
    *,
    evaluation: CellAllocationEvaluationData,
    queue_ue_by_candidate: Sequence[int] | "torch.Tensor",
    slot_duration_s: float,
) -> tuple[DownlinkTB, ...]:
    """Reuse already-computed PHY rate/TBLER for the *one-RBG-per-UE* case.

    The old PHY outputs ``candidate_nominal_rate_bps`` and
    ``candidate_expected_goodput_bps`` with shape [candidate, RBG]. With one
    assigned RBG, ``1 - expected/nominal`` equals its existing TBLER estimate.
    This function does NOT invoke Sionna or run channel evaluation again.

    It deliberately REFUSES candidates with >1 assigned RBG: their existing
    per-RBG TBLER values are NOT estimates of the entire NR TB. Task 3 must
    supply correct whole-grant TB size and BLER for general allocations.

    ``queue_ue_by_candidate`` maps the PHY's candidate *positions* to local
    indexes in the queue manager, not necessarily global network UE IDs.
    Use -1 only for unused/padded candidates.
    """
    import torch

    if not isinstance(slot_duration_s, (int, float)) or isinstance(slot_duration_s, bool) or not math.isfinite(slot_duration_s) or slot_duration_s <= 0:
        raise ValueError("slot_duration_s must be positive and finite")
    nominal = evaluation.candidate_nominal_rate_bps
    goodput = evaluation.candidate_expected_goodput_bps
    if not isinstance(nominal, torch.Tensor) or not isinstance(goodput, torch.Tensor):
        raise TypeError("Evaluation rates must be torch tensors")
    if nominal.ndim != 2 or nominal.shape != goodput.shape:
        raise ValueError("Expected [candidate, RBG] rates with matching shapes")
    if isinstance(queue_ue_by_candidate, torch.Tensor):
        mapping = queue_ue_by_candidate
        if mapping.ndim != 1 or mapping.dtype == torch.bool or torch.is_floating_point(mapping):
            raise ValueError("queue_ue_by_candidate must be a one-dimensional integer tensor")
        # One bulk copy, never individual CUDA scalar reads.
        ue_indices = mapping.detach().to(device="cpu", dtype=torch.long).tolist()
    else:
        ue_indices = queue_ue_by_candidate
    if len(ue_indices) != nominal.shape[0]:
        raise ValueError("queue_ue_by_candidate must match candidate count")

    # Transfer both small candidate x RBG arrays together. No repeated CUDA
    # scalar predicates / .item() calls which would serialize the GPU.
    stacked = torch.stack((nominal, goodput)).detach().to("cpu").tolist()
    nominal_rows, goodput_rows = stacked
    for nom_row, gp_row in zip(nominal_rows, goodput_rows):
        for rate, expected in zip(nom_row, gp_row):
            if not math.isfinite(rate) or not math.isfinite(expected):
                raise ValueError("PHY rates must be finite")
            if rate < 0.0 or expected < 0.0:
                raise ValueError("PHY rates must be nonnegative")
            if expected > rate + max(rate * 1e-5, 1e-5):
                raise ValueError("Expected goodput exceeds nominal rate")

    blocks: list[DownlinkTB] = []
    seen_ues: set[int] = set()
    for candidate, (nom_row, gp_row) in enumerate(zip(nominal_rows, goodput_rows)):
        active = [i for i, rate in enumerate(nom_row) if rate > 0.0]
        if not active:
            continue
        if len(active) != 1:
            raise ValueError(
                "Multi-RBG UE allocation has no valid whole-TB BLER in the "
                "existing evaluator; supply an explicit DownlinkTB instead"
            )
        ue = ue_indices[candidate]
        if isinstance(ue, bool) or not isinstance(ue, int) or ue < 0:
            raise ValueError("Scheduled candidate must map to a local UE")
        if ue in seen_ues:
            raise ValueError("Multiple candidates map to the same queue UE")
        seen_ues.add(ue)
        rbg = active[0]
        rate = nom_row[rbg]
        p = 1.0 - gp_row[rbg] / rate
        # Float32 rate rounding may put p infinitesimally outside [0,1].
        if not -1e-5 <= p <= 1.0 + 1e-5:
            raise ValueError("Inconsistent rate/goodput combination")
        # Existing rate = TB_size_bits/slot_duration. TB_size_bits are
        # byte-aligned; round once to recover nominal bytes without truncating
        # a floating-point value such as 99.9999999999 to 99.
        size = round(rate * slot_duration_s / 8.0)
        if size < 1:
            raise ValueError("PHY TB payload rounded to zero bytes")
        blocks.append(DownlinkTB(ue, size, min(1.0, max(0.0, p))))
    return tuple(blocks)
