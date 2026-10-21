"""Opt-in mobility/handover bridge for service-aware packet scheduling.

The authoritative packet queues remain indexed by *global UE ID*, not by
serving-cell-local slot. A successful handover changes scheduling eligibility,
never ownership of packet objects; no queue copying or parallel packet ledger.

Uses existing MobilityEngine (x=x0+v*t), HandoverController (hysteresis/TTT),
and accepts an existing received-power provider. Does not construct or claim a
spatially consistent Sionna channel, nor simulate full NR RRC/HARQ handover.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch

from oran_scheduler.simulator.mobility import MobilityEngine, MobilitySnapshot
from oran_scheduler.simulator.handover import HandoverController, HandoverStepResult
from oran_scheduler.simulator.service_aware_traffic import ServiceAwareTrafficManager
from oran_scheduler.simulator.service_aware_observation import QoSCellObservation


# Caller can wrap an existing Sionna measurement provider with:
#     lambda snap: sionna_measurements(snap.tti_index)
# or reuse powers already computed for the actual channel.
MeasurementProvider = Callable[[MobilitySnapshot], torch.Tensor]


@dataclass(frozen=True, slots=True)
class UEHandoverEvent:
    tti_index: int
    global_ue_id: int
    old_cell: int
    new_cell: int
    preserved_queue_bytes: int


@dataclass(frozen=True, slots=True)
class MobilityRoutingSnapshot:
    tti_index: int
    geometry: MobilitySnapshot
    # Global persistent IDs for each serving cell, NOT QoS/TDS-selected subset.
    ue_ids_by_cell: tuple[tuple[int, ...], ...]
    # Changed when any serving-cell membership changes; rebuild per-cell CSI
    # reports and candidate mappings, do not relabel stale CSI tensors.
    membership_generation: int
    handovers: tuple[UEHandoverEvent, ...]
    queue_backlog_bytes: int


class ServiceAwareMobilityHandoverBridge:
    """Global UE routing, existing mobility, and handover in causal TTI order.

    Call `advance(tti)` once at the START of the same slot, after packet
    arrivals are admitted (or before them if no new arrivals). The queue
    manager is never changed by reassociation. A caller may use
    `cell_observation()` to prevent a cell's policy from seeing other cells'
    QoS state, and `eligible_ue_ids()` for its TDS input population.

    Only one *authoritative* ServiceAwareTrafficManager may be attached.
    The bridge deliberately does NOT schedule more than one cell or model PHY;
    the existing runner/PHY must consume the resulting cell membership.
    """

    def __init__(
        self,
        *,
        mobility: MobilityEngine,
        handover: HandoverController,
        queues: ServiceAwareTrafficManager,
        measurements: MeasurementProvider,
    ) -> None:
        if not isinstance(mobility, MobilityEngine):
            raise TypeError("mobility must be MobilityEngine")
        if not isinstance(handover, HandoverController):
            raise TypeError("handover must be HandoverController")
        if not isinstance(queues, ServiceAwareTrafficManager):
            raise TypeError("queues must be ServiceAwareTrafficManager")
        if not callable(measurements):
            raise TypeError("measurements must be callable")
        if queues.num_ues != handover.num_ues:
            raise ValueError("Queue and handover UE populations must agree")
        topology = mobility.current_topology
        if topology.ut_loc.ndim != 3 or tuple(topology.ut_loc.shape[:2]) != (1, queues.num_ues):
            raise ValueError("Mobility topology must have batch 1 and every global UE")
        if topology.ut_loc.device != handover.device:
            raise ValueError("Mobility and handover must use the same torch device")
        if abs(mobility.config.tti_duration_s - queues.tti_duration_s) > 1e-12:
            raise ValueError("Mobility and packet queues must agree on TTI duration")
        if handover._last_tti_index is not None:
            raise ValueError("Use a new handover controller at episode start")
        if mobility.current_tti_index != 0:
            raise ValueError("Use mobility starting at TTI 0")
        if queues.has_active_tti or queues.last_completed_tti_index is not None:
            raise ValueError("Construct before queue episode begins")

        self.mobility = mobility
        self.handover = handover
        self.queues = queues
        self.measurements = measurements
        self._last_tti: int | None = None
        self._generation = 0
        self._membership = self._build_membership()
        self._snapshot: MobilityRoutingSnapshot | None = None
        self._measurement_calls = 0

    def _build_membership(self) -> tuple[tuple[int, ...], ...]:
        # Host transfer only at start and on an actual handover, never once per
        # UE or RBG; no transfer of any PHY/CSI/channel tensor.
        serving = self.handover.serving_bs.detach().cpu().tolist()
        groups: list[list[int]] = [[] for _ in range(self.handover.num_bs)]
        for ue, cell in enumerate(serving):
            groups[cell].append(ue)
        return tuple(tuple(group) for group in groups)

    @property
    def membership_generation(self) -> int:
        return self._generation

    @property
    def measurement_calls(self) -> int:
        return self._measurement_calls

    @property
    def current(self) -> MobilityRoutingSnapshot | None:
        return self._snapshot

    def advance(self, tti_index: int) -> MobilityRoutingSnapshot:
        expected = 0 if self._last_tti is None else self._last_tti + 1
        if type(tti_index) is not int or tti_index != expected:
            raise ValueError(f"Expected contiguous mobility TTI {expected}")
        if self.queues.has_active_tti:
            if self.queues.active_tti_index != tti_index:
                raise RuntimeError("Active packet-queue TTI and mobility TTI differ")
        elif self.queues.last_completed_tti_index != (None if tti_index == 0 else tti_index - 1):
            raise RuntimeError("Queue timeline must be aligned with mobility")

        # All state changes after an invalid measurement are avoided by
        # validating the provider's tensor before stepping the handover FSM.
        geometry = self.mobility.advance_to(tti_index)
        with torch.no_grad():
            powers = self.measurements(geometry)
            if not isinstance(powers, torch.Tensor):
                raise TypeError("Measurement callback must return a torch.Tensor")
            if tuple(powers.shape) != (self.handover.num_ues, self.handover.num_bs):
                raise ValueError("Measurement shape must be [global_UE, BS]")
            if powers.device != self.handover.device or not torch.is_floating_point(powers):
                raise ValueError("Measurement must be a floating tensor on the mobility device")
            # HandoverController.step validates finite, nonnegative powers;
            # do not introduce an extra GPU-to-CPU synchronization here.
            self._measurement_calls += 1
            before_bytes = self.queues.ue_backlog_bytes()  # cached; no packet traversal
            before_total = self.queues.backlog_bytes
            handover_result: HandoverStepResult = self.handover.step(
                tti_index=tti_index, link_power=powers,
            )
            moved_ids = torch.nonzero(handover_result.handover_mask, as_tuple=False).flatten()
            moved = tuple(int(ue) for ue in moved_ids.detach().cpu().tolist())
            events: tuple[UEHandoverEvent, ...] = ()
            if moved:
                previous = handover_result.previous_serving_bs.detach().cpu().tolist()
                current = handover_result.serving_bs.detach().cpu().tolist()
                events = tuple(UEHandoverEvent(
                    tti_index=tti_index, global_ue_id=ue,
                    old_cell=int(previous[ue]), new_cell=int(current[ue]),
                    preserved_queue_bytes=before_bytes[ue],
                ) for ue in moved)
                self._membership = self._build_membership()
                self._generation += 1
            if self.queues.backlog_bytes != before_total or self.queues.ue_backlog_bytes() != before_bytes:
                raise AssertionError("Handover unexpectedly modified packet queues")
            if sum(map(len, self._membership)) != self.queues.num_ues:
                raise AssertionError("UE lost or duplicated during reassociation")
            result = MobilityRoutingSnapshot(
                tti_index=tti_index, geometry=geometry,
                ue_ids_by_cell=self._membership,
                membership_generation=self._generation, handovers=events,
                queue_backlog_bytes=before_total,
            )
            self._snapshot = result
            self._last_tti = tti_index
            return result

    def eligible_ue_ids(self, cell_index: int, *, require_backlog: bool = True) -> tuple[int, ...]:
        if self._snapshot is None:
            raise RuntimeError("Advance mobility before reading a cell population")
        if type(cell_index) is not int or not 0 <= cell_index < self.handover.num_bs:
            raise ValueError("Unknown serving cell")
        members = self._membership[cell_index]
        if not require_backlog:
            return members
        bytes_by_ue = self.queues.ue_backlog_bytes()
        return tuple(ue for ue in members if bytes_by_ue[ue] > 0)

    def cell_observation(self, observed: QoSCellObservation, cell_index: int) -> QoSCellObservation:
        """Project global QoS to a cell; global IDs deliberately remain stable."""
        if not isinstance(observed, QoSCellObservation):
            raise TypeError("Expected QoSCellObservation")
        if self._snapshot is None or self._snapshot.tti_index != observed.tti_index:
            raise ValueError("QoS observation and current mobility TTI must agree")
        allowed = set(self.eligible_ue_ids(cell_index, require_backlog=False))
        ues = tuple(u for u in observed.ues if u.ue_index in allowed)
        flows = tuple(f for f in observed.flows if f.ue_index in allowed)
        return QoSCellObservation(
            tti_index=observed.tti_index,
            tti_duration_s=observed.tti_duration_s,
            flows=flows,
            ues=ues,
            total_backlog_bytes=sum(u.backlog_bytes for u in ues),
        )

    def serving_cell(self, global_ue_id: int) -> int:
        if type(global_ue_id) is not int or not 0 <= global_ue_id < self.queues.num_ues:
            raise ValueError("Unknown global UE")
        if self._snapshot is None:
            raise RuntimeError("Advance mobility first")
        for cell, ids in enumerate(self._membership):
            if global_ue_id in ids:
                return cell
        raise AssertionError("Global UE lost from association")
