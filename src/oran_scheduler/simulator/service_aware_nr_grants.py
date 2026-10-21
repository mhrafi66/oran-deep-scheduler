"""Task 3: opt-in NR downlink grant geometry and whole-TB payload sizing.

This module does NOT modify legacy Nokia per-RBG rates or the PHY evaluator.
It converts a *final* cell allocation to complete per-UE transport blocks,
using a previously determined WHOLE-GRANT MCS/TBLER and Sionna's NR TB-size
functions. It never combines per-RBG TB sizes/BLERs or reevaluates channels.

One MAC TB per UE per slot; rank 1 or 2. It requires the actual simulated
channel bandwidth to match the chosen BWP rather than silently reinterpreting
a reduced 18-PRB channel as 273 PRBs.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
import math
from typing import Callable, Mapping, Sequence

from oran_scheduler.simulator.service_aware_transmission import DownlinkTB


def _int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _p(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite probability in [0,1]")
    return float(value)


@dataclass(frozen=True)
class NRBandwidthPart:
    """PRB geometry of an active contiguous BWP, RBG Type-0-like alignment.

    RBG boundaries align to multiples of rbg_size_prbs in carrier PRB space;
    first/last RBG may be partial. The RBG-size parameter is explicit, not
    inferred from a possibly mismatched simulation bandwidth. This is geometry,
    not a full 3GPP RBG-size configuration selector.
    """

    num_prbs: int
    rbg_size_prbs: int
    bwp_start_prb: int = 0

    def __post_init__(self) -> None:
        _int(self.num_prbs, "num_prbs", minimum=1)
        _int(self.rbg_size_prbs, "rbg_size_prbs", minimum=1)
        _int(self.bwp_start_prb, "bwp_start_prb")

    @cached_property
    def rbg_prb_spans(self) -> tuple[tuple[int, int], ...]:
        """(start, stop) PRB intervals, zero-based relative to the BWP."""
        start = self.bwp_start_prb
        stop = start + self.num_prbs
        group_size = self.rbg_size_prbs
        return tuple(
            (
                max(start, group * group_size) - start,
                min(stop, (group + 1) * group_size) - start,
            )
            for group in range(start // group_size, (stop - 1) // group_size + 1)
        )

    @cached_property
    def prbs_per_rbg(self) -> tuple[int, ...]:
        return tuple(end - start for start, end in self.rbg_prb_spans)

    @property
    def num_rbgs(self) -> int:
        return len(self.prbs_per_rbg)


@dataclass(frozen=True)
class PDSCHDataRegion:
    """Explicit slot data-RE budget for NR TBS sizing (38.214, 5.1.3.2).

    n_re_per_prb = min(156, 12*allocated_symbols - dmrs_re - x_overhead).
    dmrs_re_per_prb already includes all DMRS symbol/port use; DO NOT subtract
    DMRS or control overhead again downstream. This is an explicit grant-side
    budget, not a complete NR control-channel/signaling model.
    """

    num_pdsch_symbols: int = 14
    dmrs_re_per_prb: int = 12
    x_overhead_re_per_prb: int = 0
    higher_layer_header_bytes: int = 0
    mcs_table_index: int = 2
    max_layers_per_rbg: int = 16

    def __post_init__(self) -> None:
        if not 1 <= _int(self.num_pdsch_symbols, "num_pdsch_symbols") <= 14:
            raise ValueError("num_pdsch_symbols must be 1..14")
        _int(self.dmrs_re_per_prb, "dmrs_re_per_prb")
        if self.x_overhead_re_per_prb not in (0, 6, 12, 18) or type(self.x_overhead_re_per_prb) is not int:
            raise ValueError("x_overhead_re_per_prb must be one of 0,6,12,18")
        _int(self.higher_layer_header_bytes, "higher_layer_header_bytes")
        if self.mcs_table_index != 2 or type(self.mcs_table_index) is not int:
            raise ValueError("Only downlink NR MCS table 2 is supported here")
        _int(self.max_layers_per_rbg, "max_layers_per_rbg", minimum=1)
        if self.data_re_per_prb <= 0:
            raise ValueError("Nonpositive data RE/PRB after DMRS and overhead")

    @property
    def data_re_per_prb(self) -> int:
        return min(156, 12 * self.num_pdsch_symbols - self.dmrs_re_per_prb - self.x_overhead_re_per_prb)


@dataclass(frozen=True)
class WholeGrantLink:
    """One TB-wide link result for an EXACT scheduled candidate/RBG set.

    TBLER and MCS refer to all assigned RBGs jointly, NOT a per-RBG average
    or product. This must come from a whole-grant PHY abstraction/measurement.
    Task 3 does not synthesize such a link estimate from partial RBG rates.
    """

    candidate_index: int
    assigned_rbgs: tuple[int, ...]
    mcs_index: int
    tbler: float
    num_layers: int = 1

    def __post_init__(self) -> None:
        _int(self.candidate_index, "candidate_index")
        if not isinstance(self.assigned_rbgs, tuple) or not self.assigned_rbgs:
            raise ValueError("assigned_rbgs must be a nonempty tuple")
        if any(type(idx) is not int or idx < 0 for idx in self.assigned_rbgs):
            raise ValueError("RBG indices must be nonnegative integers")
        if tuple(sorted(set(self.assigned_rbgs))) != self.assigned_rbgs:
            raise ValueError("RBG indices must be unique and sorted")
        if not 0 <= _int(self.mcs_index, "mcs_index") <= 27:
            raise ValueError("Only nonreserved PDSCH MCS Table-2 indexes 0..27 are allowed")
        if _int(self.num_layers, "num_layers", minimum=1) not in (1, 2):
            raise ValueError("Service-aware model supports rank 1 or 2 per UE")
        _p(self.tbler, "tbler")


@dataclass(frozen=True)
class PlannedNRGrant:
    candidate_index: int
    ue_index: int
    assigned_rbgs: tuple[int, ...]
    num_prbs: int
    data_re: int
    num_layers: int
    mcs_index: int
    tbler: float


@dataclass(frozen=True)
class SizedNRGrant:
    plan: PlannedNRGrant
    tbs_bits: int
    payload_capacity_bytes: int

    def to_downlink_tb(self) -> DownlinkTB:
        return DownlinkTB(self.plan.ue_index, self.payload_capacity_bytes, self.plan.tbler)


# Batched callback receives (mcs_index, data_re, layers) tuples and returns
# Sionna-sized TBS (integer bits) once per distinct uncached shape.
TBSBackend = Callable[[Sequence[tuple[int, int, int]]], Sequence[int]]


def sionna_nr_tb_size_bits(specs: Sequence[tuple[int, int, int]]) -> list[int]:
    """Reuse Sionna NR MCS decoder and NR TB-size calculator in ONE CPU batch.

    Small final-grant batch is intentionally CPU-side. No channel/SINR tensors
    or CUDA per-element synchronization are involved. Invoked only for unseen
    (MCS, RE, layers) combinations by NRGrantAssembler's bounded cache.
    """
    if not specs:
        return []
    import torch
    from sionna.phy.nr.utils import calculate_tb_size, decode_mcs_index

    mcs = torch.tensor([s[0] for s in specs], dtype=torch.int32)
    re = torch.tensor([s[1] for s in specs], dtype=torch.int32)
    layers = torch.tensor([s[2] for s in specs], dtype=torch.int32)
    qm, rate = decode_mcs_index(
        mcs_index=mcs, table_index=2, is_pusch=False, device="cpu"
    )
    tbs_bits, _, _, _, _ = calculate_tb_size(
        modulation_order=qm,
        target_coderate=rate,
        num_coded_bits=qm * re * layers,
        return_cw_length=False,
        device="cpu",
    )
    return [int(v) for v in tbs_bits.detach().cpu().tolist()]


class NRGrantAssembler:
    """Construct TBs using actual allocation and a whole-grant link estimate.

    Reusable for many TTIs. Caches PRB geometry and bounded TBS results.
    It does NOT maintain UE queues, choose flow priorities or calculate SINR.
    """

    def __init__(
        self,
        *,
        bandwidth: NRBandwidthPart,
        region: PDSCHDataRegion,
        channel_num_prbs: int,
        tbs_backend: TBSBackend | None = None,
        max_cache_entries: int = 2048,
    ) -> None:
        if not isinstance(bandwidth, NRBandwidthPart) or not isinstance(region, PDSCHDataRegion):
            raise TypeError("bandwidth and region must be valid configuration objects")
        if _int(channel_num_prbs, "channel_num_prbs", minimum=1) != bandwidth.num_prbs:
            raise ValueError("BWP width and actual simulated PHY channel PRBs must match")
        _int(max_cache_entries, "max_cache_entries", minimum=1)
        if tbs_backend is not None and not callable(tbs_backend):
            raise TypeError("tbs_backend must be callable")
        self.bandwidth = bandwidth
        self.region = region
        self._backend = tbs_backend or sionna_nr_tb_size_bits
        self._max_cache = max_cache_entries
        self._cache: OrderedDict[tuple[int, int, int], int] = OrderedDict()
        self._counts = bandwidth.prbs_per_rbg

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    def plan(
        self,
        *,
        allocation: "CellAllocation",
        queue_ue_by_candidate: Sequence[int],
        whole_links: Mapping[int, WholeGrantLink],
    ) -> tuple[PlannedNRGrant, ...]:
        """Read final [user_slot,RBG] actions ONCE, with MU-MIMO sharing allowed.

        Each occupied (candidate,RBG) is unique. A candidate may occupy many
        RBGs and is grouped into ONE TB. Padded -1 mappings cannot be served.
        """
        import torch
        from oran_scheduler.schedulers.allocation import CellAllocation

        if not isinstance(allocation, CellAllocation):
            raise TypeError("allocation must be CellAllocation")
        matrix = allocation.candidate_by_user_slot
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2 or matrix.dtype == torch.bool or matrix.is_floating_point():
            raise ValueError("Allocation must be a two-dimensional integer tensor")
        if matrix.shape[1] != self.bandwidth.num_rbgs:
            raise ValueError("Allocation RBG count does not match BWP RBG geometry")
        if matrix.shape[0] < 1:
            raise ValueError("Allocation must have at least one user slot")
        # This is a *final* small cell allocation, not the full-channel tensor.
        # Transfer from GPU to CPU once, not one .item() for every RBG.
        cells = matrix.detach().to(device="cpu", dtype=torch.long).tolist()
        if isinstance(queue_ue_by_candidate, torch.Tensor):
            if (queue_ue_by_candidate.ndim != 1 or queue_ue_by_candidate.dtype == torch.bool
                    or queue_ue_by_candidate.is_floating_point()):
                raise ValueError("Queue UE mapping must be a one-dimensional integer tensor")
            # One bulk GPU->CPU transfer, never repeated CUDA scalar reads.
            mapping = tuple(queue_ue_by_candidate.detach().to(device="cpu", dtype=torch.long).tolist())
        else:
            mapping = tuple(queue_ue_by_candidate)
        candidate_count = len(mapping)
        if any(type(x) is not int or x < -1 for x in mapping):
            raise ValueError("Queue UE mappings must be integers >= -1")
        if not isinstance(whole_links, Mapping):
            raise TypeError("whole_links must be a mapping keyed by candidate index")
        assignments: dict[int, set[int]] = {}
        for rbg in range(self.bandwidth.num_rbgs):
            seen_candidates = set()
            for user_slot in range(len(cells)):
                candidate = cells[user_slot][rbg]
                if candidate == -1:
                    continue
                if candidate < 0 or candidate >= candidate_count:
                    raise ValueError("Allocation references invalid candidate")
                if candidate in seen_candidates:
                    raise ValueError("UE scheduled twice on same RBG")
                seen_candidates.add(candidate)
                assignments.setdefault(candidate, set()).add(rbg)
        used_ues: set[int] = set()
        layer_count_by_rbg = [0] * self.bandwidth.num_rbgs
        plans: list[PlannedNRGrant] = []
        if set(whole_links) != set(assignments):
            raise ValueError("Whole-grant link entries must match exactly the scheduled candidates")
        for candidate in sorted(assignments):
            ue = mapping[candidate]
            if ue < 0 or ue in used_ues:
                raise ValueError("Each scheduled candidate must map to a distinct real queue UE")
            used_ues.add(ue)
            link = whole_links[candidate]
            if not isinstance(link, WholeGrantLink) or link.candidate_index != candidate:
                raise ValueError("WholeGrantLink identity mismatch")
            rbgs = tuple(sorted(assignments[candidate]))
            if link.assigned_rbgs != rbgs:
                raise ValueError("Whole-grant link result is stale or from a different RBG allocation")
            for rbg in rbgs:
                layer_count_by_rbg[rbg] += link.num_layers
                if layer_count_by_rbg[rbg] > self.region.max_layers_per_rbg:
                    raise ValueError("Exceeds maximum physical layers on an RBG")
            prbs = sum(self._counts[rbg] for rbg in rbgs)
            plans.append(PlannedNRGrant(
                candidate_index=candidate,
                ue_index=ue,
                assigned_rbgs=rbgs,
                num_prbs=prbs,
                data_re=prbs * self.region.data_re_per_prb,
                num_layers=link.num_layers,
                mcs_index=link.mcs_index,
                tbler=link.tbler,
            ))
        return tuple(plans)

    def size(self, plans: Sequence[PlannedNRGrant]) -> tuple[SizedNRGrant, ...]:
        """Batch-calculate only cache misses; never sum per-RBG TB sizes."""
        if not isinstance(plans, (tuple, list)):
            raise TypeError("plans must be a list or tuple")
        unique: dict[tuple[int, int, int], None] = {}
        used_ue: set[int] = set()
        for plan in plans:
            if not isinstance(plan, PlannedNRGrant):
                raise TypeError("Expected PlannedNRGrant")
            if plan.ue_index in used_ue:
                raise ValueError("One TB per UE per slot")
            used_ue.add(plan.ue_index)
            if (not plan.assigned_rbgs
                    or any(type(r) is not int or r < 0 or r >= self.bandwidth.num_rbgs for r in plan.assigned_rbgs)
                    or tuple(sorted(set(plan.assigned_rbgs))) != plan.assigned_rbgs):
                raise ValueError("Grant plan has invalid RBG assignment")
            if (plan.num_prbs != sum(self._counts[r] for r in plan.assigned_rbgs)
                    or plan.data_re != plan.num_prbs * self.region.data_re_per_prb):
                raise ValueError("Grant plan RE/PRB geometry mismatch")
            key = (plan.mcs_index, plan.data_re, plan.num_layers)
            if key not in self._cache:
                unique[key] = None
        calculated: dict[tuple[int, int, int], int] = {}
        if unique:
            uncached = tuple(unique)
            sizes = self._backend(uncached)
            if len(sizes) != len(uncached):
                raise ValueError("TBS backend returned wrong number of sizes")
            # Transactional: validate all outputs before changing the cache.
            verified = {}
            for key, bits in zip(uncached, sizes):
                if type(bits) is not int or bits <= 0 or bits % 8 != 0:
                    raise ValueError("NR TBS backend must return positive byte-aligned bit counts")
                if bits // 8 <= self.region.higher_layer_header_bytes:
                    raise ValueError("Configured headers consume the complete transport block")
                verified[key] = bits
            calculated = verified
        output = []
        for plan in plans:
            key = (plan.mcs_index, plan.data_re, plan.num_layers)
            bits = self._cache[key] if key in self._cache else calculated[key]
            payload = bits // 8 - self.region.higher_layer_header_bytes
            if payload <= 0:
                raise ValueError("Configured headers consume the complete transport block")
            output.append(SizedNRGrant(plan=plan, tbs_bits=bits, payload_capacity_bytes=payload))
        # Cache remains bounded even when one slot has more unique grants than
        # max_cache_entries; those grants still use this slot's local results.
        for key, bits in calculated.items():
            self._cache[key] = bits
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_cache:
                self._cache.popitem(last=False)
        for plan in plans:
            key = (plan.mcs_index, plan.data_re, plan.num_layers)
            if key in self._cache:
                self._cache.move_to_end(key)
        return tuple(output)

    def build(
        self,
        *,
        allocation: "CellAllocation",
        queue_ue_by_candidate: Sequence[int],
        whole_links: Mapping[int, WholeGrantLink],
    ) -> tuple[DownlinkTB, ...]:
        """Final Task-2-ready transport blocks, using one TB per selected UE."""
        return tuple(
            sized.to_downlink_tb() for sized in self.size(self.plan(
                allocation=allocation,
                queue_ue_by_candidate=queue_ue_by_candidate,
                whole_links=whole_links,
            ))
        )
