"""Opt-in causal CSI aging for Task-5 service-aware downlink scheduling.

Only the scheduling policy receives a dated CSI report. The actual PHY link
provider receives the *current* temporally evolving channel tensor, plus the
reported CSI (for CSI-based precoding/combining). This module never computes
Sionna channel samples, channel matrices, or new packet queues.

The source is one precomputed short Sionna temporal window or a callable that
returns ONE current [B, UE, RX, BS, TX, 1, SC] tensor per TTI. The latter can
reuse an existing memory-safe slice provider rather than store a large window.
There is NO implied temporal continuity across independent channel windows.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

import torch

from oran_scheduler.simulator.service_aware_downlink import (
    DownlinkSchedulingDecision,
)
from oran_scheduler.simulator.service_aware_nr_grants import WholeGrantLink
from oran_scheduler.simulator.service_aware_observation import QoSCellObservation


class TemporalWindowLike(Protocol):
    @property
    def num_ttis(self) -> int: ...
    def tti_slice(self, tti_offset: int) -> torch.Tensor: ...


class TemporalWindowFrameSource:
    """No-copy, small-window bridge for existing TemporalChannelWindow.

    Does not regenerate a channel or move it to CPU. Limit window duration and
    dimensions to the GPU memory budget; use an on-demand source for large runs.
    """

    def __init__(self, window: TemporalWindowLike) -> None:
        if type(window.num_ttis) is not int or window.num_ttis < 1:
            raise ValueError("Temporal window must have at least one TTI")
        self.window = window

    def __call__(self, tti_index: int) -> torch.Tensor:
        if type(tti_index) is not int or not 0 <= tti_index < self.window.num_ttis:
            raise ValueError("TTI outside the available temporal window")
        return self.window.tti_slice(tti_index)


@dataclass(frozen=True, slots=True)
class CSIReport:
    """GPU-resident channel report. A report payload must not be mutated.

    `membership` encodes global UE identities in the scheduler's local candidate
    ordering. Strictly reject identity changes instead of silently reusing CSI
    reports/combiner matrices for a different user. Rebuild/reset at handover.
    """
    tti_index: int
    membership: tuple[int, ...]
    payload: object


@dataclass(frozen=True, slots=True)
class PublishedCSI:
    """Only the report available to a scheduler at a particular TTI.

    No actual H_t or channel source is stored here; the policy gets no future
    or current physical channel except the report legitimately available.
    """
    decision_tti: int
    report_tti: int
    age_ttis: int
    configured_delay_ttis: int
    report: CSIReport
    bootstrap: bool


ReportBuilder = Callable[[torch.Tensor, int], CSIReport]
PolicyWithCSI = Callable[[QoSCellObservation, PublishedCSI], DownlinkSchedulingDecision]
CurrentPhysicalLink = Callable[
    [QoSCellObservation, DownlinkSchedulingDecision, torch.Tensor, PublishedCSI],
    Mapping[int, WholeGrantLink],
]


class ServiceAwareTemporalCSIAdapter:
    """Exactly one physical frame per TTI; one published CSI snapshot per policy.

    Uses the existing Task-5 runner without changing it:

        adapter.policy -> runner.policy
        adapter.link_provider -> runner.link_provider

    With feedback delay d, TTI t uses most recent periodic report sampled at
    s <= t-d, while link_provider always receives actual H_t. During initial
    TTI(s) with no report old enough, report H_0 is used as an explicitly
    identified **bootstrap**. This startup assumption is causal but not a
    faithful CSI-feedback protocol. Drop startup slots from steady-state QoS
    statistics if exact fixed-age CSI is required.

    CSI calculation runs once only on report TTIs (periodic rate configurable),
    without autograd. Stores at most O(ceil(delay/period)+2) report references,
    not large channel windows. Does not retain actual frames between TTIs.
    """

    def __init__(
        self,
        *,
        frame_source: Callable[[int], torch.Tensor],
        report_builder: ReportBuilder,
        policy: PolicyWithCSI,
        physical_link_provider: CurrentPhysicalLink,
        delay_ttis: int = 0,
        report_period_ttis: int = 1,
    ) -> None:
        if type(delay_ttis) is not int or delay_ttis < 0:
            raise ValueError("delay_ttis must be a non-negative integer")
        if type(report_period_ttis) is not int or report_period_ttis < 1:
            raise ValueError("report_period_ttis must be a positive integer")
        if any(not callable(f) for f in (frame_source, report_builder, policy, physical_link_provider)):
            raise TypeError("All provider/policy arguments must be callable")
        self.frame_source = frame_source
        self.report_builder = report_builder
        self._select = policy
        self._link = physical_link_provider
        self.delay_ttis = delay_ttis
        self.report_period_ttis = report_period_ttis
        self._last_tti: int | None = None
        self._history: deque[CSIReport] = deque()
        self._membership: tuple[int, ...] | None = None
        self._physical_frame: torch.Tensor | None = None
        self._layout: tuple[tuple[int, ...], torch.device, torch.dtype] | None = None
        self._published: PublishedCSI | None = None
        self._decision: DownlinkSchedulingDecision | None = None
        self._awaiting_link = False
        self._report_count = 0
        self._frame_count = 0

    @property
    def report_build_count(self) -> int:
        return self._report_count

    @property
    def frame_fetch_count(self) -> int:
        return self._frame_count

    @property
    def retained_reports(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        """Call after runner.reset(). No channel or PHY state is changed."""
        self._last_tti = None
        self._history.clear()
        self._membership = None
        self._physical_frame = None
        self._layout = None
        self._published = None
        self._decision = None
        self._awaiting_link = False
        self._report_count = 0
        self._frame_count = 0

    def policy(self, observed: QoSCellObservation) -> DownlinkSchedulingDecision:
        if self._awaiting_link:
            raise RuntimeError("Previous CSI policy decision has no matching PHY link")
        t = observed.tti_index
        expected = 0 if self._last_tti is None else self._last_tti + 1
        if type(t) is not int or t != expected:
            raise ValueError(f"Temporal CSI requires contiguous TTI {expected}, got {t}")

        with torch.no_grad():
            physical = self.frame_source(t)
            if not isinstance(physical, torch.Tensor) or physical.ndim != 7:
                raise ValueError("Current frequency channel must be a 7D torch tensor")
            if not torch.is_complex(physical) or physical.shape[-2] != 1:
                raise ValueError("Current channel must be complex with exactly one temporal sample")
            if any(x <= 0 for x in physical.shape):
                raise ValueError("Current channel has an empty physical axis")
            physical = physical.detach()  # no graph may be retained across TTIs
            layout = (tuple(physical.shape), physical.device, physical.dtype)
            if self._layout is not None and layout != self._layout:
                raise RuntimeError("Physical channel dimensions/device/dtype changed during CSI episode")
            self._layout = layout
            self._frame_count += 1
            if t % self.report_period_ttis == 0:
                report = self.report_builder(physical, t)
                if not isinstance(report, CSIReport) or report.tti_index != t:
                    raise ValueError("CSI builder must return a report for this exact TTI")
                if (not report.membership or any(type(u) is not int or u < 0 for u in report.membership)
                        or len(set(report.membership)) != len(report.membership)):
                    raise ValueError("CSI report must have unique non-negative UE membership IDs")
                if self._membership is None:
                    self._membership = report.membership
                if report.membership != self._membership:
                    raise RuntimeError("Candidate membership changed; reset/rebuild CSI adapter at handover")
                self._history.append(report)
                self._report_count += 1
            if not self._history:
                raise AssertionError("First CSI report was not created at TTI 0")
            target_tti = t - self.delay_ttis
            chosen = next((r for r in reversed(self._history) if r.tti_index <= target_tti), None)
            eligible = chosen is not None
            if chosen is None:
                chosen = self._history[0]
            if chosen.tti_index > t:
                raise AssertionError("Future CSI leaked into scheduler")
            published = PublishedCSI(
                decision_tti=t, report_tti=chosen.tti_index,
                age_ttis=t - chosen.tti_index,
                configured_delay_ttis=self.delay_ttis,
                report=chosen, bootstrap=not eligible,
            )
            # After the decision, older CSI history cannot affect a future
            # scheduling choice, even if report periods are >1.
            while len(self._history) > 1 and self._history[1].tti_index <= chosen.tti_index:
                self._history.popleft()
            decision = self._select(observed, published)
            if not isinstance(decision, DownlinkSchedulingDecision):
                raise TypeError("CSI scheduling policy must return DownlinkSchedulingDecision")
        self._physical_frame = physical
        self._published = published
        self._decision = decision
        self._awaiting_link = True
        self._last_tti = t
        return decision

    def link_provider(
        self, observed: QoSCellObservation, decision: DownlinkSchedulingDecision,
    ) -> Mapping[int, WholeGrantLink]:
        if not self._awaiting_link or self._physical_frame is None or self._published is None:
            raise RuntimeError("Link callback called without preceding CSI policy")
        if observed.tti_index != self._published.decision_tti or decision is not self._decision:
            raise RuntimeError("Stale or mismatched schedule-to-actual-PHY binding")
        try:
            with torch.no_grad():
                links = self._link(observed, decision, self._physical_frame, self._published)
            if not isinstance(links, Mapping):
                raise TypeError("Physical link provider must return a mapping")
            return links
        finally:
            # No cached physical H_t survives into a later TTI. Report payloads
            # remain only for as long as needed by configured CSI delay.
            self._physical_frame = None
            self._published = None
            self._decision = None
            self._awaiting_link = False
