from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhysicalNetworkPhase:
    """
    Persistent physical network condition beginning
    at start_tti.

    These values match the semantics of the existing
    post-CSI PhysicalExecutionStressConfig.

    They can later drive:

        interference increase,
        serving-link degradation,
        BS outage,
        outage recovery.

    Unlike a single static stress flag, phases allow:

        normal -> failure -> recovery.
    """

    start_tti: int

    name: str

    non_serving_interference_power_scale: float = 1.0

    serving_signal_power_scale: float = 1.0

    failed_bs_index: int | None = None


    def __post_init__(
        self,
    ) -> None:

        if self.start_tti < 0:
            raise ValueError(
                "start_tti must be non-negative."
            )

        if not self.name:
            raise ValueError(
                "phase name cannot be empty."
            )

        for value, field in (
            (
                self.non_serving_interference_power_scale,
                "non_serving_interference_power_scale",
            ),
            (
                self.serving_signal_power_scale,
                "serving_signal_power_scale",
            ),
        ):
            if (
                not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{field} must be finite "
                    "and non-negative."
                )

        if (
            self.failed_bs_index is not None
            and self.failed_bs_index < 0
        ):
            raise ValueError(
                "failed_bs_index cannot be negative."
            )


class PhysicalNetworkScenario:
    """
    Piecewise-constant persistent network-event
    schedule.
    """

    def __init__(
        self,
        phases: tuple[
            PhysicalNetworkPhase,
            ...
        ],
    ) -> None:

        if not phases:
            raise ValueError(
                "At least one phase is required."
            )

        ordered = tuple(
            sorted(
                phases,
                key=lambda phase: phase.start_tti,
            )
        )

        if ordered[0].start_tti != 0:
            raise ValueError(
                "First phase must start at TTI 0."
            )

        starts = [
            phase.start_tti
            for phase in ordered
        ]

        if len(set(starts)) != len(starts):
            raise ValueError(
                "Phase start TTIs must be unique."
            )

        self.phases = ordered


    def phase_for_tti(
        self,
        tti_index: int,
    ) -> PhysicalNetworkPhase:

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        selected = self.phases[0]

        for phase in self.phases:

            if phase.start_tti > tti_index:
                break

            selected = phase

        return selected
