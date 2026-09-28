from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ControlLoopTimingConfig:
    """
    Synthetic control-loop timing model.

    This does NOT claim to measure actual inference
    runtime.

    It models a configured compute/control latency
    distribution and an execution deadline.

    miss_policy:
        "empty"
            execute no scheduling action.

        "repeat_previous"
            replay the previous successful schedule,
            identity-safe where possible.
    """

    deadline_ms: float

    base_compute_ms: float

    jitter_std_ms: float = 0.0

    miss_policy: str = "empty"

    seed: int = 1729


    def __post_init__(
        self,
    ) -> None:

        for value, name in (
            (
                self.deadline_ms,
                "deadline_ms",
            ),
            (
                self.base_compute_ms,
                "base_compute_ms",
            ),
            (
                self.jitter_std_ms,
                "jitter_std_ms",
            ),
        ):
            if (
                not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be finite "
                    "and non-negative."
                )

        if self.deadline_ms <= 0.0:
            raise ValueError(
                "deadline_ms must be positive."
            )

        if self.miss_policy not in {
            "empty",
            "repeat_previous",
        }:
            raise ValueError(
                "Unsupported miss_policy."
            )


@dataclass(frozen=True)
class ControlLoopDecision:
    tti_index: int
    latency_ms: float
    deadline_missed: bool
    miss_policy: str


class ControlLoopTimingModel:
    def __init__(
        self,
        config: ControlLoopTimingConfig,
    ) -> None:

        self.config = config


    def decision(
        self,
        tti_index: int,
    ) -> ControlLoopDecision:

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        jitter = 0.0

        if (
            self.config.jitter_std_ms
            > 0.0
        ):
            generator = torch.Generator(
                device="cpu"
            )

            generator.manual_seed(
                self.config.seed
                + tti_index
            )

            jitter = float(
                torch.randn(
                    (),
                    generator=generator,
                ).item()
                * self.config.jitter_std_ms
            )

        latency = max(
            0.0,
            (
                self.config.base_compute_ms
                + jitter
            ),
        )

        return ControlLoopDecision(
            tti_index=tti_index,

            latency_ms=latency,

            deadline_missed=(
                latency
                > self.config.deadline_ms
            ),

            miss_policy=(
                self.config.miss_policy
            ),
        )
