from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import os
import time

import torch


_ENABLED = (
    os.environ.get(
        "ORAN_PERF_PROFILE",
        "0",
    )
    == "1"
)

_SECONDS: dict[str, float] = defaultdict(
    float
)

_CALLS: dict[str, int] = defaultdict(
    int
)


def perf_enabled() -> bool:
    return _ENABLED


def _synchronize(
    device: str | torch.device | None,
) -> None:
    if device is None:
        return

    resolved = torch.device(
        device
    )

    if (
        resolved.type == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(
            resolved
        )


@contextmanager
def perf_region(
    name: str,
    *,
    device: (
        str
        | torch.device
        | None
    ) = None,
):
    """
    Measure synchronized wall-clock time.

    Profiling is completely disabled unless:

        ORAN_PERF_PROFILE=1

    CUDA synchronization is intentional here because
    we need physical phase timings rather than merely
    kernel-launch timings.
    """

    if not _ENABLED:
        yield
        return

    _synchronize(
        device
    )

    start = time.perf_counter()

    try:
        yield

    finally:
        _synchronize(
            device
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        _SECONDS[name] += elapsed
        _CALLS[name] += 1


def reset_perf_timings() -> None:
    _SECONDS.clear()
    _CALLS.clear()


def get_perf_timings() -> tuple[
    tuple[
        str,
        float,
        int,
    ],
    ...,
]:
    rows = [
        (
            name,
            seconds,
            _CALLS[name],
        )
        for name, seconds
        in _SECONDS.items()
    ]

    rows.sort(
        key=lambda row: row[1],
        reverse=True,
    )

    return tuple(
        rows
    )


def print_perf_timings() -> None:
    rows = get_perf_timings()

    print()
    print("=" * 72)
    print("ORAN SYNCHRONIZED PERFORMANCE PROFILE")
    print("=" * 72)

    if not rows:
        print(
            "No timing samples recorded."
        )
        print("=" * 72)
        return

    for (
        name,
        seconds,
        calls,
    ) in rows:

        per_call_ms = (
            seconds
            / float(calls)
            * 1000.0
        )

        print(
            f"{name:<34} "
            f"{seconds:>9.3f} s   "
            f"{calls:>5} calls   "
            f"{per_call_ms:>10.3f} ms/call"
        )

    print("=" * 72)