from __future__ import annotations

import csv
from pathlib import Path


OUTPUT = Path(
    "experiments/stress_tests/"
    "wave4_temporal_manifest.csv"
)


SPEEDS_KMH = (
    0.0,
    3.0,
    30.0,
    120.0,
)

DELAYS = (
    1,
    2,
    4,
)

STALE_MODES = (
    "all",
    "td_rate_only",
    "ppo_radio",
)

REPLICATES = (
    (0, 42, 310000),
    (1, 43, 320000),
    (2, 44, 330000),
)


rows = []

task_id = 0


#
# Clean controls.
#
for speed in SPEEDS_KMH:

    condition = (
        f"speed_{speed:g}_clean"
    )

    for (
        replicate,
        topology_seed,
        run_seed,
    ) in REPLICATES:

        rows.append(
            {
                "task_id": task_id,
                "condition": condition,
                "replicate": replicate,
                "speed_kmh": speed,
                "delay_ttis": 0,
                "stale_mode": "none",
                "topology_seed": topology_seed,
                "run_seed": run_seed,
                "num_eval_ttis": 6,
                "temporal_window_ttis": 8,
            }
        )

        task_id += 1


#
# CSI-aging decomposition.
#
for speed in SPEEDS_KMH:

    for delay in DELAYS:

        for stale_mode in STALE_MODES:

            condition = (
                f"speed_{speed:g}"
                f"_d{delay}"
                f"_{stale_mode}"
            )

            for (
                replicate,
                topology_seed,
                run_seed,
            ) in REPLICATES:

                rows.append(
                    {
                        "task_id": task_id,
                        "condition": condition,
                        "replicate": replicate,
                        "speed_kmh": speed,
                        "delay_ttis": delay,
                        "stale_mode": stale_mode,
                        "topology_seed": topology_seed,
                        "run_seed": run_seed,
                        "num_eval_ttis": 6,
                        "temporal_window_ttis": 8,
                    }
                )

                task_id += 1


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with OUTPUT.open(
    "w",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "task_id",
            "condition",
            "replicate",
            "speed_kmh",
            "delay_ttis",
            "stale_mode",
            "topology_seed",
            "run_seed",
            "num_eval_ttis",
            "temporal_window_ttis",
        ],
    )

    writer.writeheader()

    writer.writerows(rows)


conditions = sorted(
    {
        row["condition"]
        for row in rows
    }
)


print(
    f"Manifest: {OUTPUT}"
)

print(
    f"Conditions: {len(conditions)}"
)

print(
    f"Replicates: {len(REPLICATES)}"
)

print(
    f"Total jobs: {len(rows)}"
)
