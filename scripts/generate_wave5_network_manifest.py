from __future__ import annotations

import csv
from pathlib import Path


OUTPUT = Path(
    "experiments/stress_tests/"
    "wave5_network_manifest.csv"
)


SPEEDS_KMH = (
    0.0,
    3.0,
    30.0,
    120.0,
)


FAMILIES = (
    "clean",

    "csi_d2_all",

    "csi_d2_td_rate_only",

    "csi_d2_ppo_radio",

    "interference_x4",

    "serving_power_0p25",

    "availability_25pct",

    "execution_delay_1",

    "execution_delay_2",

    "deadline_empty",

    "deadline_repeat",
)


REPLICATES = (
    {
        "replicate": 0,
        "topology_seed": 42,
        "run_seed": 410000,
    },

    {
        "replicate": 1,
        "topology_seed": 43,
        "run_seed": 420000,
    },

    {
        "replicate": 2,
        "topology_seed": 44,
        "run_seed": 430000,
    },
)


rows = []

task_id = 0


for speed_kmh in SPEEDS_KMH:

    for family in FAMILIES:

        condition = (
            f"speed_{speed_kmh:g}"
            f"__{family}"
        )

        for replicate in REPLICATES:

            rows.append(
                {
                    "task_id": task_id,

                    "condition": (
                        condition
                    ),

                    "family": family,

                    "replicate": (
                        replicate[
                            "replicate"
                        ]
                    ),

                    "speed_kmh": (
                        speed_kmh
                    ),

                    "topology_seed": (
                        replicate[
                            "topology_seed"
                        ]
                    ),

                    "run_seed": (
                        replicate[
                            "run_seed"
                        ]
                    ),

                    #
                    # Entire run stays inside one
                    # 8-sample temporal window:
                    # TTIs 0..7.
                    #
                    "num_eval_ttis": 8,

                    "temporal_window_ttis": 8,

                    #
                    # 30-kHz NR scheduler slot:
                    #
                    "tti_duration_ms": 0.5,
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
            "family",
            "replicate",
            "speed_kmh",
            "topology_seed",
            "run_seed",
            "num_eval_ttis",
            "temporal_window_ttis",
            "tti_duration_ms",
        ],
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print(
    f"Manifest: {OUTPUT}"
)

print(
    f"Speeds: {len(SPEEDS_KMH)}"
)

print(
    f"Families: {len(FAMILIES)}"
)

print(
    f"Replicates: {len(REPLICATES)}"
)

print(
    f"Conditions: "
    f"{len(SPEEDS_KMH) * len(FAMILIES)}"
)

print(
    f"Total tasks: {len(rows)}"
)
