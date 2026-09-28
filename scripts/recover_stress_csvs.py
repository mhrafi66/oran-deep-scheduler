from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(
    "experiments/stress_tests/parallel"
)

TAGS = [
    "csi_ppo_radio_d1",
    "csi_td_rate_only_d1",
]


for tag in TAGS:

    path = (
        ROOT
        / f"{tag}_kpi.csv"
    )

    backup = (
        ROOT
        / f"{tag}_kpi.before_recovery.csv"
    )

    print()
    print("=" * 72)
    print(tag)
    print("=" * 72)

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open() as f:
        reader = csv.DictReader(f)

        fieldnames = (
            reader.fieldnames
        )

        rows = list(reader)

    print(
        "rows before:",
        len(rows),
    )

    #
    # Walk backward.
    #
    # This ensures that if an old run and a newer run
    # contain the same TTI, the newest occurrence wins.
    #
    latest_by_tti = {}

    for row in reversed(rows):

        tti = int(
            row["tti"]
        )

        if (
            0 <= tti <= 99
            and tti not in latest_by_tti
        ):
            latest_by_tti[
                tti
            ] = row

    missing = [
        tti
        for tti in range(100)
        if tti not in latest_by_tti
    ]

    if missing:
        print(
            "CANNOT RECOVER COMPLETE RUN."
        )

        print(
            "Missing TTIs:",
            missing,
        )

        continue

    recovered = [
        latest_by_tti[
            tti
        ]
        for tti in range(100)
    ]

    #
    # Preserve the original contaminated file.
    #
    shutil.copy2(
        path,
        backup,
    )

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            recovered
        )

    print(
        "RECOVERED: 100 TTIs"
    )

    print(
        "backup:",
        backup
    )

    print(
        "clean:",
        path
    )
