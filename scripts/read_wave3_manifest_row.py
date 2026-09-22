from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


MANIFEST = Path(
    "experiments/stress_tests/"
    "wave3_manifest.csv"
)


def main() -> None:

    if len(
        sys.argv
    ) != 2:
        raise SystemExit(
            "Usage: read_wave3_manifest_row.py TASK_ID"
        )

    task_id = int(
        sys.argv[
            1
        ]
    )

    with MANIFEST.open() as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    if not (
        0
        <= task_id
        < len(
            rows
        )
    ):
        raise ValueError(
            f"Invalid task_id {task_id}."
        )

    row = rows[
        task_id
    ]

    if int(
        row[
            "task_id"
        ]
    ) != task_id:
        raise RuntimeError(
            "Manifest task IDs are not contiguous."
        )

    print(
        json.dumps(
            row,
            separators=(
                ",",
                ":",
            ),
        )
    )


if __name__ == "__main__":
    main()
