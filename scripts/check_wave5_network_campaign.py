from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import sys


if len(sys.argv) != 2:

    raise SystemExit(
        "usage: "
        "check_wave5_network_campaign.py "
        "<slurm-array-job-id>"
    )


job_id = sys.argv[1]


manifest_path = Path(
    "experiments/stress_tests/"
    "wave5_network_manifest.csv"
)


with manifest_path.open() as handle:

    rows = list(
        csv.DictReader(
            handle
        )
    )


counts = Counter()

failed = []


for row in rows:

    task_id = int(
        row["task_id"]
    )

    path = Path(
        "output2"
    ) / (
        f"wave5-full-"
        f"{job_id}_{task_id}.out"
    )


    if not path.exists():

        counts[
            "missing"
        ] += 1

        continue


    text = path.read_text(
        errors="replace"
    )


    if (
        "WAVE5_NETWORK_TASK_PASS"
        in text
    ):

        counts[
            "passed"
        ] += 1

        continue


    counts[
        "failed_or_incomplete"
    ] += 1

    failed.append(
        (
            task_id,
            row["condition"],
        )
    )


print(
    "=" * 72
)

print(
    "WAVE-5 NETWORK CAMPAIGN STATUS"
)

print(
    "=" * 72
)

print(
    f"Total tasks:          {len(rows)}"
)

print(
    f"Passed:               {counts['passed']}"
)

print(
    "Failed/incomplete:    "
    f"{counts['failed_or_incomplete']}"
)

print(
    f"No output yet:        {counts['missing']}"
)

print(
    "=" * 72
)


if failed:

    print()

    print(
        "FAILED / INCOMPLETE TASKS:"
    )

    for task_id, condition in failed[:30]:

        print(
            f"  task {task_id:3d}: "
            f"{condition}"
        )

    if len(failed) > 30:

        print(
            f"  ... plus "
            f"{len(failed) - 30} more"
        )
