from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys


if len(sys.argv) != 2:
    raise SystemExit(
        "usage: "
        "verify_wave5_network_smoke.py "
        "<slurm-array-job-id>"
    )


job_id = sys.argv[1]

output_dir = Path(
    "output2"
)

expected_tasks = range(
    5
)

marker = (
    "WAVE5_NETWORK_CAPABILITY_PASS"
)

failures = []


print(
    "=" * 72
)

print(
    "WAVE-5 NETWORK SMOKE VERIFICATION"
)

print(
    "=" * 72
)


for task_id in expected_tasks:

    path = (
        output_dir
        / f"wave5-net-{job_id}_{task_id}.out"
    )

    if not path.exists():

        failures.append(
            (
                task_id,
                "output file missing",
            )
        )

        print(
            f"task {task_id}: MISSING"
        )

        continue


    text = path.read_text(
        errors="replace"
    )


    if marker not in text:

        failures.append(
            (
                task_id,
                "PASS marker missing",
            )
        )

        print(
            f"task {task_id}: FAIL"
        )

        continue


    suspicious = [
        token
        for token in (
            "Traceback (most recent call last)",
            "CUDA out of memory",
            "RuntimeError:",
            "ValueError:",
            "UnboundLocalError:",
            "TypeError:",
        )
        if token in text
    ]


    if suspicious:

        failures.append(
            (
                task_id,
                "error token(s): "
                + ", ".join(
                    suspicious
                ),
            )
        )

        print(
            f"task {task_id}: FAIL "
            f"({', '.join(suspicious)})"
        )

        continue


    print(
        f"task {task_id}: PASS"
    )


print(
    "=" * 72
)


if failures:

    print(
        "SMOKE VERIFICATION FAILED"
    )

    for task_id, reason in failures:

        print(
            f"task {task_id}: {reason}"
        )

    raise SystemExit(
        1
    )


stamp = Path(
    "experiments/stress_tests/"
    "wave5_network_smoke_passed.txt"
)

stamp.parent.mkdir(
    parents=True,
    exist_ok=True,
)


stamp.write_text(
    "\n".join(
        [
            f"job_id={job_id}",
            (
                "verified_at_utc="
                + datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "tasks=0,1,2,3,4",
            "status=PASS",
            "",
        ]
    )
)


print(
    "ALL 5 WAVE-5 CAPABILITY TASKS PASSED"
)

print(
    f"Certification written to: {stamp}"
)
