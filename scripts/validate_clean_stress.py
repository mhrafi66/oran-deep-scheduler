from __future__ import annotations

import csv
import math
from pathlib import Path


RUN_ID_PATH = Path(
    "experiments/stress_tests/"
    "clean_runs/latest_run_id.txt"
)

RUN_ID = (
    RUN_ID_PATH
    .read_text()
    .strip()
)

ROOT = Path(
    "experiments/stress_tests/parallel"
)


EXPERIMENTS = [
    "csi_delay_d0",
    "csi_delay_d1",
    "csi_delay_d2",
    "csi_delay_d4",
    "csi_delay_d8",
    "csi_delay_d16",
    "csi_ppo_radio_d1",
    "csi_td_rate_only_d1",
    "csi_cqi_only_d1",
    "csi_rank_only_d1",
    "csi_precoder_only_d1",
    "csi_none_d1",
    "traffic_fb000",
    "traffic_fb025",
    "traffic_fb050",
    "traffic_fb075",
    "traffic_fb100",
]


print()
print(f"Validating run: {RUN_ID}")
print("=" * 78)

all_ok = True


for name in EXPERIMENTS:

    path = (
        ROOT
        / f"{RUN_ID}_{name}_kpi.csv"
    )

    problems = []

    if not path.exists():
        problems.append(
            "missing"
        )

        rows = []

    else:
        with path.open() as f:
            rows = list(
                csv.DictReader(f)
            )

        if len(rows) != 100:
            problems.append(
                f"rows={len(rows)}"
            )

        if rows:

            try:
                ttis = [
                    int(row["tti"])
                    for row in rows
                ]

                if ttis != list(
                    range(100)
                ):
                    problems.append(
                        "TTIs != 0..99"
                    )

            except Exception as exc:
                problems.append(
                    f"bad TTI field: {exc}"
                )

            for i, row in enumerate(
                rows
            ):
                for key, value in (
                    row.items()
                ):

                    if key == "tti":
                        continue

                    try:
                        x = float(value)
                    except (
                        ValueError,
                        TypeError,
                    ):
                        continue

                    if not math.isfinite(
                        x
                    ):
                        problems.append(
                            f"non-finite "
                            f"{key}@{i}"
                        )
                        break

                if problems:
                    break

    status = (
        "PASS"
        if not problems
        else "FAIL"
    )

    if problems:
        all_ok = False

    detail = (
        "100 TTIs"
        if not problems
        else ", ".join(problems)
    )

    print(
        f"{name:<28}"
        f"{status:<7}"
        f"{detail}"
    )


print("=" * 78)

if all_ok:
    print(
        "ALL 17 CLEAN STRESS RUNS PASSED"
    )
else:
    print(
        "DO NOT AGGREGATE YET"
    )

print()
