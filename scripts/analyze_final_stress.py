from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(
    "experiments/stress_tests/parallel"
)

FRESH_ROOT = Path(
    "experiments/stress_tests/csi_staleness"
)

START_TTI = 20
END_TTI = 99


# ============================================================
# IMPORTANT:
#
# These are the files we now consider canonical.
#
# The two *_recovery files replace the contaminated/incomplete
# files from the earlier manual experiments.
# ============================================================

CSI_EXPERIMENTS = [
    (
        "fresh",
        FRESH_ROOT
        / "delay_0_kpi.csv",
    ),

    (
        "all_stale",
        ROOT
        / "csi_all_d1_kpi.csv",
    ),

    (
        "ppo_radio_stale",
        ROOT
        / "csi_ppo_radio_d1_recovery_kpi.csv",
    ),

    (
        "td_rate_stale",
        ROOT
        / "csi_td_rate_only_d1_recovery_kpi.csv",
    ),

    (
        "cqi_stale",
        ROOT
        / "csi_cqi_only_d1_kpi.csv",
    ),

    (
        "rank_stale",
        ROOT
        / "csi_rank_only_d1_kpi.csv",
    ),

    (
        "precoder_stale",
        ROOT
        / "csi_precoder_only_d1_kpi.csv",
    ),
]


TRAFFIC_EXPERIMENTS = [
    (
        "FB_0",
        ROOT
        / "traffic_fb000_kpi.csv",
    ),

    (
        "FB_25",
        ROOT
        / "traffic_fb025_kpi.csv",
    ),

    (
        "FB_50",
        ROOT
        / "traffic_fb050_kpi.csv",
    ),

    (
        "FB_75",
        ROOT
        / "traffic_fb075_kpi.csv",
    ),

    (
        "FB_100",
        ROOT
        / "traffic_fb100_kpi.csv",
    ),
]


def load_and_validate(
    name: str,
    path: Path,
):

    if not path.exists():
        raise FileNotFoundError(
            f"{name}: missing file:\n"
            f"{path}"
        )

    with path.open() as f:

        rows = list(
            csv.DictReader(f)
        )

    if len(rows) != 100:
        raise RuntimeError(
            f"{name}: expected 100 rows, "
            f"got {len(rows)}\n"
            f"{path}"
        )

    ttis = [
        int(row["tti"])
        for row in rows
    ]

    if ttis != list(
        range(100)
    ):
        raise RuntimeError(
            f"{name}: TTIs are not "
            f"exactly 0..99\n"
            f"{path}"
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
                TypeError,
                ValueError,
            ):
                continue

            if not math.isfinite(x):
                raise RuntimeError(
                    f"{name}: non-finite "
                    f"{key} at row {i}"
                )

    eval_rows = [
        row
        for row in rows
        if (
            START_TTI
            <= int(row["tti"])
            <= END_TTI
        )
    ]

    if len(eval_rows) != 80:
        raise RuntimeError(
            f"{name}: expected 80 "
            f"evaluation rows, got "
            f"{len(eval_rows)}"
        )

    return eval_rows


def avg(
    rows,
    field: str,
) -> float:

    return statistics.fmean(
        float(row[field])
        for row in rows
    )


def summarize(
    name: str,
    path: Path,
):

    rows = load_and_validate(
        name,
        path,
    )

    return {
        "experiment": name,

        "delivered": avg(
            rows,
            "mean_delivered_mbps",
        ),

        "p05": avg(
            rows,
            "p05_history_mbps",
        ),

        "median": avg(
            rows,
            "median_history_mbps",
        ),

        "geomean": avg(
            rows,
            "paper_floor_history_geomean_mbps",
        ),
    }


print()
print(
    "VALIDATING FINAL STRESS DATA"
)

print("=" * 72)

all_experiments = (
    CSI_EXPERIMENTS
    + TRAFFIC_EXPERIMENTS
)

for name, path in (
    all_experiments
):

    load_and_validate(
        name,
        path,
    )

    print(
        f"{name:<22}"
        "PASS -- 100 TTIs"
    )

print("=" * 72)

print(
    "ALL FINAL STRESS FILES PASSED"
)


# ============================================================
# CSI decomposition
# ============================================================

csi_results = [
    summarize(
        name,
        path,
    )
    for name, path
    in CSI_EXPERIMENTS
]

fresh = csi_results[0]


print()
print()
print(
    "CSI ROBUSTNESS DECOMPOSITION"
)

print(
    f"Evaluation window: "
    f"TTI {START_TTI}..{END_TTI}"
)

print()

print(
    f"{'Experiment':<20}"
    f"{'Delivered':>11}"
    f"{'P05':>9}"
    f"{'Median':>10}"
    f"{'GeoMean':>10}"
    f"{'Geo Ret.':>11}"
)

print("-" * 71)

for result in csi_results:

    retention = (
        100.0
        * result["geomean"]
        / fresh["geomean"]
    )

    print(
        f"{result['experiment']:<20}"
        f"{result['delivered']:>11.3f}"
        f"{result['p05']:>9.3f}"
        f"{result['median']:>10.3f}"
        f"{result['geomean']:>10.3f}"
        f"{retention:>10.2f}%"
    )


# ============================================================
# Traffic distribution shift
# ============================================================

traffic_results = [
    summarize(
        name,
        path,
    )
    for name, path
    in TRAFFIC_EXPERIMENTS
]


print()
print()
print(
    "TRAFFIC-DISTRIBUTION SHIFT"
)

print(
    f"Evaluation window: "
    f"TTI {START_TTI}..{END_TTI}"
)

print()

print(
    f"{'FB %':<10}"
    f"{'Delivered':>11}"
    f"{'P05':>9}"
    f"{'Median':>10}"
    f"{'GeoMean':>10}"
)

print("-" * 50)

for result in traffic_results:

    fb = (
        result["experiment"]
        .replace(
            "FB_",
            "",
        )
    )

    print(
        f"{fb:<10}"
        f"{result['delivered']:>11.3f}"
        f"{result['p05']:>9.3f}"
        f"{result['median']:>10.3f}"
        f"{result['geomean']:>10.3f}"
    )


# ============================================================
# Save compact summary
# ============================================================

output = (
    ROOT
    / "final_stress_summary.csv"
)

with output.open(
    "w",
    newline="",
) as f:

    fieldnames = [
        "group",
        "experiment",
        "delivered_mbps",
        "p05_mbps",
        "median_mbps",
        "geomean_mbps",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for result in csi_results:

        writer.writerow(
            {
                "group": "csi",
                "experiment": (
                    result[
                        "experiment"
                    ]
                ),
                "delivered_mbps": (
                    result[
                        "delivered"
                    ]
                ),
                "p05_mbps": (
                    result["p05"]
                ),
                "median_mbps": (
                    result[
                        "median"
                    ]
                ),
                "geomean_mbps": (
                    result[
                        "geomean"
                    ]
                ),
            }
        )

    for result in traffic_results:

        writer.writerow(
            {
                "group": "traffic",
                "experiment": (
                    result[
                        "experiment"
                    ]
                ),
                "delivered_mbps": (
                    result[
                        "delivered"
                    ]
                ),
                "p05_mbps": (
                    result["p05"]
                ),
                "median_mbps": (
                    result[
                        "median"
                    ]
                ),
                "geomean_mbps": (
                    result[
                        "geomean"
                    ]
                ),
            }
        )


print()
print()
print(
    "FINAL SUMMARY:"
)

print(output)

print()
