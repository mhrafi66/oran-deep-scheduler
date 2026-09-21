from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(
    "experiments/stress_tests/parallel"
)

START_TTI = 20
END_TTI = 99


EXPERIMENTS = [
    #
    # CSI decomposition
    #
    (
        "fresh",
        Path(
            "experiments/stress_tests/"
            "csi_staleness/delay_0_kpi.csv"
        ),
    ),
    (
        "all_stale",
        ROOT / "csi_all_d1_kpi.csv",
    ),
    (
        "ppo_radio_stale",
        ROOT / "csi_ppo_radio_d1_recovery_kpi.csv",
    ),
    (
        "td_rate_stale",
        ROOT / "csi_td_rate_only_d1_recovery_kpi.csv",
    ),
    (
        "cqi_stale",
        ROOT / "csi_cqi_only_d1_kpi.csv",
    ),
    (
        "rank_stale",
        ROOT / "csi_rank_only_d1_kpi.csv",
    ),
    (
        "precoder_stale",
        ROOT / "csi_precoder_only_d1_kpi.csv",
    ),

    #
    # Traffic shift
    #
    (
        "traffic_FB_0",
        ROOT / "traffic_fb000_kpi.csv",
    ),
    (
        "traffic_FB_25",
        ROOT / "traffic_fb025_kpi.csv",
    ),
    (
        "traffic_FB_50",
        ROOT / "traffic_fb050_kpi.csv",
    ),
    (
        "traffic_FB_75",
        ROOT / "traffic_fb075_kpi.csv",
    ),
    (
        "traffic_FB_100",
        ROOT / "traffic_fb100_kpi.csv",
    ),
]


def load_rows(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open() as f:
        rows = list(
            csv.DictReader(f)
        )

    rows = [
        row
        for row in rows
        if (
            START_TTI
            <= int(row["tti"])
            <= END_TTI
        )
    ]

    expected = (
        END_TTI
        - START_TTI
        + 1
    )

    if len(rows) != expected:
        raise RuntimeError(
            f"{path}: expected "
            f"{expected} rows in evaluation "
            f"window, got {len(rows)}."
        )

    return rows


def avg(
    rows,
    field,
):
    return statistics.fmean(
        float(row[field])
        for row in rows
    )


results = []

for name, path in EXPERIMENTS:

    rows = load_rows(
        path
    )

    results.append(
        {
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
    )


fresh = next(
    result
    for result in results
    if result["experiment"] == "fresh"
)


print()
print(
    "CSI ROBUSTNESS DECOMPOSITION"
)
print(
    f"TTIs {START_TTI}..{END_TTI}"
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

for result in results[:7]:

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


print()
print(
    "TRAFFIC-DISTRIBUTION SHIFT"
)
print(
    f"TTIs {START_TTI}..{END_TTI}"
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

for result in results[7:]:

    fb = (
        result["experiment"]
        .replace(
            "traffic_FB_",
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
