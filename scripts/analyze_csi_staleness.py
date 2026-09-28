from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


DELAYS = (
    0,
    1,
    2,
    4,
    8,
    16,
)

START_TTI = 20
END_TTI = 99

ROOT = Path(
    "experiments/stress_tests/"
    "csi_staleness"
)

OUTPUT_PATH = (
    ROOT
    / "csi_staleness_summary.csv"
)


def mean(
    values,
):
    return statistics.fmean(
        values
    )


def std(
    values,
):
    if len(values) < 2:
        return 0.0

    return statistics.stdev(
        values
    )


results = []


for delay in DELAYS:

    path = (
        ROOT
        / f"delay_{delay}_kpi.csv"
    )

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

    if len(rows) != (
        END_TTI
        - START_TTI
        + 1
    ):
        raise RuntimeError(
            f"Delay {delay}: expected "
            f"{END_TTI - START_TTI + 1} "
            f"evaluation rows, got "
            f"{len(rows)}."
        )

    delivered = [
        float(
            row[
                "mean_delivered_mbps"
            ]
        )
        for row in rows
    ]

    p05 = [
        float(
            row[
                "p05_history_mbps"
            ]
        )
        for row in rows
    ]

    median = [
        float(
            row[
                "median_history_mbps"
            ]
        )
        for row in rows
    ]

    geomean = [
        float(
            row[
                "paper_floor_history_geomean_mbps"
            ]
        )
        for row in rows
    ]

    last = rows[-1]

    results.append(
        {
            "delay_ttis": delay,
            "num_eval_ttis": len(
                rows
            ),

            "mean_delivered_mbps": (
                mean(delivered)
            ),

            "std_delivered_mbps": (
                std(delivered)
            ),

            "mean_p05_history_mbps": (
                mean(p05)
            ),

            "mean_median_history_mbps": (
                mean(median)
            ),

            "mean_geomean_history_mbps": (
                mean(geomean)
            ),

            "final_p05_history_mbps": (
                float(
                    last[
                        "p05_history_mbps"
                    ]
                )
            ),

            "final_median_history_mbps": (
                float(
                    last[
                        "median_history_mbps"
                    ]
                )
            ),

            "final_geomean_history_mbps": (
                float(
                    last[
                        "paper_floor_history_geomean_mbps"
                    ]
                )
            ),
        }
    )


baseline = results[0]

for result in results:

    result[
        "delivered_retention_pct"
    ] = (
        100.0
        * result[
            "mean_delivered_mbps"
        ]
        / baseline[
            "mean_delivered_mbps"
        ]
    )

    result[
        "geomean_retention_pct"
    ] = (
        100.0
        * result[
            "mean_geomean_history_mbps"
        ]
        / baseline[
            "mean_geomean_history_mbps"
        ]
    )

    result[
        "p05_retention_pct"
    ] = (
        100.0
        * result[
            "mean_p05_history_mbps"
        ]
        / baseline[
            "mean_p05_history_mbps"
        ]
    )

    result[
        "median_retention_pct"
    ] = (
        100.0
        * result[
            "mean_median_history_mbps"
        ]
        / baseline[
            "mean_median_history_mbps"
        ]
    )


fieldnames = list(
    results[0].keys()
)

with OUTPUT_PATH.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        results
    )


print()
print(
    "CSI STALENESS RESULTS"
)

print(
    f"Evaluation window: "
    f"TTI {START_TTI}..{END_TTI}"
)

print()

header = (
    f"{'Delay':>5}  "
    f"{'Delivered':>10}  "
    f"{'P05':>8}  "
    f"{'Median':>8}  "
    f"{'GeoMean':>8}  "
    f"{'Del Ret.':>9}  "
    f"{'Geo Ret.':>9}"
)

print(header)

print(
    "-" * len(header)
)

for r in results:

    print(
        f"{r['delay_ttis']:>5d}  "
        f"{r['mean_delivered_mbps']:>10.3f}  "
        f"{r['mean_p05_history_mbps']:>8.3f}  "
        f"{r['mean_median_history_mbps']:>8.3f}  "
        f"{r['mean_geomean_history_mbps']:>8.3f}  "
        f"{r['delivered_retention_pct']:>8.2f}%  "
        f"{r['geomean_retention_pct']:>8.2f}%"
    )


print()
print(
    "Final TTI-99 history metrics"
)

print()

header = (
    f"{'Delay':>5}  "
    f"{'P05':>8}  "
    f"{'Median':>8}  "
    f"{'GeoMean':>8}"
)

print(header)

print(
    "-" * len(header)
)

for r in results:

    print(
        f"{r['delay_ttis']:>5d}  "
        f"{r['final_p05_history_mbps']:>8.3f}  "
        f"{r['final_median_history_mbps']:>8.3f}  "
        f"{r['final_geomean_history_mbps']:>8.3f}"
    )


print()
print(
    "Summary written to:"
)

print(
    OUTPUT_PATH
)
