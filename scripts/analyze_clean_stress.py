from __future__ import annotations

import csv
import statistics
from pathlib import Path


RUN_ID = (
    Path(
        "experiments/stress_tests/"
        "clean_runs/latest_run_id.txt"
    )
    .read_text()
    .strip()
)

ROOT = Path(
    "experiments/stress_tests/parallel"
)

START_TTI = 20
END_TTI = 99


def load(
    name: str,
):

    path = (
        ROOT
        / f"{RUN_ID}_{name}_kpi.csv"
    )

    with path.open() as f:
        all_rows = list(
            csv.DictReader(f)
        )

    #
    # Strictly reject contaminated CSVs.
    #
    if len(all_rows) != 100:
        raise RuntimeError(
            f"{name}: total rows "
            f"{len(all_rows)}, expected 100."
        )

    ttis = [
        int(row["tti"])
        for row in all_rows
    ]

    if ttis != list(range(100)):
        raise RuntimeError(
            f"{name}: TTIs are not "
            "exactly 0..99."
        )

    return [
        row
        for row in all_rows
        if (
            START_TTI
            <= int(row["tti"])
            <= END_TTI
        )
    ]


def mean(
    rows,
    field: str,
) -> float:

    return statistics.fmean(
        float(row[field])
        for row in rows
    )


def summarize(
    name: str,
):

    rows = load(name)

    return {
        "name": name,

        "delivered": mean(
            rows,
            "mean_delivered_mbps",
        ),

        "p05": mean(
            rows,
            "p05_history_mbps",
        ),

        "median": mean(
            rows,
            "median_history_mbps",
        ),

        "geomean": mean(
            rows,
            "paper_floor_history_geomean_mbps",
        ),
    }


def print_table(
    title,
    results,
    baseline=None,
):

    print()
    print(title)
    print(
        f"Evaluation window: "
        f"TTI {START_TTI}..{END_TTI}"
    )
    print()

    if baseline is None:

        print(
            f"{'Experiment':<24}"
            f"{'Delivered':>11}"
            f"{'P05':>9}"
            f"{'Median':>10}"
            f"{'GeoMean':>10}"
        )

        print("-" * 64)

    else:

        print(
            f"{'Experiment':<24}"
            f"{'Delivered':>11}"
            f"{'P05':>9}"
            f"{'Median':>10}"
            f"{'GeoMean':>10}"
            f"{'Geo Ret.':>11}"
        )

        print("-" * 75)

    for result in results:

        line = (
            f"{result['name']:<24}"
            f"{result['delivered']:>11.3f}"
            f"{result['p05']:>9.3f}"
            f"{result['median']:>10.3f}"
            f"{result['geomean']:>10.3f}"
        )

        if baseline is not None:

            retention = (
                100
                * result["geomean"]
                / baseline["geomean"]
            )

            line += (
                f"{retention:>10.2f}%"
            )

        print(line)


#
# ==========================================================
# 1. CSI DELAY SWEEP
# ==========================================================
#

delay_names = [
    "csi_delay_d0",
    "csi_delay_d1",
    "csi_delay_d2",
    "csi_delay_d4",
    "csi_delay_d8",
    "csi_delay_d16",
]

delay_results = [
    summarize(name)
    for name in delay_names
]

fresh = delay_results[0]

print_table(
    "CSI STALENESS SWEEP",
    delay_results,
    baseline=fresh,
)


#
# ==========================================================
# 2. CSI COMPONENT DECOMPOSITION
# ==========================================================
#

decomposition_names = [
    "csi_delay_d0",
    "csi_none_d1",
    "csi_delay_d1",
    "csi_ppo_radio_d1",
    "csi_td_rate_only_d1",
    "csi_cqi_only_d1",
    "csi_rank_only_d1",
    "csi_precoder_only_d1",
]

decomposition_results = [
    summarize(name)
    for name in decomposition_names
]

print_table(
    "CSI FAILURE DECOMPOSITION",
    decomposition_results,
    baseline=fresh,
)


#
# ==========================================================
# 3. TRAFFIC DISTRIBUTION SHIFT
# ==========================================================
#

traffic_names = [
    "traffic_fb000",
    "traffic_fb025",
    "traffic_fb050",
    "traffic_fb075",
    "traffic_fb100",
]

traffic_results = [
    summarize(name)
    for name in traffic_names
]

print_table(
    "TRAFFIC DISTRIBUTION SHIFT",
    traffic_results,
)


#
# ==========================================================
# 4. REPRODUCIBILITY CONTROL
# ==========================================================
#

control = summarize(
    "traffic_fb050"
)

print()
print(
    "REPRODUCIBILITY CONTROL"
)
print()

print(
    "Fresh CSI / 50-50 run A "
    f"GeoMean: {fresh['geomean']:.6f}"
)

print(
    "Fresh CSI / 50-50 run B "
    f"GeoMean: {control['geomean']:.6f}"
)

difference = abs(
    fresh["geomean"]
    - control["geomean"]
)

print(
    "Absolute difference:       "
    f"{difference:.9f}"
)

print()
