from __future__ import annotations

import csv
from pathlib import Path


OUTPUT = Path(
    "experiments/stress_tests/"
    "wave9_eval_manifest.csv"
)


REPLICATES = (
    {
        "replicate": 0,
        "topology_seed": 42,
        "association_seed": 1000,
        "mimo_seed": 2000,
        "measurement_seed": 9100,
        "traffic_seed": 1234,
        "local_traffic_seed": 5000,
    },
    {
        "replicate": 1,
        "topology_seed": 43,
        "association_seed": 1100,
        "mimo_seed": 2100,
        "measurement_seed": 9200,
        "traffic_seed": 2234,
        "local_traffic_seed": 5100,
    },
    {
        "replicate": 2,
        "topology_seed": 44,
        "association_seed": 1200,
        "mimo_seed": 2200,
        "measurement_seed": 9300,
        "traffic_seed": 3234,
        "local_traffic_seed": 5200,
    },
    {
        "replicate": 3,
        "topology_seed": 45,
        "association_seed": 1300,
        "mimo_seed": 2300,
        "measurement_seed": 9400,
        "traffic_seed": 4234,
        "local_traffic_seed": 5300,
    },
    {
        "replicate": 4,
        "topology_seed": 46,
        "association_seed": 1400,
        "mimo_seed": 2400,
        "measurement_seed": 9500,
        "traffic_seed": 5234,
        "local_traffic_seed": 5400,
    },
)


rows = []
task_id = 0


def add_condition(
    *,
    family: str,
    condition: str,
    speed_kmh: float,
    hysteresis_db: float,
    ttt_ttis: int,
    expect_handover: bool,
) -> None:

    global task_id

    for rep in REPLICATES:

        rows.append(
            {
                "task_id": task_id,
                "family": family,
                "condition": condition,
                "replicate": rep["replicate"],
                "speed_kmh": speed_kmh,
                "hysteresis_db": hysteresis_db,
                "ttt_ttis": ttt_ttis,
                "expect_handover": (
                    1
                    if expect_handover
                    else 0
                ),
                "num_ttis": 8,
                "min_advantage_db": 6.0,
                "topology_seed": (
                    rep["topology_seed"]
                ),
                "association_seed": (
                    rep["association_seed"]
                ),
                "mimo_seed": (
                    rep["mimo_seed"]
                ),
                "measurement_seed": (
                    rep["measurement_seed"]
                ),
                "traffic_seed": (
                    rep["traffic_seed"]
                ),
                "local_traffic_seed": (
                    rep["local_traffic_seed"]
                ),
            }
        )

        task_id += 1


# ============================================================
# A. MOBILITY x TTT
#
# 4 speeds x 3 TTT values x 5 paired realizations = 60
# ============================================================

for speed in (
    0.0,
    3.0,
    30.0,
    120.0,
):

    for ttt in (
        1,
        2,
        4,
    ):

        add_condition(
            family="speed_ttt",
            condition=(
                f"speed_{speed:g}"
                f"__ttt_{ttt}"
                "__hyst_1"
            ),
            speed_kmh=speed,
            hysteresis_db=1.0,
            ttt_ttis=ttt,
            expect_handover=True,
        )


# ============================================================
# B. HYSTERESIS SENSITIVITY
#
# H=1 dB already exists in the core campaign above.
#
# Add:
#   H=0 dB
#   H=3 dB
#
# at:
#   speed=30 km/h
#   TTT=2
#
# 2 x 5 = 10 additional runs.
# ============================================================

for hysteresis in (
    0.0,
    3.0,
):

    add_condition(
        family="hysteresis",
        condition=(
            "speed_30"
            "__ttt_2"
            f"__hyst_{hysteresis:g}"
        ),
        speed_kmh=30.0,
        hysteresis_db=hysteresis,
        ttt_ttis=2,
        expect_handover=True,
    )


# ============================================================
# C. PAIRED NO-HANDOVER COUNTERFACTUAL
#
# Same seeds as:
#   speed=30
#   TTT=2
#   hysteresis=1 dB
#
# except hysteresis=100 dB, effectively preventing the
# controlled mismatch from being corrected.
#
# The matching enabled runs already exist in section A.
#
# 5 additional runs.
# ============================================================

add_condition(
    family="handover_counterfactual",
    condition=(
        "speed_30"
        "__ttt_2"
        "__handover_blocked"
    ),
    speed_kmh=30.0,
    hysteresis_db=100.0,
    ttt_ttis=2,
    expect_handover=False,
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT.open(
    "w",
    newline="",
) as handle:

    fieldnames = [
        "task_id",
        "family",
        "condition",
        "replicate",
        "speed_kmh",
        "hysteresis_db",
        "ttt_ttis",
        "expect_handover",
        "num_ttis",
        "min_advantage_db",
        "topology_seed",
        "association_seed",
        "mimo_seed",
        "measurement_seed",
        "traffic_seed",
        "local_traffic_seed",
    ]

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


print(
    f"Wrote {len(rows)} Wave-9 evaluation tasks."
)

assert len(rows) == 75
