from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import math
import re
import statistics


ROOT = Path(
    "experiments/stress_tests"
)

MANIFEST = (
    ROOT
    / "wave9_eval_manifest.csv"
)

OUTPUT = (
    ROOT
    / "wave9_eval_summary.csv"
)

LOG_ROOT = Path(
    "output2"
)


METRIC_RE = re.compile(
    r"W9_EVAL_METRIC "
    r"tti=(\d+) "
    r"stream=(\d+) "
    r"cell=(\d+) "
    r"total_delivered_mbps=([0-9eE+.\-]+) "
    r"forced_ue_delivered_mbps=([0-9eE+.\-]+)"
)

ADVANTAGE_RE = re.compile(
    r"Initial measured advantage:\s+"
    r"([0-9eE+.\-]+)\s+dB"
)

HANDOVER_RE = re.compile(
    r"FORCED UE HANDOVER OBSERVED:"
    r".*TTI=(\d+)"
)

COMPLETED_HO_RE = re.compile(
    r"Completed handovers total:\s+(\d+)"
)


def mean_or_nan(values):
    if not values:
        return math.nan
    return statistics.mean(values)


def sd_or_nan(values):
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values)


with MANIFEST.open() as handle:

    manifest_rows = list(
        csv.DictReader(
            handle
        )
    )


summary_rows = []


for row in manifest_rows:

    task_id = int(
        row["task_id"]
    )

    matches = sorted(
        LOG_ROOT.glob(
            f"wave9-eval-*_{task_id}.out"
        ),
        key=lambda p: p.stat().st_mtime,
    )

    if not matches:

        summary_rows.append(
            {
                **row,
                "status": "missing",
                "application_pass": 0,
                "handover_observed": 0,
                "handover_tti": "",
                "initial_advantage_db": "",
                "completed_handovers": "",
                "observed_ttis": 0,
                "mean_total_delivered_mbps": "",
                "mean_forced_ue_delivered_mbps": "",
                "pre_handover_total_mbps": "",
                "post_handover_total_mbps": "",
                "pre_handover_forced_ue_mbps": "",
                "post_handover_forced_ue_mbps": "",
                "log_path": "",
            }
        )

        continue


    path = matches[-1]

    text = path.read_text(
        errors="replace"
    )


    if (
        "WAVE9_EVAL_JOB_PASS"
        in text
    ):
        status = "complete"

    elif (
        "DUE TO PREEMPTION"
        in text
    ):
        status = "preempted"

    elif (
        "No UE has a sufficiently strong"
        in text
    ):
        status = "scenario_unavailable"

    elif (
        "never triggered the controlled UE handover"
        in text
    ):
        status = "handover_not_triggered"

    else:
        status = "failed"


    application_pass = int(
        "WAVE9_EVAL_JOB_PASS"
        in text
    )


    advantage_match = (
        ADVANTAGE_RE.search(
            text
        )
    )

    advantage = (
        float(
            advantage_match.group(1)
        )
        if advantage_match
        else math.nan
    )


    ho_matches = list(
        HANDOVER_RE.finditer(
            text
        )
    )

    handover_tti = (
        int(
            ho_matches[0]
            .group(1)
        )
        if ho_matches
        else None
    )


    completed_match = (
        COMPLETED_HO_RE.search(
            text
        )
    )

    completed_handovers = (
        int(
            completed_match.group(1)
        )
        if completed_match
        else None
    )


    total_by_tti = defaultdict(
        float
    )

    forced_by_tti = defaultdict(
        float
    )


    for match in METRIC_RE.finditer(
        text
    ):

        tti = int(
            match.group(1)
        )

        total_by_tti[tti] += float(
            match.group(4)
        )

        forced_by_tti[tti] += float(
            match.group(5)
        )


    ttis = sorted(
        total_by_tti
    )

    total_values = [
        total_by_tti[tti]
        for tti in ttis
    ]

    forced_values = [
        forced_by_tti[tti]
        for tti in ttis
    ]


    if handover_tti is None:

        pre_total = total_values
        post_total = []

        pre_forced = forced_values
        post_forced = []

    else:

        pre_total = [
            total_by_tti[tti]
            for tti in ttis
            if tti < handover_tti
        ]

        post_total = [
            total_by_tti[tti]
            for tti in ttis
            if tti >= handover_tti
        ]

        pre_forced = [
            forced_by_tti[tti]
            for tti in ttis
            if tti < handover_tti
        ]

        post_forced = [
            forced_by_tti[tti]
            for tti in ttis
            if tti >= handover_tti
        ]


    summary_rows.append(
        {
            **row,
            "status": status,
            "application_pass": (
                application_pass
            ),
            "handover_observed": int(
                handover_tti
                is not None
            ),
            "handover_tti": (
                ""
                if handover_tti is None
                else handover_tti
            ),
            "initial_advantage_db": (
                ""
                if math.isnan(advantage)
                else advantage
            ),
            "completed_handovers": (
                ""
                if completed_handovers is None
                else completed_handovers
            ),
            "observed_ttis": len(
                ttis
            ),
            "mean_total_delivered_mbps": (
                mean_or_nan(
                    total_values
                )
            ),
            "mean_forced_ue_delivered_mbps": (
                mean_or_nan(
                    forced_values
                )
            ),
            "pre_handover_total_mbps": (
                mean_or_nan(
                    pre_total
                )
            ),
            "post_handover_total_mbps": (
                mean_or_nan(
                    post_total
                )
            ),
            "pre_handover_forced_ue_mbps": (
                mean_or_nan(
                    pre_forced
                )
            ),
            "post_handover_forced_ue_mbps": (
                mean_or_nan(
                    post_forced
                )
            ),
            "log_path": str(
                path
            ),
        }
    )


fieldnames = list(
    summary_rows[0].keys()
)

with OUTPUT.open(
    "w",
    newline="",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


print("=" * 90)
print("WAVE-9 EVALUATION STATUS")
print("=" * 90)

status_counts = defaultdict(
    int
)

for row in summary_rows:
    status_counts[
        row["status"]
    ] += 1

for status in sorted(
    status_counts
):
    print(
        f"{status:<28s} "
        f"{status_counts[status]:>3d}"
    )

print()
print(
    f"Summary CSV: {OUTPUT}"
)


# ============================================================
# GROUP SUMMARIES
# ============================================================

groups = defaultdict(
    list
)

for row in summary_rows:

    key = (
        row["family"],
        float(
            row["speed_kmh"]
        ),
        float(
            row["hysteresis_db"]
        ),
        int(
            row["ttt_ttis"]
        ),
        int(
            row["expect_handover"]
        ),
    )

    groups[key].append(
        row
    )


print()
print("=" * 90)
print("GROUP RESULTS")
print("=" * 90)

for key in sorted(
    groups
):

    (
        family,
        speed,
        hysteresis,
        ttt,
        expect_handover,
    ) = key

    rows = groups[key]

    complete = [
        row
        for row in rows
        if row["status"]
        == "complete"
    ]

    if expect_handover:

        successful = [
            row
            for row in rows
            if int(
                row["handover_observed"]
            )
            == 1
        ]

    else:

        successful = [
            row
            for row in rows
            if (
                row["status"]
                == "complete"
                and int(
                    row["handover_observed"]
                )
                == 0
            )
        ]


    ho_ttis = [
        float(
            row["handover_tti"]
        )
        for row in complete
        if str(
            row["handover_tti"]
        )
        != ""
    ]

    total_rates = [
        float(
            row[
                "mean_total_delivered_mbps"
            ]
        )
        for row in complete
    ]

    forced_rates = [
        float(
            row[
                "mean_forced_ue_delivered_mbps"
            ]
        )
        for row in complete
    ]


    print()
    print(
        f"{family} | "
        f"speed={speed:g} | "
        f"hyst={hysteresis:g} | "
        f"TTT={ttt} | "
        f"expect_HO={expect_handover}"
    )

    print(
        "  complete:       "
        f"{len(complete)}/{len(rows)}"
    )

    print(
        "  desired outcome:"
        f" {len(successful)}/{len(rows)}"
    )

    if ho_ttis:

        print(
            "  HO TTI:         "
            f"{mean_or_nan(ho_ttis):.3f}"
            " ± "
            f"{sd_or_nan(ho_ttis):.3f}"
        )

    if total_rates:

        print(
            "  total Mbps:     "
            f"{mean_or_nan(total_rates):.3f}"
            " ± "
            f"{sd_or_nan(total_rates):.3f}"
        )

        print(
            "  controlled UE:  "
            f"{mean_or_nan(forced_rates):.3f}"
            " ± "
            f"{sd_or_nan(forced_rates):.3f}"
            " Mbps"
        )


# ============================================================
# PAIRED ENABLED-vs-BLOCKED COMPARISON
# ============================================================

enabled = {}

blocked = {}


for row in summary_rows:

    if (
        row["status"]
        != "complete"
    ):
        continue

    replicate = int(
        row["replicate"]
    )

    is_enabled_reference = (
        row["family"]
        == "speed_ttt"
        and float(
            row["speed_kmh"]
        )
        == 30.0
        and float(
            row["hysteresis_db"]
        )
        == 1.0
        and int(
            row["ttt_ttis"]
        )
        == 2
    )

    is_blocked = (
        row["family"]
        == "handover_counterfactual"
    )

    if is_enabled_reference:
        enabled[
            replicate
        ] = row

    if is_blocked:
        blocked[
            replicate
        ] = row


paired_replicates = sorted(
    set(enabled)
    & set(blocked)
)


if paired_replicates:

    total_deltas = []

    forced_deltas = []

    for replicate in paired_replicates:

        enabled_row = (
            enabled[replicate]
        )

        blocked_row = (
            blocked[replicate]
        )

        total_delta = (
            float(
                enabled_row[
                    "mean_total_delivered_mbps"
                ]
            )
            -
            float(
                blocked_row[
                    "mean_total_delivered_mbps"
                ]
            )
        )

        forced_delta = (
            float(
                enabled_row[
                    "mean_forced_ue_delivered_mbps"
                ]
            )
            -
            float(
                blocked_row[
                    "mean_forced_ue_delivered_mbps"
                ]
            )
        )

        total_deltas.append(
            total_delta
        )

        forced_deltas.append(
            forced_delta
        )


    print()
    print("=" * 90)
    print(
        "PAIRED 30 km/h HANDOVER "
        "ENABLED - BLOCKED"
    )
    print("=" * 90)

    print(
        f"Paired realizations: "
        f"{len(paired_replicates)}"
    )

    print(
        "Total throughput delta: "
        f"{mean_or_nan(total_deltas):+.3f}"
        " ± "
        f"{sd_or_nan(total_deltas):.3f}"
        " Mbps"
    )

    print(
        "Controlled-UE delta:    "
        f"{mean_or_nan(forced_deltas):+.3f}"
        " ± "
        f"{sd_or_nan(forced_deltas):.3f}"
        " Mbps"
    )


expected = len(
    manifest_rows
)

complete = sum(
    row["status"]
    == "complete"
    for row in summary_rows
)


print()
print("=" * 90)

if complete == expected:

    print(
        "WAVE9_EVAL_DATASET_PASS"
    )

else:

    print(
        "WAVE9_EVAL_DATASET_INCOMPLETE"
    )

print(
    f"Complete: {complete}/{expected}"
)

print("=" * 90)
