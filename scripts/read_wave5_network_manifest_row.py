from __future__ import annotations

import csv
import shlex
import sys
from pathlib import Path


if len(sys.argv) != 3:

    raise SystemExit(
        "usage: "
        "read_wave5_network_manifest_row.py "
        "<manifest.csv> <task-id>"
    )


path = Path(
    sys.argv[1]
)

task_id = int(
    sys.argv[2]
)


with path.open() as handle:

    rows = list(
        csv.DictReader(
            handle
        )
    )


matches = [
    row
    for row in rows
    if int(
        row["task_id"]
    ) == task_id
]


if len(matches) != 1:

    raise RuntimeError(
        f"Expected exactly one manifest row "
        f"for task {task_id}; "
        f"found {len(matches)}."
    )


row = matches[0]


mapping = {
    "W5_CONDITION":
        row["condition"],

    "W5_FAMILY":
        row["family"],

    "W5_REPLICATE":
        row["replicate"],

    "UT_SPEED_KMH":
        row["speed_kmh"],

    "TOPOLOGY_SEED":
        row["topology_seed"],

    "RUN_SEED":
        row["run_seed"],

    "NUM_EVAL_TTIS":
        row["num_eval_ttis"],

    "TEMPORAL_WINDOW_TTIS":
        row[
            "temporal_window_ttis"
        ],

    "SIM_TTI_DURATION_MS":
        row[
            "tti_duration_ms"
        ],
}


for key, value in mapping.items():

    print(
        f"export {key}="
        f"{shlex.quote(str(value))}"
    )
