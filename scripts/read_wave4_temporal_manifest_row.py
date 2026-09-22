from __future__ import annotations

import csv
import shlex
import sys
from pathlib import Path


if len(sys.argv) != 3:
    raise SystemExit(
        "usage: read_wave4_temporal_manifest_row.py "
        "<manifest.csv> <task_id>"
    )


path = Path(
    sys.argv[1]
)

task_id = int(
    sys.argv[2]
)


with path.open() as handle:

    rows = list(
        csv.DictReader(handle)
    )


matching = [
    row
    for row in rows
    if int(row["task_id"]) == task_id
]


if len(matching) != 1:
    raise RuntimeError(
        f"Expected one row for task {task_id}, "
        f"found {len(matching)}."
    )


row = matching[0]


mapping = {
    "W4_CONDITION": row["condition"],
    "W4_REPLICATE": row["replicate"],
    "UT_SPEED_KMH": row["speed_kmh"],
    "CSI_DELAY_TTIS": row["delay_ttis"],
    "CSI_STALE_MODE": row["stale_mode"],
    "TOPOLOGY_SEED": row["topology_seed"],
    "RUN_SEED": row["run_seed"],
    "NUM_EVAL_TTIS": row["num_eval_ttis"],
    "TEMPORAL_WINDOW_TTIS": (
        row["temporal_window_ttis"]
    ),
}


for key, value in mapping.items():

    print(
        f"export {key}="
        f"{shlex.quote(str(value))}"
    )
