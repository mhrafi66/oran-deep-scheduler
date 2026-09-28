#!/bin/bash
#SBATCH --job-name=wave9-eval
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-74%24
#SBATCH --output=./output2/wave9-eval-%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

set -euo pipefail


cd "${SLURM_SUBMIT_DIR}"

mkdir -p output2

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler

unset PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"


PYTHON="$(
    command -v python
)"


EXPORTS="$(
    "${PYTHON}" - \
        "${SLURM_ARRAY_TASK_ID}" <<'PY'
import csv
import shlex
import sys
from pathlib import Path


task_id = int(
    sys.argv[1]
)

manifest = Path(
    "experiments/stress_tests/"
    "wave9_eval_manifest.csv"
)

with manifest.open() as handle:

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
    )
    == task_id
]

if len(matches) != 1:
    raise SystemExit(
        f"Expected one manifest row "
        f"for task {task_id}; "
        f"found {len(matches)}."
    )


row = matches[0]


mapping = {
    "num_ttis": "W9_NUM_TTIS",
    "speed_kmh": "W9_SPEED_KMH",
    "hysteresis_db": "W9_HYSTERESIS_DB",
    "ttt_ttis": "W9_TTT_TTIS",
    "expect_handover": "W9_EXPECT_HANDOVER",
    "min_advantage_db": "W9_MIN_ADVANTAGE_DB",
    "topology_seed": "W9_TOPOLOGY_SEED",
    "association_seed": "W9_ASSOCIATION_SEED",
    "mimo_seed": "W9_MIMO_SEED",
    "measurement_seed": "W9_MEASUREMENT_SEED",
    "traffic_seed": "W9_TRAFFIC_SEED",
    "local_traffic_seed": "W9_LOCAL_TRAFFIC_SEED",
}


for key, env_name in mapping.items():

    print(
        "export "
        f"{env_name}="
        f"{shlex.quote(str(row[key]))}"
    )


for key in (
    "family",
    "condition",
    "replicate",
):

    env_name = (
        "W9_"
        + key.upper()
    )

    print(
        "export "
        f"{env_name}="
        f"{shlex.quote(str(row[key]))}"
    )
PY
)"


eval "${EXPORTS}"


echo "============================================================"
echo "WAVE-9 STATISTICAL HANDOVER EVALUATION"
echo "============================================================"
echo "Array job:          ${SLURM_ARRAY_JOB_ID}"
echo "Task:               ${SLURM_ARRAY_TASK_ID}"
echo "Node:               $(hostname)"
echo "Commit:             $(git rev-parse HEAD)"
echo
echo "Family:             ${W9_FAMILY}"
echo "Condition:          ${W9_CONDITION}"
echo "Replicate:          ${W9_REPLICATE}"
echo "Speed:              ${W9_SPEED_KMH} km/h"
echo "Hysteresis:         ${W9_HYSTERESIS_DB} dB"
echo "TTT:                ${W9_TTT_TTIS} TTIs"
echo "Expect handover:    ${W9_EXPECT_HANDOVER}"
echo "Evaluation TTIs:    ${W9_NUM_TTIS}"
echo "Topology seed:      ${W9_TOPOLOGY_SEED}"
echo "MIMO seed:          ${W9_MIMO_SEED}"
echo "Measurement seed:   ${W9_MEASUREMENT_SEED}"
echo "Traffic seed:       ${W9_TRAFFIC_SEED}"
echo "============================================================"

nvidia-smi \
  --query-gpu=name,memory.total \
  --format=csv,noheader

echo


"${PYTHON}" -u \
  scripts/run_wave9_eval_one.py


echo
echo "WAVE9_EVAL_JOB_PASS"
