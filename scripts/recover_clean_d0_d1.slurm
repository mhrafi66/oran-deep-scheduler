#!/bin/bash
#SBATCH --job-name=oran-recover
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --requeue
#SBATCH --output=experiments/stress_tests/clean_runs/recover-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail

REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler"

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

cd "${REPO}"

RUN_ID="$(
    cat experiments/stress_tests/clean_runs/latest_run_id.txt
)"

case "${SLURM_ARRAY_TASK_ID}" in

    0)
        DELAY=0
        ;;

    1)
        DELAY=1
        ;;

    *)
        echo "Invalid array index"
        exit 1
        ;;

esac

TAG="${RUN_ID}_csi_delay_d${DELAY}"

echo "============================================================"
echo "RECOVERY RUN"
echo "RUN ID:       ${RUN_ID}"
echo "TAG:          ${TAG}"
echo "CSI DELAY:    ${DELAY}"
echo "HOST:         $(hostname)"
echo "============================================================"

#
# Critical:
# remove ONLY the two contaminated/partial outputs.
#
# The KPI writer can otherwise append to an old CSV.
#
rm -f \
    "experiments/stress_tests/parallel/${TAG}_metrics.csv" \
    "experiments/stress_tests/parallel/${TAG}_kpi.csv" \
    "/tmp/oran_${TAG}_snapshot.pt"

CSI_DELAY_TTIS="${DELAY}" \
CSI_STALE_MODE="all" \
FB_FRACTION="0.5" \
STRESS_TAG="${TAG}" \
NUM_EVAL_TTIS="100" \
"${PYTHON}" -u \
scripts/run_csi_staleness_eval.py

#
# Validate THIS experiment before Slurm is allowed
# to call the job successful.
#
"${PYTHON}" - "${TAG}" <<'PY'
import csv
import math
import sys
from pathlib import Path

tag = sys.argv[1]

path = (
    Path("experiments/stress_tests/parallel")
    / f"{tag}_kpi.csv"
)

assert path.exists(), path

with path.open() as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 100, (
    f"{tag}: expected 100 rows, got {len(rows)}"
)

ttis = [
    int(row["tti"])
    for row in rows
]

assert ttis == list(range(100)), (
    f"{tag}: TTI sequence is not 0..99"
)

for i, row in enumerate(rows):

    for key, value in row.items():

        if key == "tti":
            continue

        try:
            x = float(value)
        except (TypeError, ValueError):
            continue

        assert math.isfinite(x), (
            f"{tag}: non-finite {key} at row {i}"
        )

print(
    f"CLEAN RECOVERY VALIDATION PASSED: {tag}"
)
PY

echo
echo "============================================================"
echo "RECOVERY COMPLETED: ${TAG}"
echo "============================================================"
