#!/bin/bash

set -euo pipefail


REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler-wave3"

MANIFEST="${REPO}/experiments/stress_tests/wave3_manifest.csv"

RUN_ROOT="${REPO}/experiments/stress_tests/wave3_runs"


cd "${REPO}"


# ============================================================
# BASIC SAFETY CHECKS
# ============================================================

if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: manifest does not exist:"
    echo "  ${MANIFEST}"
    exit 1
fi


mkdir -p "${RUN_ROOT}"
mkdir -p output2


# ============================================================
# COUNT MANIFEST TASKS
# ============================================================

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"


NUM_TASKS=$(
    "${PYTHON}" - <<'PY'
import csv
from pathlib import Path

path = Path(
    "experiments/stress_tests/"
    "wave3_manifest.csv"
)

with path.open() as f:
    rows = list(
        csv.DictReader(f)
    )

print(
    len(rows)
)
PY
)


if [[ -z "${NUM_TASKS}" ]]; then
    echo "ERROR: failed to determine manifest task count."
    exit 1
fi


if ! [[ "${NUM_TASKS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUM_TASKS is not an integer:"
    echo "  ${NUM_TASKS}"
    exit 1
fi


if (( NUM_TASKS <= 0 )); then
    echo "ERROR: manifest contains no tasks."
    exit 1
fi


LAST_TASK=$((NUM_TASKS - 1))


# ============================================================
# CREATE UNIQUE CAMPAIGN ID
# ============================================================

RUN_ID="wave3_$(date +%Y%m%d_%H%M%S)"

RUN_DIR="${RUN_ROOT}/${RUN_ID}"

mkdir -p "${RUN_DIR}"


echo "${RUN_ID}" > \
    "${RUN_ROOT}/latest_run_id.txt"


#
# Freeze a copy of the exact manifest used for this campaign.
#
cp \
    "${MANIFEST}" \
    "${RUN_DIR}/manifest.csv"


# ============================================================
# CAMPAIGN SUMMARY
# ============================================================

echo
echo "============================================================"
echo "WAVE-3 OVERNIGHT CAMPAIGN"
echo "============================================================"
echo "Run ID:          ${RUN_ID}"
echo "Manifest:        ${MANIFEST}"
echo "Tasks:           ${NUM_TASKS}"
echo "Array:           0-${LAST_TASK}"
echo "Concurrency:     64"
echo "Run directory:   ${RUN_DIR}"
echo "============================================================"
echo


# ============================================================
# SUBMIT
#
# Command-line --array overrides any static #SBATCH --array
# declaration in the worker script.
# ============================================================

SUBMISSION_OUTPUT=$(
    sbatch \
        --array="0-${LAST_TASK}%64" \
        --export=ALL,RUN_ID="${RUN_ID}" \
        scripts/run_wave3_manifest_array.slurm
)


echo "${SUBMISSION_OUTPUT}"


JOB_ID=$(
    echo "${SUBMISSION_OUTPUT}" \
    | awk '{print $4}'
)


if [[ -z "${JOB_ID}" ]]; then
    echo "ERROR: unable to parse Slurm job ID."
    exit 1
fi


echo "${JOB_ID}" > \
    "${RUN_DIR}/slurm_job_id.txt"


# ============================================================
# FINAL SUMMARY
# ============================================================

echo
echo "============================================================"
echo "SUBMISSION COMPLETE"
echo "============================================================"
echo "Run ID:          ${RUN_ID}"
echo "Job ID:          ${JOB_ID}"
echo "Tasks:           ${NUM_TASKS}"
echo "Array:           0-${LAST_TASK}%64"
echo "Run directory:   ${RUN_DIR}"
echo
echo "Monitor with:"
echo
echo "  squeue -r -j ${JOB_ID} -o '%.18i %.8T %.10M %.14R'"
echo
echo "============================================================"
