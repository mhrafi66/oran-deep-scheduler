#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST="experiments/stress_tests/wave4_temporal_manifest.csv"

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing ${MANIFEST}"
    exit 1
fi


TASKS=$(
    python - <<'PY'
import csv

with open(
    "experiments/stress_tests/wave4_temporal_manifest.csv"
) as f:
    print(
        sum(
            1
            for _ in csv.DictReader(f)
        )
    )
PY
)


if [[ "${TASKS}" -le 0 ]]; then
    echo "Manifest has no jobs."
    exit 1
fi


LAST_TASK=$((TASKS - 1))


RUN_ID="wave4temp_$(date +%Y%m%d_%H%M%S)"


RUN_DIR="experiments/stress_tests/wave4_runs/${RUN_ID}"

mkdir -p "${RUN_DIR}"


cp \
  "${MANIFEST}" \
  "${RUN_DIR}/manifest.csv"


echo "============================================================"
echo "WAVE-4 TEMPORAL CAMPAIGN"
echo "============================================================"
echo "Run ID:       ${RUN_ID}"
echo "Tasks:        ${TASKS}"
echo "Array:        0-${LAST_TASK}"
echo "Concurrency:  32"
echo "============================================================"


JOB_ID=$(
    sbatch \
      --parsable \
      --array="0-${LAST_TASK}%32" \
      --export=ALL,WAVE4_RUN_ID="${RUN_ID}" \
      scripts/run_wave4_temporal_array.slurm
)


echo "${RUN_ID}" \
  > experiments/stress_tests/wave4_runs/latest_run_id.txt

echo "${JOB_ID}" \
  > "${RUN_DIR}/slurm_job_id.txt"


echo
echo "Submitted job ${JOB_ID}"
echo "Run ID: ${RUN_ID}"
