#!/bin/bash

set -euo pipefail


cd "$(dirname "$0")/.."


MANIFEST="experiments/stress_tests/wave5_network_manifest.csv"

SMOKE_CERT="experiments/stress_tests/wave5_network_smoke_passed.txt"


if [[ ! -f "${SMOKE_CERT}" ]]; then

    echo "ERROR:"
    echo
    echo "Wave-5 GPU capability smoke has not been"
    echo "certified yet."
    echo
    echo "Run:"
    echo
    echo "  python scripts/verify_wave5_network_smoke.py <JOB_ID>"
    echo
    echo "after all five smoke tasks complete."
    echo

    exit 1
fi


if [[ ! -f "${MANIFEST}" ]]; then

    echo "Missing manifest:"
    echo "${MANIFEST}"

    exit 1
fi


TASKS=$(
python - <<'PY'
import csv

with open(
    "experiments/stress_tests/"
    "wave5_network_manifest.csv"
) as handle:

    print(
        sum(
            1
            for _ in csv.DictReader(
                handle
            )
        )
    )
PY
)


if [[ "${TASKS}" -ne 132 ]]; then

    echo \
      "Expected 132 tasks; found ${TASKS}."

    exit 1
fi


LAST_TASK=$(( TASKS - 1 ))


RUN_ID="wave5net_$(date +%Y%m%d_%H%M%S)"


RUN_DIR="experiments/stress_tests/wave5_runs/${RUN_ID}"


mkdir -p \
  "${RUN_DIR}"


cp \
  "${MANIFEST}" \
  "${RUN_DIR}/manifest.csv"


cp \
  "${SMOKE_CERT}" \
  "${RUN_DIR}/smoke_certification.txt"


git rev-parse HEAD \
  > "${RUN_DIR}/git_commit.txt"


git status --short \
  > "${RUN_DIR}/git_status_at_submit.txt"


echo "============================================================"

echo "WAVE-5 NETWORK CAMPAIGN"

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
      --export=ALL,WAVE5_RUN_ID="${RUN_ID}" \
      scripts/run_wave5_network_array.slurm
)


mkdir -p \
  experiments/stress_tests/wave5_runs


echo "${RUN_ID}" \
  > experiments/stress_tests/wave5_runs/latest_run_id.txt


echo "${JOB_ID}" \
  > "${RUN_DIR}/slurm_job_id.txt"


echo

echo "Submitted Wave-5 job: ${JOB_ID}"

echo "Run ID: ${RUN_ID}"
