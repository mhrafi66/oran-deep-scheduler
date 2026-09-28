#!/bin/bash

set -euo pipefail


REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler-wave2"

cd "${REPO}"


RUN_ROOT="experiments/stress_tests/wave2_candidate_rescue"

mkdir -p "${RUN_ROOT}"

mkdir -p output2


RUN_ID="wave2rescue_$(date +%Y%m%d_%H%M%S)"


echo "${RUN_ID}" > \
    "${RUN_ROOT}/latest_run_id.txt"


RUN_DIR="${RUN_ROOT}/${RUN_ID}"

mkdir -p "${RUN_DIR}"


cat > "${RUN_DIR}/README.txt" <<EOF2
Run ID: ${RUN_ID}

Experiment:
2x2 candidate/PPO-feature causal rescue.

Task 0:
stale candidates + stale PPO features

Task 1:
fresh candidates + stale PPO features

Task 2:
stale candidates + fresh PPO features

Task 3:
fresh candidates + fresh PPO features

Common stress:
CSI_DELAY_TTIS=1
CSI_STALE_MODE=all

TTIs:
30

Topology seed:
42

Run seed:
1234
EOF2


echo "Run ID:      ${RUN_ID}"
echo "Conditions:  4"
echo


SUBMISSION_OUTPUT=$(
    sbatch \
        --export=ALL,RUN_ID="${RUN_ID}" \
        scripts/run_wave2_candidate_rescue_array.slurm
)


echo "${SUBMISSION_OUTPUT}"


JOB_ID=$(
    echo "${SUBMISSION_OUTPUT}" \
    | awk '{print $4}'
)


echo "${JOB_ID}" > \
    "${RUN_DIR}/slurm_job_id.txt"


echo
echo "Job ID:      ${JOB_ID}"
echo "Run folder:  ${RUN_DIR}"
