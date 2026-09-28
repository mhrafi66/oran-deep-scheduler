#!/bin/bash
set -euo pipefail

cd ~/oran-deep-scheduler

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

"${PYTHON}" scripts/generate_network_v3_manifest.py

RUN_ID=$(cat experiments/stress_tests/network_v3/latest_run_id.txt)
MANIFEST="experiments/stress_tests/network_v3/${RUN_ID}/manifest.tsv"
N=$(( $(wc -l < "${MANIFEST}") - 1 ))
LAST=$((N - 1))
MAX_PARALLEL="${MAX_PARALLEL:-64}"

echo "Run ID:        ${RUN_ID}"
echo "Array tasks:   ${N}"
echo "Concurrency:   ${MAX_PARALLEL}"
echo

sbatch \
  --array="0-${LAST}%${MAX_PARALLEL}" \
  --export="ALL,RUN_ID=${RUN_ID}" \
  scripts/run_network_v3_array.slurm
