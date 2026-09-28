#!/bin/bash
#SBATCH --job-name=wave7-pq
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=./output2/wave7-packet-handover-%j.out
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p output2

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler

unset PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

echo "============================================================"
echo "WAVE-7 PACKET-AWARE REAL-SIONNA HANDOVER"
echo "============================================================"
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Worktree: ${SLURM_SUBMIT_DIR}"
echo "Commit:   $(git rev-parse HEAD)"
echo "============================================================"

nvidia-smi

python -u \
  scripts/run_wave7_packet_handover_smoke.py

echo
echo "WAVE7_PACKET_HANDOVER_JOB_PASS"
