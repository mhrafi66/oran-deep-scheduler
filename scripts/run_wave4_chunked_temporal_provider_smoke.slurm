#!/bin/bash
#SBATCH --job-name=wave4-prov
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=./output2/wave4-provider-%j.out
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p output2

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler

unset PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

echo "============================================================"
echo "WAVE-4 CHUNKED TEMPORAL PROVIDER"
echo "============================================================"
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Worktree: ${SLURM_SUBMIT_DIR}"
echo "============================================================"

nvidia-smi

python \
  scripts/run_wave4_chunked_temporal_provider_smoke.py
