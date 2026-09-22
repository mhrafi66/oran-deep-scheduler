#!/bin/bash
#SBATCH --job-name=wave4-temp
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=./output2/wave4-temporal-%j.out
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p output2

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler

unset PYTHONPATH

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

echo "============================================================"
echo "WAVE-4 TEMPORAL CHANNEL GPU SMOKE"
echo "============================================================"
echo "Job:       ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Worktree:  ${SLURM_SUBMIT_DIR}"
echo "Python:    $(which python)"
echo "============================================================"

nvidia-smi

python \
  scripts/run_wave4_temporal_channel_smoke.py \
  --device cuda:0 \
  --num-ttis 8 \
  --tti-duration-ms 1.0
