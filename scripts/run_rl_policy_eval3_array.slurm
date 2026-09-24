#!/bin/bash
#SBATCH --job-name=rl-pol3
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-29%12
#SBATCH --output=./output2/rl-pol3-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p output2

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler

unset PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

: "${RL_EVAL3_RUN_ID:?RL_EVAL3_RUN_ID required}"
: "${RL_EVAL3_OUTPUT_ROOT:?RL_EVAL3_OUTPUT_ROOT required}"
: "${PPO_CHECKPOINT:?PPO_CHECKPOINT required}"

mkdir -p \
    "${RL_EVAL3_OUTPUT_ROOT}/${RL_EVAL3_RUN_ID}"

echo "============================================================"
echo "FROZEN PPO RL POLICY EVALUATION — SET 3"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Array:      ${SLURM_ARRAY_JOB_ID}"
echo "Task:       ${SLURM_ARRAY_TASK_ID}"
echo "Node:       $(hostname)"
echo "Worktree:   ${SLURM_SUBMIT_DIR}"
echo "Commit:     $(git rev-parse HEAD)"
echo "Run ID:     ${RL_EVAL3_RUN_ID}"
echo "Checkpoint: ${PPO_CHECKPOINT}"
echo "============================================================"

nvidia-smi \
    --query-gpu=name,memory.total \
    --format=csv,noheader

echo

python -u \
    scripts/rl_eval/run_frozen_actor_capacity_diagnostics.py \
    --task-id "${SLURM_ARRAY_TASK_ID}"

echo
echo "RL_POLICY_EVAL3_ARRAY_TASK_PASS"
