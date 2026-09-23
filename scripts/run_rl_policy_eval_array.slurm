#!/bin/bash
#SBATCH --job-name=rl-policy
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-14%8
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u1472438/oran_experiments/rl_eval/slurm/rl-policy-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler

unset PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

: "${RL_EVAL_RUN_ID:?RL_EVAL_RUN_ID is required}"
: "${RL_EVAL_OUTPUT_ROOT:?RL_EVAL_OUTPUT_ROOT is required}"
: "${PPO_CHECKPOINT:?PPO_CHECKPOINT is required}"

mkdir -p \
    "${RL_EVAL_OUTPUT_ROOT}/${RL_EVAL_RUN_ID}"

echo "============================================================"
echo "FROZEN PPO RL POLICY EVALUATION"
echo "============================================================"
echo "Job:       ${SLURM_JOB_ID}"
echo "Array:     ${SLURM_ARRAY_JOB_ID}"
echo "Task:      ${SLURM_ARRAY_TASK_ID}"
echo "Node:      $(hostname)"
echo "Worktree:  ${SLURM_SUBMIT_DIR}"
echo "Commit:    $(git rev-parse HEAD)"
echo "Run ID:    ${RL_EVAL_RUN_ID}"
echo "Checkpoint:${PPO_CHECKPOINT}"
echo "============================================================"

nvidia-smi

python -u \
    scripts/rl_eval/run_frozen_actor_diagnostics.py \
    --task-id "${SLURM_ARRAY_TASK_ID}"

echo
echo "RL_POLICY_ARRAY_TASK_PASS"
