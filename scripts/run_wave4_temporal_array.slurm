#!/bin/bash
#SBATCH --job-name=wave4-temp
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=./output2/wave4-temp-%A_%a.out
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p output2
mkdir -p experiments/stress_tests/parallel

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler

unset PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"


MANIFEST="${SLURM_SUBMIT_DIR}/experiments/stress_tests/wave4_temporal_manifest.csv"

eval "$(
    python \
      scripts/read_wave4_temporal_manifest_row.py \
      "${MANIFEST}" \
      "${SLURM_ARRAY_TASK_ID}"
)"


#
# Optional smoke override.
#
if [[ -n "${WAVE4_NUM_TTIS_OVERRIDE:-}" ]]; then
    export NUM_EVAL_TTIS="${WAVE4_NUM_TTIS_OVERRIDE}"
fi


export TEMPORAL_RADIO_MODE="velocity_window"

export SIM_TTI_DURATION_MS="1.0"

export TEMPORAL_MAX_DISPLACEMENT_M="20.0"

export TRAINED_CHECKPOINT_PATH="${HOME}/oran-deep-scheduler/experiments/checkpoints/paper_1lds_ppo_500tti.pt"


export STRESS_TAG="${WAVE4_RUN_ID}__${W4_CONDITION}__r${W4_REPLICATE}"


echo "============================================================"
echo "WAVE-4 TEMPORAL ROBUSTNESS"
echo "============================================================"
echo "Job:               ${SLURM_JOB_ID}"
echo "Task:              ${SLURM_ARRAY_TASK_ID}"
echo "Node:              $(hostname)"
echo "Run ID:            ${WAVE4_RUN_ID}"
echo "Condition:         ${W4_CONDITION}"
echo "Replicate:         ${W4_REPLICATE}"
echo "Speed km/h:        ${UT_SPEED_KMH}"
echo "CSI delay:         ${CSI_DELAY_TTIS}"
echo "CSI stale mode:    ${CSI_STALE_MODE}"
echo "Topology seed:     ${TOPOLOGY_SEED}"
echo "Run seed:          ${RUN_SEED}"
echo "Eval TTIs:         ${NUM_EVAL_TTIS}"
echo "Temporal window:   ${TEMPORAL_WINDOW_TTIS}"
echo "============================================================"


python \
  scripts/run_robustness_eval.py


echo
echo "WAVE4_TEMPORAL_TASK_PASS"
