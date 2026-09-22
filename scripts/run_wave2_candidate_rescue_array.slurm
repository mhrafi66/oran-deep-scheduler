#!/bin/bash
#SBATCH --job-name=wave2-rescue
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --array=0-3%4
#SBATCH --output=./output2/wave2-rescue-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail


REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler-wave2"

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

ORIGINAL_REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler"


cd "${REPO}"

export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"


if [[ -z "${RUN_ID:-}" ]]; then
    echo "ERROR: RUN_ID was not supplied."
    exit 1
fi


export TRAINED_CHECKPOINT_PATH="${ORIGINAL_REPO}/experiments/checkpoints/paper_1lds_ppo_500tti.pt"


# ============================================================
# COMMON EXPERIMENT CONFIGURATION
# ============================================================

export NUM_EVAL_TTIS=30

export RUN_SEED=1234

export TOPOLOGY_SEED=42

export NUM_UT_PER_SECTOR=20

export NUM_USER_SLOTS=4


# Training-distribution traffic.
export FB_FRACTION=0.5

export FTP3_RATE_SCALE=1.0

export FTP3_PACKET_SIZE_SCALE=1.0


# ============================================================
# BASE FAILURE
#
# Use the existing one-TTI full scheduler-side CSI mismatch.
#
# IMPORTANT:
# With the current independent channel realizations, this is
# an observation-mismatch experiment, NOT a realistic statement
# about one physical TTI of channel aging.
# ============================================================

export CSI_DELAY_TTIS=1

export CSI_STALE_MODE="all"


# ============================================================
# NO EXTRA SYNTHETIC CORRUPTION
# ============================================================

export CQI_BIAS=0.0

export CQI_NOISE_STD=0.0

export CQI_QUANT_STEP=0.0

export FLATTEN_SUBBAND_CQI=0


export RANK_FLIP_PROB=0.0

export RANK_FORCE=0


export PRECODER_UE_SHUFFLE=0

export PRECODER_RBG_SHUFFLE=0


export TD_RATE_LOG_NOISE_STD=0.0

export TD_RATE_DROPOUT_PROB=0.0

export TD_RATE_SHUFFLE=0


export CORRUPTION_SEED=918273


# ============================================================
# 2 x 2 CAUSAL INTERVENTION
# ============================================================

case "${SLURM_ARRAY_TASK_ID}" in

    0)
        LABEL="stale_candidates__stale_features"

        export CANDIDATE_INTERVENTION_MODE="native"
        ;;

    1)
        LABEL="fresh_candidates__stale_features"

        export CANDIDATE_INTERVENTION_MODE="fresh_candidates"
        ;;

    2)
        LABEL="stale_candidates__fresh_features"

        export CANDIDATE_INTERVENTION_MODE="fresh_features"
        ;;

    3)
        LABEL="fresh_candidates__fresh_features"

        export CANDIDATE_INTERVENTION_MODE="fresh_both"
        ;;

    *)
        echo "ERROR: invalid array task."
        exit 1
        ;;

esac


export STRESS_TAG="${RUN_ID}__${LABEL}"


echo "============================================================"
echo "WAVE-2 CANDIDATE RESCUE"
echo "============================================================"

echo "Run ID:               ${RUN_ID}"
echo "Slurm job:             ${SLURM_JOB_ID}"
echo "Array task:            ${SLURM_ARRAY_TASK_ID}"
echo "Node:                  $(hostname)"
echo
echo "Condition:             ${LABEL}"
echo "Intervention:          ${CANDIDATE_INTERVENTION_MODE}"
echo
echo "CSI delay:             ${CSI_DELAY_TTIS}"
echo "CSI stale mode:        ${CSI_STALE_MODE}"
echo
echo "Run seed:              ${RUN_SEED}"
echo "Topology seed:         ${TOPOLOGY_SEED}"
echo
echo "Evaluation TTIs:       ${NUM_EVAL_TTIS}"
echo
echo "Output tag:"
echo "  ${STRESS_TAG}"
echo "============================================================"

echo

nvidia-smi \
    --query-gpu=name,memory.total \
    --format=csv,noheader

echo


"${PYTHON}" -u \
    scripts/run_robustness_eval.py


echo
echo "============================================================"
echo "WAVE2_RESCUE_COMPLETED"
echo "Run ID:    ${RUN_ID}"
echo "Condition: ${LABEL}"
echo "============================================================"
