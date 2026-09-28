#!/bin/bash
#SBATCH --job-name=wave2-candidate
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=./output2/wave2-candidate-%j.out
#SBATCH --export=ALL

set -euo pipefail

cd ~/oran-deep-scheduler-wave2

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

#
# Read the trained actor/critic checkpoint from the untouched
# original repository.
#
export TRAINED_CHECKPOINT_PATH="$HOME/oran-deep-scheduler/experiments/checkpoints/paper_1lds_ppo_500tti.pt"

#
# Unique output tag.
#
export STRESS_TAG="wave2_candidate_tdnoise_smoke_${SLURM_JOB_ID}"

#
# Short runtime test only.
#
export NUM_EVAL_TTIS=3

#
# Reproducible topology / traffic.
#
export RUN_SEED=1234
export TOPOLOGY_SEED=42

export NUM_UT_PER_SECTOR=20
export NUM_USER_SLOTS=4

export FB_FRACTION=0.5
export FTP3_RATE_SCALE=1.0
export FTP3_PACKET_SIZE_SCALE=1.0

#
# No delayed-CSI path in this smoke.
#
# We are deliberately isolating instantaneous PF/TDS-rate
# corruption.
#
export CSI_DELAY_TTIS=0
export CSI_STALE_MODE="none"

#
# Deliberately corrupt ONLY TD instantaneous rate.
#
export TD_RATE_LOG_NOISE_STD=0.75
export TD_RATE_DROPOUT_PROB=0.0
export TD_RATE_SHUFFLE=0

#
# All PPO radio features remain clean.
#
export CQI_BIAS=0.0
export CQI_NOISE_STD=0.0
export CQI_QUANT_STEP=0.0
export FLATTEN_SUBBAND_CQI=0

export RANK_FLIP_PROB=0.0
export RANK_FORCE=0

export PRECODER_UE_SHUFFLE=0
export PRECODER_RBG_SHUFFLE=0

export CORRUPTION_SEED=918273


echo "============================================================"
echo "WAVE-2 PF CANDIDATE DIAGNOSTIC SMOKE"
echo "============================================================"
echo "Host:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Repository:  $(pwd)"
echo "Python:      ${PYTHON}"
echo "Stress tag:  ${STRESS_TAG}"
echo "Checkpoint:  ${TRAINED_CHECKPOINT_PATH}"
echo "============================================================"
echo

nvidia-smi \
    --query-gpu=name,memory.total,memory.used \
    --format=csv,noheader

echo

"${PYTHON}" -u \
    scripts/run_robustness_eval.py

echo
echo "============================================================"
echo "WAVE2_CANDIDATE_SMOKE_COMPLETED"
echo "Stress tag: ${STRESS_TAG}"
echo "============================================================"
