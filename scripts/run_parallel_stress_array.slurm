#!/bin/bash
#SBATCH --job-name=oran-stress
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-10%11
#SBATCH --output=experiments/stress_tests/parallel/slurm-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail

REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler"

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

cd "${REPO}"

#
# ----------------------------------------------------------
# Tasks 0-5:
# CSI feature decomposition.
#
# All use delay=1 because with the current independent
# temporal model any nonzero delay already creates strong
# channel mismatch.
# ----------------------------------------------------------
#

case "${SLURM_ARRAY_TASK_ID}" in

    0)
        TAG="csi_all_d1"
        DELAY="1"
        MODE="all"
        FB="0.5"
        ;;

    1)
        TAG="csi_ppo_radio_d1"
        DELAY="1"
        MODE="ppo_radio"
        FB="0.5"
        ;;

    2)
        TAG="csi_td_rate_only_d1"
        DELAY="1"
        MODE="td_rate_only"
        FB="0.5"
        ;;

    3)
        TAG="csi_cqi_only_d1"
        DELAY="1"
        MODE="cqi_only"
        FB="0.5"
        ;;

    4)
        TAG="csi_rank_only_d1"
        DELAY="1"
        MODE="rank_only"
        FB="0.5"
        ;;

    5)
        TAG="csi_precoder_only_d1"
        DELAY="1"
        MODE="precoder_only"
        FB="0.5"
        ;;


    #
    # ------------------------------------------------------
    # Tasks 6-10:
    # Traffic-distribution shift.
    #
    # CSI remains fresh.
    # ------------------------------------------------------
    #

    6)
        TAG="traffic_fb000"
        DELAY="0"
        MODE="all"
        FB="0.0"
        ;;

    7)
        TAG="traffic_fb025"
        DELAY="0"
        MODE="all"
        FB="0.25"
        ;;

    8)
        TAG="traffic_fb050"
        DELAY="0"
        MODE="all"
        FB="0.5"
        ;;

    9)
        TAG="traffic_fb075"
        DELAY="0"
        MODE="all"
        FB="0.75"
        ;;

    10)
        TAG="traffic_fb100"
        DELAY="0"
        MODE="all"
        FB="1.0"
        ;;

    *)
        echo "Unknown array index."
        exit 1
        ;;
esac


echo "============================================================"
echo "SLURM JOB:       ${SLURM_JOB_ID}"
echo "ARRAY INDEX:     ${SLURM_ARRAY_TASK_ID}"
echo "HOST:            $(hostname)"
echo "TAG:             ${TAG}"
echo "CSI DELAY:       ${DELAY}"
echo "CSI MODE:        ${MODE}"
echo "FB FRACTION:     ${FB}"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total \
    --format=csv,noheader

echo


CSI_DELAY_TTIS="${DELAY}" \
CSI_STALE_MODE="${MODE}" \
FB_FRACTION="${FB}" \
STRESS_TAG="${TAG}" \
NUM_EVAL_TTIS=100 \
"${PYTHON}" -u \
scripts/run_csi_staleness_eval.py


echo
echo "============================================================"
echo "COMPLETED: ${TAG}"
echo "============================================================"
