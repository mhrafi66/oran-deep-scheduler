#!/bin/bash
#SBATCH --job-name=oran-clean
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-16%17
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail

REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler"

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

cd "${REPO}"

if [[ -z "${RUN_ID:-}" ]]; then
    echo "ERROR: RUN_ID was not supplied."
    exit 1
fi

case "${SLURM_ARRAY_TASK_ID}" in

    # ========================================================
    # COMPLETE CSI DELAY SWEEP
    # ========================================================

    0)
        NAME="csi_delay_d0"
        DELAY="0"
        MODE="all"
        FB="0.5"
        ;;

    1)
        NAME="csi_delay_d1"
        DELAY="1"
        MODE="all"
        FB="0.5"
        ;;

    2)
        NAME="csi_delay_d2"
        DELAY="2"
        MODE="all"
        FB="0.5"
        ;;

    3)
        NAME="csi_delay_d4"
        DELAY="4"
        MODE="all"
        FB="0.5"
        ;;

    4)
        NAME="csi_delay_d8"
        DELAY="8"
        MODE="all"
        FB="0.5"
        ;;

    5)
        NAME="csi_delay_d16"
        DELAY="16"
        MODE="all"
        FB="0.5"
        ;;


    # ========================================================
    # CSI FAILURE DECOMPOSITION
    #
    # All use d=1 because our current channel realizations
    # are temporally independent.
    # ========================================================

    6)
        NAME="csi_ppo_radio_d1"
        DELAY="1"
        MODE="ppo_radio"
        FB="0.5"
        ;;

    7)
        NAME="csi_td_rate_only_d1"
        DELAY="1"
        MODE="td_rate_only"
        FB="0.5"
        ;;

    8)
        NAME="csi_cqi_only_d1"
        DELAY="1"
        MODE="cqi_only"
        FB="0.5"
        ;;

    9)
        NAME="csi_rank_only_d1"
        DELAY="1"
        MODE="rank_only"
        FB="0.5"
        ;;

    10)
        NAME="csi_precoder_only_d1"
        DELAY="1"
        MODE="precoder_only"
        FB="0.5"
        ;;

    # Integrity control:
    # d=1 exists, but no observation field is delayed.
    11)
        NAME="csi_none_d1"
        DELAY="1"
        MODE="none"
        FB="0.5"
        ;;


    # ========================================================
    # TRAFFIC DISTRIBUTION SHIFT
    #
    # CSI is completely fresh because delay=0.
    # ========================================================

    12)
        NAME="traffic_fb000"
        DELAY="0"
        MODE="all"
        FB="0.0"
        ;;

    13)
        NAME="traffic_fb025"
        DELAY="0"
        MODE="all"
        FB="0.25"
        ;;

    14)
        NAME="traffic_fb050"
        DELAY="0"
        MODE="all"
        FB="0.5"
        ;;

    15)
        NAME="traffic_fb075"
        DELAY="0"
        MODE="all"
        FB="0.75"
        ;;

    16)
        NAME="traffic_fb100"
        DELAY="0"
        MODE="all"
        FB="1.0"
        ;;

    *)
        echo "Invalid array task."
        exit 1
        ;;
esac


TAG="${RUN_ID}_${NAME}"

echo "============================================================"
echo "CLEAN STRESS EXPERIMENT"
echo "============================================================"
echo "Run ID:       ${RUN_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Experiment:   ${NAME}"
echo "Output tag:   ${TAG}"
echo "Node:         $(hostname)"
echo "CSI delay:    ${DELAY}"
echo "CSI mode:     ${MODE}"
echo "FB fraction:  ${FB}"
echo "============================================================"

nvidia-smi \
    --query-gpu=name,memory.total \
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
echo "CLEAN_EXPERIMENT_COMPLETED ${NAME}"
echo "============================================================"
