#!/bin/bash
#SBATCH --job-name=wave3-smoke
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-8%9
#SBATCH --output=./output2/wave3-smoke-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail

REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler-wave3"

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

ORIGINAL="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler"

cd "${REPO}"

export PYTHONPATH="${REPO}/src"

export TRAINED_CHECKPOINT_PATH="${ORIGINAL}/experiments/checkpoints/paper_1lds_ppo_500tti.pt"

export NUM_EVAL_TTIS=2
export TOPOLOGY_SEED=42
export RUN_SEED=1234

export NUM_UT_PER_SECTOR=20
export NUM_USER_SLOTS=4

export FB_FRACTION=0.5
export FTP3_RATE_SCALE=1.0
export FTP3_PACKET_SIZE_SCALE=1.0

export CSI_DELAY_TTIS=0
export CSI_STALE_MODE="none"

export TD_RATE_LOG_NOISE_STD=0.0
export TD_RATE_DROPOUT_PROB=0.0
export TD_RATE_SHUFFLE=0

export CQI_BIAS=0.0
export CQI_NOISE_STD=0.0
export CQI_QUANT_STEP=0.0
export FLATTEN_SUBBAND_CQI=0

export RANK_FLIP_PROB=0.0
export RANK_FORCE=0

export PRECODER_UE_SHUFFLE=0
export PRECODER_RBG_SHUFFLE=0

export CANDIDATE_INTERVENTION_MODE="native"

export RUNTIME_SCENARIO_JSON=""

export POST_CSI_INTERFERENCE_POWER_SCALE=1.0
export POST_CSI_SERVING_POWER_SCALE=1.0
export POST_CSI_FAILED_BS=""


case "${SLURM_ARRAY_TASK_ID}" in

    0)
        export SCHEDULER_MODE="ppo"
        LABEL="ppo_clean"
        ;;

    1)
        export SCHEDULER_MODE="baseline"
        LABEL="baseline_clean"
        ;;

    2)
        export SCHEDULER_MODE="pf_greedy"
        LABEL="pf_greedy_clean"
        ;;

    3)
        export SCHEDULER_MODE="ppo"
        export POST_CSI_INTERFERENCE_POWER_SCALE=4.0
        LABEL="ppo_interference"
        ;;

    4)
        export SCHEDULER_MODE="baseline"
        export POST_CSI_INTERFERENCE_POWER_SCALE=4.0
        LABEL="baseline_interference"
        ;;

    5)
        export SCHEDULER_MODE="pf_greedy"
        export POST_CSI_INTERFERENCE_POWER_SCALE=4.0
        LABEL="pf_greedy_interference"
        ;;

    6)
        export SCHEDULER_MODE="ppo"
        export RUNTIME_SCENARIO_JSON='[{"start_tti":0,"name":"load","ftp_rate_scale":2.0}]'
        LABEL="ppo_dynamic"
        ;;

    7)
        export SCHEDULER_MODE="ppo"
        export CSI_DELAY_TTIS=1
        export CSI_STALE_MODE="all"
        LABEL="ppo_csi"
        ;;

    8)
        export SCHEDULER_MODE="ppo"
        export POST_CSI_FAILED_BS="0"
        LABEL="ppo_bs_failure"
        ;;

esac


export STRESS_TAG="wave3smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${LABEL}"

echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Scheduler: ${SCHEDULER_MODE}"
echo "Label: ${LABEL}"

"${PYTHON}" -u \
    scripts/run_robustness_eval.py

echo "WAVE3_CAPABILITY_SMOKE_PASS"
