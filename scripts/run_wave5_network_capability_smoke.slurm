#!/bin/bash
#SBATCH --job-name=wave5-net
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --array=0-4%5
#SBATCH --output=./output2/wave5-net-%A_%a.out
#SBATCH --export=ALL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

mkdir -p output2
mkdir -p experiments/stress_tests/parallel

source "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler

unset PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"


# ============================================================
# CLEAN ENVIRONMENT
# ============================================================

unset PHYSICAL_NETWORK_SCENARIO_JSON || true
unset UE_AVAILABILITY_SCENARIO_JSON || true

export EXECUTION_DELAY_TTIS=0

export CONTROL_LOOP_ENABLE=0
export CONTROL_DEADLINE_MS=0.5
export CONTROL_BASE_COMPUTE_MS=0.0
export CONTROL_JITTER_STD_MS=0.0
export CONTROL_MISS_POLICY=empty

export CSI_DELAY_TTIS=0
export CSI_STALE_MODE=none


# ============================================================
# COMMON RADIO / EVALUATION CONFIG
# ============================================================

export NUM_EVAL_TTIS=4

export TEMPORAL_RADIO_MODE="velocity_window"

export TEMPORAL_WINDOW_TTIS=8

#
# Wave-5 correction:
# scheduler slot / traffic TTI = 0.5 ms.
#
export SIM_TTI_DURATION_MS="0.5"

export UT_SPEED_KMH="30"

export TEMPORAL_MAX_DISPLACEMENT_M="20.0"

export TRAINED_CHECKPOINT_PATH="${HOME}/oran-deep-scheduler/experiments/checkpoints/paper_1lds_ppo_500tti.pt"


# ============================================================
# PACKET-LEVEL QoS SHADOW TRACKER
# ============================================================

export PACKET_QOS_ENABLE=1

#
# 4 TTIs = 2 ms at 0.5 ms / TTI.
#
export PACKET_QOS_DEADLINE_TTIS=4


# ============================================================
# CAPABILITY-SPECIFIC STRESS
# ============================================================

case "${SLURM_ARRAY_TASK_ID}" in

    0)
        LABEL="clean_qos"
        ;;


    1)
        LABEL="persistent_interference"

        export PHYSICAL_NETWORK_SCENARIO_JSON='[
          {
            "start_tti": 0,
            "name": "normal"
          },
          {
            "start_tti": 1,
            "name": "interference",
            "non_serving_interference_power_scale": 4.0
          },
          {
            "start_tti": 3,
            "name": "recovery"
          }
        ]'
        ;;


    2)
        LABEL="ue_availability"

        export UE_AVAILABILITY_SCENARIO_JSON='{
          "seed": 12345,
          "phases": [
            {
              "start_tti": 0,
              "name": "normal",
              "unavailable_fraction": 0.0
            },
            {
              "start_tti": 1,
              "name": "outage",
              "unavailable_fraction": 0.25
            },
            {
              "start_tti": 3,
              "name": "recovery",
              "unavailable_fraction": 0.0
            }
          ]
        }'
        ;;


    3)
        LABEL="identity_safe_delay"

        export EXECUTION_DELAY_TTIS=1
        ;;


    4)
        LABEL="control_deadline"

        export CONTROL_LOOP_ENABLE=1

        export CONTROL_DEADLINE_MS=0.5

        export CONTROL_BASE_COMPUTE_MS=0.8

        export CONTROL_JITTER_STD_MS=0.0

        export CONTROL_MISS_POLICY=empty
        ;;


    *)
        echo "Unknown Wave-5 smoke task."
        exit 1
        ;;
esac


export STRESS_TAG="wave5smoke_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${LABEL}"


echo "============================================================"
echo "WAVE-5 NETWORK CAPABILITY SMOKE"
echo "============================================================"
echo "Job:             ${SLURM_JOB_ID}"
echo "Array job:       ${SLURM_ARRAY_JOB_ID}"
echo "Task:            ${SLURM_ARRAY_TASK_ID}"
echo "Label:           ${LABEL}"
echo "Node:            $(hostname)"
echo "Temporal mode:   ${TEMPORAL_RADIO_MODE}"
echo "Window:          ${TEMPORAL_WINDOW_TTIS}"
echo "TTI duration:    ${SIM_TTI_DURATION_MS} ms"
echo "Speed:           ${UT_SPEED_KMH} km/h"
echo "Packet QoS:      ${PACKET_QOS_ENABLE}"
echo "Stress tag:      ${STRESS_TAG}"
echo "============================================================"

nvidia-smi


python \
  scripts/run_robustness_eval.py


echo
echo "WAVE5_NETWORK_CAPABILITY_PASS"
