#!/bin/bash
#SBATCH --job-name=wave5-full
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=./output2/wave5-full-%A_%a.out
#SBATCH --export=ALL

set -euo pipefail


cd "${SLURM_SUBMIT_DIR}"


mkdir -p output2

mkdir -p \
  experiments/stress_tests/parallel


source \
  "${HOME}/miniconda3/etc/profile.d/conda.sh"

conda activate oran_scheduler


unset PYTHONPATH

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"


MANIFEST="${SLURM_SUBMIT_DIR}/experiments/stress_tests/wave5_network_manifest.csv"


eval "$(
    python \
      scripts/read_wave5_network_manifest_row.py \
      "${MANIFEST}" \
      "${SLURM_ARRAY_TASK_ID}"
)"


# ============================================================
# CLEAN EVERY STRESS CONTROL FIRST
# ============================================================

unset PHYSICAL_NETWORK_SCENARIO_JSON || true

unset UE_AVAILABILITY_SCENARIO_JSON || true


export CSI_DELAY_TTIS=0

export CSI_STALE_MODE=none


export EXECUTION_DELAY_TTIS=0


export CONTROL_LOOP_ENABLE=0

export CONTROL_DEADLINE_MS=0.5

export CONTROL_BASE_COMPUTE_MS=0.0

export CONTROL_JITTER_STD_MS=0.0

export CONTROL_MISS_POLICY=empty


# ============================================================
# COMMON WAVE-5 CONFIG
# ============================================================

export TEMPORAL_RADIO_MODE="velocity_window"

export TEMPORAL_MAX_DISPLACEMENT_M="20.0"


export TRAINED_CHECKPOINT_PATH="${HOME}/oran-deep-scheduler/experiments/checkpoints/paper_1lds_ppo_500tti.pt"


#
# Packet diagnostics enabled for every condition.
#
export PACKET_QOS_ENABLE=1

#
# 4 TTIs = 2 ms.
#
export PACKET_QOS_DEADLINE_TTIS=4


# ============================================================
# CONDITION FAMILY
# ============================================================

case "${W5_FAMILY}" in


    clean)
        ;;


    csi_d2_all)

        export CSI_DELAY_TTIS=2

        export CSI_STALE_MODE=all
        ;;


    csi_d2_td_rate_only)

        export CSI_DELAY_TTIS=2

        export CSI_STALE_MODE=td_rate_only
        ;;


    csi_d2_ppo_radio)

        export CSI_DELAY_TTIS=2

        export CSI_STALE_MODE=ppo_radio
        ;;


    interference_x4)

        export PHYSICAL_NETWORK_SCENARIO_JSON='[
          {
            "start_tti": 0,
            "name": "normal"
          },
          {
            "start_tti": 2,
            "name": "interference",
            "non_serving_interference_power_scale": 4.0
          },
          {
            "start_tti": 6,
            "name": "recovery"
          }
        ]'
        ;;


    serving_power_0p25)

        export PHYSICAL_NETWORK_SCENARIO_JSON='[
          {
            "start_tti": 0,
            "name": "normal"
          },
          {
            "start_tti": 2,
            "name": "serving_degradation",
            "serving_signal_power_scale": 0.25
          },
          {
            "start_tti": 6,
            "name": "recovery"
          }
        ]'
        ;;


    availability_25pct)

        export UE_AVAILABILITY_SCENARIO_JSON='{
          "seed": 57211,
          "phases": [
            {
              "start_tti": 0,
              "name": "normal",
              "unavailable_fraction": 0.0
            },
            {
              "start_tti": 2,
              "name": "availability_loss",
              "unavailable_fraction": 0.25
            },
            {
              "start_tti": 6,
              "name": "recovery",
              "unavailable_fraction": 0.0
            }
          ]
        }'
        ;;


    execution_delay_1)

        export EXECUTION_DELAY_TTIS=1
        ;;


    execution_delay_2)

        export EXECUTION_DELAY_TTIS=2
        ;;


    deadline_empty)

        export CONTROL_LOOP_ENABLE=1

        #
        # Synthetic mixed hit/miss control timing.
        #
        # This is NOT measured inference latency.
        #
        export CONTROL_DEADLINE_MS=0.5

        export CONTROL_BASE_COMPUTE_MS=0.45

        export CONTROL_JITTER_STD_MS=0.10

        export CONTROL_MISS_POLICY=empty
        ;;


    deadline_repeat)

        export CONTROL_LOOP_ENABLE=1

        #
        # SAME latency process as deadline_empty.
        #
        # Only miss recovery policy differs.
        #
        export CONTROL_DEADLINE_MS=0.5

        export CONTROL_BASE_COMPUTE_MS=0.45

        export CONTROL_JITTER_STD_MS=0.10

        export CONTROL_MISS_POLICY=repeat_previous
        ;;


    *)
        echo \
          "Unknown Wave-5 family: ${W5_FAMILY}"

        exit 1
        ;;

esac


export STRESS_TAG="${WAVE5_RUN_ID}__${W5_CONDITION}__r${W5_REPLICATE}"


echo "============================================================"
echo "WAVE-5 NETWORK ROBUSTNESS"
echo "============================================================"

echo "Job:               ${SLURM_JOB_ID}"

echo "Task:              ${SLURM_ARRAY_TASK_ID}"

echo "Node:              $(hostname)"

echo "Run ID:            ${WAVE5_RUN_ID}"

echo "Condition:         ${W5_CONDITION}"

echo "Family:            ${W5_FAMILY}"

echo "Replicate:         ${W5_REPLICATE}"

echo "Speed:             ${UT_SPEED_KMH} km/h"

echo "TTIs:              ${NUM_EVAL_TTIS}"

echo "Temporal window:   ${TEMPORAL_WINDOW_TTIS}"

echo "TTI duration:      ${SIM_TTI_DURATION_MS} ms"

echo "CSI delay:         ${CSI_DELAY_TTIS}"

echo "CSI stale mode:    ${CSI_STALE_MODE}"

echo "Execution delay:   ${EXECUTION_DELAY_TTIS}"

echo "Control loop:      ${CONTROL_LOOP_ENABLE}"

echo "QoS deadline:      ${PACKET_QOS_DEADLINE_TTIS} TTIs"

echo "Topology seed:     ${TOPOLOGY_SEED}"

echo "Run seed:          ${RUN_SEED}"

echo "Stress tag:        ${STRESS_TAG}"

echo "============================================================"


python \
  scripts/run_robustness_eval.py


echo

echo "WAVE5_NETWORK_TASK_PASS"
