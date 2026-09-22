#!/bin/bash
#SBATCH --job-name=wave3-net
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-233%64
#SBATCH --output=./output2/wave3-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail


REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler-wave3"

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

ORIGINAL_REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler"


cd "${REPO}"

export PYTHONPATH="${REPO}/src"


if [[ -z "${RUN_ID:-}" ]]; then
    echo "ERROR: RUN_ID missing."
    exit 1
fi


ROW_JSON=$(
    "${PYTHON}" \
        scripts/read_wave3_manifest_row.py \
        "${SLURM_ARRAY_TASK_ID}"
)


eval "$(
    ROW_JSON="${ROW_JSON}" \
    "${PYTHON}" - <<'PY'
import json
import os
import shlex


row = json.loads(
    os.environ[
        "ROW_JSON"
    ]
)


mapping = {
    "scheduler_mode": "SCHEDULER_MODE",
    "topology_seed": "TOPOLOGY_SEED",
    "run_seed": "RUN_SEED",
    "num_eval_ttis": "NUM_EVAL_TTIS",
    "num_ut_per_sector": "NUM_UT_PER_SECTOR",
    "num_user_slots": "NUM_USER_SLOTS",
    "csi_delay_ttis": "CSI_DELAY_TTIS",
    "csi_stale_mode": "CSI_STALE_MODE",
    "td_rate_log_noise_std": "TD_RATE_LOG_NOISE_STD",
    "cqi_noise_std": "CQI_NOISE_STD",
    "ftp_rate_scale": "FTP3_RATE_SCALE",
    "runtime_scenario_json": "RUNTIME_SCENARIO_JSON",
    "post_csi_interference_power_scale": (
        "POST_CSI_INTERFERENCE_POWER_SCALE"
    ),
    "post_csi_serving_power_scale": (
        "POST_CSI_SERVING_POWER_SCALE"
    ),
    "post_csi_failed_bs": "POST_CSI_FAILED_BS",
}


for key, env_name in (
    mapping.items()
):

    print(
        "export "
        f"{env_name}="
        f"{shlex.quote(str(row[key]))}"
    )


condition = row[
    "condition"
]

scheduler = row[
    "scheduler_mode"
]

replicate = row[
    "replicate"
]


tag = (
    f"{os.environ['RUN_ID']}"
    f"__{condition}"
    f"__{scheduler}"
    f"__r{replicate}"
)


print(
    "export STRESS_TAG="
    f"{shlex.quote(tag)}"
)
PY
)"


export TRAINED_CHECKPOINT_PATH="${ORIGINAL_REPO}/experiments/checkpoints/paper_1lds_ppo_500tti.pt"

# Optional infrastructure-smoke override.
if [[ -n "${WAVE3_NUM_TTIS_OVERRIDE:-}" ]]; then
    export NUM_EVAL_TTIS="${WAVE3_NUM_TTIS_OVERRIDE}"
fi


#
# Common defaults.
#
export FB_FRACTION=0.5
export NUM_UT_PER_SECTOR=20
export NUM_USER_SLOTS=4

export FTP3_PACKET_SIZE_SCALE=1.0

export CQI_BIAS=0.0
export CQI_QUANT_STEP=0.0
export FLATTEN_SUBBAND_CQI=0

export RANK_FLIP_PROB=0.0
export RANK_FORCE=0

export PRECODER_UE_SHUFFLE=0
export PRECODER_RBG_SHUFFLE=0

export TD_RATE_DROPOUT_PROB=0.0
export TD_RATE_SHUFFLE=0

export CORRUPTION_SEED="$((918273 + SLURM_ARRAY_TASK_ID))"

export CANDIDATE_INTERVENTION_MODE="native"


echo "============================================================"
echo "WAVE-3 MANIFEST RUN"
echo "============================================================"
echo "Run ID:             ${RUN_ID}"
echo "Task:               ${SLURM_ARRAY_TASK_ID}"
echo "Node:               $(hostname)"
echo "Scheduler:          ${SCHEDULER_MODE}"
echo "Topology seed:      ${TOPOLOGY_SEED}"
echo "Run seed:           ${RUN_SEED}"
echo "TTIs:               ${NUM_EVAL_TTIS}"
echo "Tag:                ${STRESS_TAG}"
echo "============================================================"

echo

nvidia-smi \
  --query-gpu=name,memory.total \
  --format=csv,noheader

echo


"${PYTHON}" -u \
    scripts/run_robustness_eval.py


echo
echo "WAVE3_TASK_COMPLETED"
echo "${STRESS_TAG}"
