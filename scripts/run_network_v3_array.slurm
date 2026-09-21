#!/bin/bash
#SBATCH --job-name=oran-net-v3
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/stress_tests/network_v3/slurm-%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL

set -euo pipefail

REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler"
PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

cd "${REPO}"

RUN_ID="${RUN_ID:-$(cat experiments/stress_tests/network_v3/latest_run_id.txt)}"
RUN_DIR="${REPO}/experiments/stress_tests/network_v3/${RUN_ID}"
MANIFEST="${RUN_DIR}/manifest.tsv"
MARKERS="${RUN_DIR}/markers"
mkdir -p "${MARKERS}"

ROW_NUMBER=$((SLURM_ARRAY_TASK_ID + 2))
ROW=$(sed -n "${ROW_NUMBER}p" "${MANIFEST}")

if [[ -z "${ROW}" ]]; then
    echo "Missing manifest row for task ${SLURM_ARRAY_TASK_ID}."
    exit 1
fi

IFS=$'\t' read -r \
TAG FAMILY CONDITION REPLICATE \
RUN_SEED_VALUE TOPOLOGY_SEED_VALUE CORRUPTION_SEED_VALUE \
NUM_UT_VALUE USER_SLOTS_VALUE FB_VALUE \
FTP_RATE_VALUE FTP_SIZE_VALUE DELAY_VALUE STALE_MODE_VALUE \
CQI_BIAS_VALUE CQI_NOISE_VALUE CQI_QUANT_VALUE FLAT_CQI_VALUE \
RANK_FLIP_VALUE RANK_FORCE_VALUE PREC_UE_VALUE PREC_RBG_VALUE \
TD_NOISE_VALUE TD_DROPOUT_VALUE TD_SHUFFLE_VALUE \
<<< "${ROW}"

OUT_METRICS="${REPO}/experiments/stress_tests/parallel/${TAG}_metrics.csv"
OUT_KPI="${REPO}/experiments/stress_tests/parallel/${TAG}_kpi.csv"

printf '%s\n' \
"============================================================" \
"NETWORK V3 ROBUSTNESS EVALUATION" \
"RUN ID:          ${RUN_ID}" \
"TAG:             ${TAG}" \
"FAMILY:          ${FAMILY}" \
"CONDITION:       ${CONDITION}" \
"REPLICATE:       ${REPLICATE}" \
"ARRAY TASK:      ${SLURM_ARRAY_TASK_ID}" \
"HOST:            $(hostname)" \
"TOPOLOGY SEED:   ${TOPOLOGY_SEED_VALUE}" \
"RUN SEED:        ${RUN_SEED_VALUE}" \
"UE/SECTOR:       ${NUM_UT_VALUE}" \
"USER SLOTS:      ${USER_SLOTS_VALUE}" \
"FB FRACTION:     ${FB_VALUE}" \
"FTP RATE SCALE:  ${FTP_RATE_VALUE}" \
"FTP SIZE SCALE:  ${FTP_SIZE_VALUE}" \
"CSI DELAY:       ${DELAY_VALUE}" \
"CSI MODE:        ${STALE_MODE_VALUE}" \
"============================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Never append onto a previous attempt for the same unique tag.
rm -f \
    "${OUT_METRICS}" \
    "${OUT_KPI}" \
    "/tmp/oran_${TAG}_snapshot.pt" \
    "${MARKERS}/${TAG}.SUCCESS" \
    "${MARKERS}/${TAG}.FAIL"

trap 'touch "'"${MARKERS}"'/'"${TAG}"'.FAIL"' ERR

RUN_SEED="${RUN_SEED_VALUE}" \
TOPOLOGY_SEED="${TOPOLOGY_SEED_VALUE}" \
NUM_UT_PER_SECTOR="${NUM_UT_VALUE}" \
NUM_USER_SLOTS="${USER_SLOTS_VALUE}" \
FB_FRACTION="${FB_VALUE}" \
FTP3_RATE_SCALE="${FTP_RATE_VALUE}" \
FTP3_PACKET_SIZE_SCALE="${FTP_SIZE_VALUE}" \
CSI_DELAY_TTIS="${DELAY_VALUE}" \
CSI_STALE_MODE="${STALE_MODE_VALUE}" \
CQI_BIAS="${CQI_BIAS_VALUE}" \
CQI_NOISE_STD="${CQI_NOISE_VALUE}" \
CQI_QUANT_STEP="${CQI_QUANT_VALUE}" \
FLATTEN_SUBBAND_CQI="${FLAT_CQI_VALUE}" \
RANK_FLIP_PROB="${RANK_FLIP_VALUE}" \
RANK_FORCE="${RANK_FORCE_VALUE}" \
PRECODER_UE_SHUFFLE="${PREC_UE_VALUE}" \
PRECODER_RBG_SHUFFLE="${PREC_RBG_VALUE}" \
TD_RATE_LOG_NOISE_STD="${TD_NOISE_VALUE}" \
TD_RATE_DROPOUT_PROB="${TD_DROPOUT_VALUE}" \
TD_RATE_SHUFFLE="${TD_SHUFFLE_VALUE}" \
CORRUPTION_SEED="${CORRUPTION_SEED_VALUE}" \
STRESS_TAG="${TAG}" \
NUM_EVAL_TTIS=100 \
"${PYTHON}" -u scripts/run_robustness_eval.py

touch "${MARKERS}/${TAG}.SUCCESS"
rm -f "${MARKERS}/${TAG}.FAIL"

echo "NETWORK_V3_EXPERIMENT_COMPLETED ${TAG}"
