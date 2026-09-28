#!/bin/bash
#SBATCH --job-name=oran-rob-v2
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/stress_tests/robustness_v2/slurm-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail

REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler"

PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

MANIFEST="${REPO}/experiments/stress_tests/robustness_v2/manifest.tsv"

OUTDIR="${REPO}/experiments/stress_tests/robustness_v2"

cd "${REPO}"


ROW_NUMBER=$(
    (
        SLURM_ARRAY_TASK_ID
        + 2
    )
)

ROW=$(
    sed -n \
    "${ROW_NUMBER}p" \
    "${MANIFEST}"
)


IFS=$'\t' read -r \
TAG \
RUN_SEED_VALUE \
TOPOLOGY_SEED_VALUE \
NUM_UT_VALUE \
USER_SLOTS_VALUE \
FB_VALUE \
FTP_RATE_VALUE \
FTP_SIZE_VALUE \
DELAY_VALUE \
STALE_MODE_VALUE \
CQI_BIAS_VALUE \
CQI_NOISE_VALUE \
CQI_QUANT_VALUE \
FLAT_CQI_VALUE \
RANK_FLIP_VALUE \
RANK_FORCE_VALUE \
PREC_UE_VALUE \
PREC_RBG_VALUE \
TD_NOISE_VALUE \
TD_DROPOUT_VALUE \
TD_SHUFFLE_VALUE \
<<< "${ROW}"


if [[ -z "${TAG}" ]]
then
    echo "Empty manifest row."
    exit 1
fi


echo "============================================================"
echo "TAG:              ${TAG}"
echo "JOB:              ${SLURM_JOB_ID}"
echo "ARRAY TASK:       ${SLURM_ARRAY_TASK_ID}"
echo "HOST:             $(hostname)"
echo "TOPOLOGY SEED:    ${TOPOLOGY_SEED_VALUE}"
echo "UE/SECTOR:        ${NUM_UT_VALUE}"
echo "USER SLOTS:       ${USER_SLOTS_VALUE}"
echo "FB FRACTION:      ${FB_VALUE}"
echo "CSI DELAY:        ${DELAY_VALUE}"
echo "CSI MODE:         ${STALE_MODE_VALUE}"
echo "============================================================"


#
# Absolutely critical:
#
# The CSV writers append across repeated runs.
# Remove files for THIS UNIQUE TAG before starting.
#
rm -f \
"${OUTDIR}/${TAG}.SUCCESS" \
"${OUTDIR}/${TAG}.FAIL" \
"${REPO}/experiments/stress_tests/parallel/${TAG}_metrics.csv" \
"${REPO}/experiments/stress_tests/parallel/${TAG}_kpi.csv" \
"/tmp/oran_${TAG}_snapshot.pt"


trap '
    touch "'"${OUTDIR}"'/'"${TAG}"'.FAIL"
' ERR


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
CORRUPTION_SEED=918273 \
STRESS_TAG="${TAG}" \
NUM_EVAL_TTIS=80 \
"${PYTHON}" -u \
scripts/run_robustness_eval.py


touch \
"${OUTDIR}/${TAG}.SUCCESS"

rm -f \
"${OUTDIR}/${TAG}.FAIL"


echo
echo "============================================================"
echo "ROBUSTNESS EXPERIMENT COMPLETE: ${TAG}"
echo "============================================================"
