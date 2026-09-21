#!/bin/bash
#SBATCH --job-name=oran-recover
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=experiments/stress_tests/parallel/recovery-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --export=ALL

set -euo pipefail

REPO="/uufs/chpc.utah.edu/common/home/u1472438/oran-deep-scheduler"
PYTHON="/uufs/chpc.utah.edu/common/home/u1472438/miniconda3/envs/oran_scheduler/bin/python"

cd "${REPO}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        TAG="csi_ppo_radio_d1_recovery"
        MODE="ppo_radio"
        ;;
    1)
        TAG="csi_td_rate_only_d1_recovery"
        MODE="td_rate_only"
        ;;
esac

echo "TAG=${TAG}"
echo "HOST=$(hostname)"

CSI_DELAY_TTIS=1 \
CSI_STALE_MODE="${MODE}" \
FB_FRACTION=0.5 \
STRESS_TAG="${TAG}" \
NUM_EVAL_TTIS=100 \
"${PYTHON}" -u \
scripts/run_csi_staleness_eval.py

echo "COMPLETED: ${TAG}"
