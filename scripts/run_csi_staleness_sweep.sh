#!/bin/bash
set -euo pipefail

OUTDIR="experiments/stress_tests/csi_staleness"

mkdir -p "${OUTDIR}"

for DELAY in 0 1 2 4 8 16
do
    echo
    echo "============================================================"
    echo "CSI STALENESS: delay=${DELAY} TTIs"
    echo "============================================================"

    CSI_DELAY_TTIS="${DELAY}" \
    NUM_EVAL_TTIS=100 \
    python -u \
        scripts/run_csi_staleness_eval.py \
        2>&1 | tee \
        "${OUTDIR}/delay_${DELAY}_100tti.log"
done

echo
echo "CSI STALENESS SWEEP COMPLETE"
