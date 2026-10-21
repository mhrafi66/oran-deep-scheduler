#!/bin/bash
#SBATCH --job-name=oran-qos-perf
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --export=ALL
# Submit with --output=<absolute-existing-directory>/bench-%j.out

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate oran_scheduler
cd "$HOME/oran-deep-scheduler-unified"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
RESULT_DIR="$HOME/oran-maintenance/benchmark_results/${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"

echo "BENCH_JOB_ID=${SLURM_JOB_ID}"
echo "BENCH_NODE=$(hostname)"
echo "BENCH_COMMIT=$(git rev-parse HEAD)"
echo "BENCH_RESULTS=${RESULT_DIR}"
nvidia-smi

# Monitor actual GPU activity separately, without synchronizing the
# benchmarking process or leaving an allocation idle after the script exits.
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv -l 1 > "$RESULT_DIR/gpu_utilization.csv" 2>"$RESULT_DIR/gpu_monitor.err" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

# 1. Steady-state throughput: NO per-stage cuda.synchronize.
python -u scripts/benchmark_service_aware_runtime.py \
  --device cuda:0 --ttis 24 --warmup 4 --ut-per-sector 2 \
  --candidates 4 --user-slots 1 --refresh-every 0 \
  --json "$RESULT_DIR/steady_throughput.json"

# 2. Detailed profile: per-stage sync intentional; timing is diagnostic ONLY.
python -u scripts/benchmark_service_aware_runtime.py \
  --device cuda:0 --ttis 12 --warmup 2 --ut-per-sector 2 \
  --candidates 4 --user-slots 1 --refresh-every 0 --profile \
  --json "$RESULT_DIR/stage_profile.json"

# 3. Expensive full PHY refresh experiment; fixed association, independent
# channel draws, not a physically continuous mobility trajectory.
python -u scripts/benchmark_service_aware_runtime.py \
  --device cuda:0 --ttis 6 --warmup 1 --ut-per-sector 2 \
  --candidates 4 --user-slots 1 --refresh-every 1 \
  --json "$RESULT_DIR/channel_refresh.json"

# 4. Two simultaneous UE slots per RBG: diagnostic MU-MIMO pairing workload.
# This is intentionally NOT a validated optimal spatial pairing algorithm.
python -u scripts/benchmark_service_aware_runtime.py \
  --device cuda:0 --ttis 8 --warmup 2 --ut-per-sector 2 \
  --candidates 4 --user-slots 2 --refresh-every 0 \
  --json "$RESULT_DIR/mu_mimo_two_slot.json"

# Run the full regression after the GPU workload (the previously validated Task 5
# baseline had 592 tests; benchmark adds 8 CPU-compatible tests).
python -m pytest -q

echo "SERVICE_AWARE_GPU_BENCH_CAMPAIGN_PASS"
