# Service-aware GPU integration and bottleneck benchmark

This **diagnostic**, opt-in benchmark consumes the verified Task 1–5
service-aware code. It does not alter the Nokia reproduction or train PPO.
Run only inside an allocated CHPC compute node; default device is `cuda:0`.

## What is exercised

1. A real Sionna UMa 21-sector network (1 ring, 3 sectors/site) on the GPU;
   generate the channel and compute association, rank diagnostics and SVD CSI.
2. Choose **one serving cell** from the network with the most associated UEs;
   up to four UEs are exposed as local packet queues with control and bulk
   flows. All other BSs are isotropic interferers through the existing PHY.
3. Every 0.5-ms logical TTI, inject reproducible traffic, observe pre-service
   packet/QoS state, and **choose a fresh allocation** for 18 RBGs based on
   backlog and already computed frequency-domain channel-quality scores.
4. Run the existing real final-allocation RZF/SINR evaluator in capture-only
   mode, Sionna whole-grant link adaptation, Sionna NR TBS, sampled TB decode,
   authoritative queue service and online exact byte-conservation checks.
5. Record QoS metrics, runtime and CUDA memory. A companion Slurm job records
   sampled GPU utilization from `nvidia-smi` independently of the hot path.

The queue-and-channel greedy policy is **a simple diagnostic workload**.
It is *not* the Nokia PF baseline, a validated QoS scheduler, or an optimized
MU-MIMO pairing algorithm. In particular this benchmark is **not** full
21-cell coordinated scheduling, nor does it model all 273 PRBs. We therefore
must **not** infer PPO sample-collection throughput or production DU latency
from these numbers. The current BWP uses the physically simulated 18 PRBs.

## Timing modes

`--refresh-every 0` (default): Generate channel/CSI ONCE, then benchmark the
steady allocation-to-packet path for multiple TTIs using the fixed snapshot.
This measures per-TTI overhead without charging channel generation every slot.
Bootstrap topology/channel/association/rank/CSI milliseconds are reported
separately as **one-time initialization**.

`--refresh-every 1`: Resample fading and redo rank/CSI for each slot after the
first, keeping initial serving association fixed. These are *independent fading
samples*, **not** time-correlated 3 km/h UE mobility. Cost is measured as an
additional workload in the unprofiled throughput number.

`--profile`: Forces `torch.cuda.synchronize` on stage boundaries to reveal
true GPU-inclusive stage latency. This **changes runtime**. Do not compare
its TTIs/s directly with uninstrumented throughput. During warm-up, stages
are not recorded. Uninstrumented mode has CUDA synchronizations only at
measurement-window boundaries; the end-to-end number is the useful one for
RL-environment budgeting.

Stages: packet arrivals/expiry; QoS observation; diagnostic policy; existing
MU-MIMO/SINR evaluation; whole-grant Sionna ILLA; NR TB sizing; post-TBS flow
packing; TB decode and queue service; conservation metrics. The results also
include live CUDA allocated memory samples and peak allocated/reserved memory.
The allocation-reserved difference is not itself a memory leak; look at live
allocated memory after successive slots for sustained growth.

## Run on a compute node

From the **login** node:

```bash
mkdir -p ~/oran-maintenance/benchmark_logs
sbatch --output="$HOME/oran-maintenance/benchmark_logs/qos-bench-%j.out" \
  ~/oran-deep-scheduler-unified/scripts/bench_service_aware_gpu.sbatch
```

Check `squeue --me`; after completion view the newest log:

```bash
tail -n 100 "$(ls -t ~/oran-maintenance/benchmark_logs/qos-bench-*.out | head -1)"
```

The log prints `BENCH_RESULTS` pointing to a directory containing four
self-describing `.json` reports and `gpu_utilization.csv`.
The job finishes by running the entire regression suite (expected **600 passed**
if the previous 592 tests remain unchanged); then it prints the campaign marker.
Success requires `SERVICE_AWARE_GPU_BENCH_CAMPAIGN_PASS`, not merely individual
step success. If any run fails, Slurm exits nonzero and does not print the
campaign pass marker. Do not commit/permanently change PHY source to fix a
performance bottleneck before inspecting the stage breakdown.

To run one experiment interactively in a GPU allocation:

```bash
conda activate oran_scheduler
cd ~/oran-deep-scheduler-unified
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -u scripts/benchmark_service_aware_runtime.py --device cuda:0 \
  --ttis 12 --warmup 2 --ut-per-sector 2 --profile
```

The Slurm job defaults to 80 GB of host RAM, one GPU, and 45 minutes maximum;
these are resource requests, not performance guarantees. Do not hold the GPU
idle between runs. If a GPU smaller than the workload's requirements is
assigned, report the GPU model/memory and traceback rather than silently
reducing simulated fidelity and treating the results as comparable.

## Interpreting results

- A dominant `initial_channel` or `initial_rank_csi` points to **PHY snapshot
  creation**, which should not be confused with packet-queue overhead.
- Dominant `final_mu_mimo_sinr` suggests examining repeated covariance/RZF,
  allocations and synchronization inside the existing PHY evaluator.
- Dominant `whole_grant_link_adaptation` suggests examining Sionna batching
  and adaptation object setup; the existing implementation does *not* reuse
  the PHYAbstraction/ILLA object across calls, which may be a hotspot.
- Dominant `packet_arrivals_and_expiry`, `qos_observation`, `flow_payload_packing`,
  or `tb_decode_and_queue_service` would support queue/service optimization.
- Persistent rising **live** CUDA bytes despite the same channel geometry
  suggests retained tensors. CUDA *reserved* bytes alone may represent normal
  allocator caching rather than a leak.
- `timed_ttis_per_second` is an **environment-only** rate for one scheduled
  cell, not PPO rollout or policy-update throughput. Estimate PPO training
  only after timing actual PPO observations/actions/updates as well.
- Report both `steady_throughput.json` and `channel_refresh.json`: one without
  expensive new-channel generation, one with it. A single total number without
  its channel mode is misleading.

The benchmark is designed to **identify** optimizations, not claim that all
previous simulator bottlenecks have already been eliminated.
