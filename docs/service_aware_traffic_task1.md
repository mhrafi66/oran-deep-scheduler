# Task 1 — Authoritative Per-Flow Downlink Packet Queues

**Target base:** `research/unified-202610`, commit `4417409`; install into the existing `feature/service-aware-diagnostic` development branch.

## Scope and isolation

This is a new, **opt-in**, authoritative traffic-queue module. It does not change the existing `TrafficBufferManager`, `PacketQoSTrafficBufferManager`, Nokia PPO environment, channel simulation, or existing experiments. The old path remains the same and existing baseline results remain comparable. The module is not yet hooked into the main Sionna scheduler loop; this is intentional until the PHY-success / service interface is implemented in Task 2 and the observation interface in Task 4.

### Delivered in Task 1

- Multiple independently buffered QoS flows per UE, identified by `(local_ue_index, flow_id)`; no extra cross-cell state or global coordinator.
- Explicit 5QI-like ID, QoS priority, packet-delay budget (PDB) in TTIs, optional GBR and queue capacity. These are *input metadata*, **not** a 3GPP-compliant 5QI mapping or QoS decision algorithm.
- Actual per-packet arrivals and bytes, including externally supplied deterministic trace replay or a batched, seeded FTP3 Poisson source based on the repository's existing `FTP3TrafficConfig`.
- Packet FIFO, partial packet service, finite-buffer whole-packet rejection, and explicit deadline expiry policy (`drop` / `retain`).
- Read-only flow snapshots and O(1) cached per-UE and global queue bytes, plus counters for admitted/delivered/expired/overflow bytes and completed packets.
- Strict begin-then-serve TTI lifecycle, optional per-UE successful-byte cap preventing double allocation to multiple flows, reset and reproducible FTP3 sampling.
- Unit/invariant tests, large-queue stress test, and a CPU-light smoke scenario.

### Intentionally **not** delivered

- No automatic QoS priority decisions, flow selection, QoS TDS or RL state augmentation (Task 4 / baseline work).
- No SINR to TB decoding trial, HARQ, retransmission scheduling, or ACK/NACK (Task 2).
- No PHY RE overhead / TBS/RBG mapping corrections (Task 3).
- No production data-plane claim, real O-DU integration, full 3GPP protocol implementation, or gNB timing benchmark.
- **Do not run the legacy aggregate manager and this packet manager as simultaneous authorities for the same UE traffic.** That would double-account arrivals and service.

## Event and deadline conventions

Each `begin_tti(tti_index=t, arrivals=...)` first expires packets whose **inclusive deadline** was before `t`, then enqueues newly arrived packets at the start of `t`. For a flow with `pdb_ttis=D`, packets arriving at TTI `t` have last on-time scheduling opportunity `t + D - 1`. Completion in that final TTI is on time. With `expiry_policy="drop"`, only still-pending bytes are discarded when deadline is exceeded; previously delivered bytes are never undone. With `"retain"`, a miss is marked **once**, and a later completion is classified as late.

Whole-packet buffer admission is based on *remaining pending bytes*. Packets may be served across multiple TTIs. `FlowTTIService.delivered_bytes` counts physically **successful application payload** as supplied by the calling simulator. This class does not infer success from a predicted BLER: any numeric service grant passed to `apply_service` is accepted as a *successful* budget. Until Task 2, use deterministic replay or explicitly documented successful-byte fixtures, not the current Nokia target-compliant rate as a reliability oracle.

Payload units are **bytes** throughout (integer, no float drift). For existing per-UE *bit* features multiply the cached bytes by 8. Do not round PHY rate bits upward to packet payload bytes. A valid caller can floor positive successful payload bits to whole bytes after accounting for PHY/TB overhead and successful decoding. This conversion and link semantics are part of the later PHY integration.

## Minimal example

```python
from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival, QoSFlowSpec, ServiceAwareTrafficManager,
)

traffic = ServiceAwareTrafficManager(
    num_ues=1,
    tti_duration_s=0.0005,
    flows=(
        QoSFlowSpec(0, "control", five_qi=7, priority=20, pdb_ttis=3, gbr_bps=128_000),
        QoSFlowSpec(0, "download", five_qi=9, priority=90),
    ),
)

for tti, events, delivered in (
    (0, (FlowPacketArrival(0, "control", 100), FlowPacketArrival(0, "download", 300)), {(0, "control"): 20, (0, "download"): 80}),
    (1, (), {(0, "control"): 80}),
):
    start = traffic.begin_tti(tti_index=tti, arrivals=events)
    # The caller explicitly chooses service budgets per flow, and can pass
    # ue_success_capacity_bytes=[...] to reject capacity double counting.
    service = traffic.apply_service(service_bytes_by_flow=delivered)
    print(start.backlog_bytes, service.delivered_bytes, traffic.ue_backlog_bytes())
```

FTP3 sampling: provide `ftp3=FTP3TrafficConfig(...)` in each desired `QoSFlowSpec`, and call `begin_tti(tti_index=t)` **without** `arrivals` to generate arrivals from one batched Poisson draw. Passing `arrivals=()` disables sampling for that TTI and reuses externally provided/replayed trace data.

## Performance assumptions

- Per-flow `backlog_bytes`, packet count, and per-UE backlog are **cached** and updated by deltas. A backlog query does **not** sum pending packet sizes.
- FIFO and deadline structures are deques. Only packets that arrive, expire or complete are touched, so routine empty TTIs do not iterate all pending packets. Flow iteration is O(number of configured flows), appropriate for local per-cell queue maintenance.
- Large-scale simulations should avoid calling `flow_snapshot` for every flow unless its data are required; a snapshot copies immutable counters but does not traverse packets.
- Auto-generated FTP3 sampling is batched once per cell TTI on a seeded CPU generator; if a simulator already has an arrival trace, pass it explicitly to avoid resampling.
- Queue management is implemented in Python for correctness/prototyping. **No throughput or 0.5-ms compute budget is claimed** until profiling at the intended scale on CHPC and in a real scheduler loop.

## Testing on CHPC

On an **allocated CHPC compute node** with the existing `oran_scheduler` conda environment (not the `notchpeak2` login node):

```bash
conda activate oran_scheduler
cd ~/oran-deep-scheduler-unified
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python -m pytest -q tests/simulator/test_service_aware_traffic.py
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python scripts/run_service_aware_queue_smoke.py
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python -m pytest -q
```

Expected new-suite result on the packaged source: **15 passed**. The smoke ends with `SERVICE_AWARE_TASK1_SMOKE_PASS`. Baseline was **431 passed** before the patch; expect **446 tests** if no additional tests have been added and the existing environment remains compatible. Do not claim 444 passed until CHPC actually runs it.

## Acceptance gate (before Task 2)

1. 13 Task-1 tests pass on CHPC, including per-flow multi-UE service and randomized conservation.
2. Smoke ends with the PASS marker.
3. Existing 431 regression tests still pass on an allocated compute node.
4. Git diff contains only the four new files in this Task-1 bundle.
5. No paper benchmark or learned scheduling experiment is changed by this patch.

## Known boundaries / later integration

- Only *local UE indices* are supported now; a global-ID handover bridge needs to be added after the new data path is integrated into handover experiments.
- 5QI and GBR metadata are stored, not yet enforced; no policy or reward uses them.
- QoS packet expiry is defined at the simulation TTI level, not a millisecond-accurate RLC timer.
- Existing Sionna PHY capacity is not translated to per-flow confirmed decoding results here; this is deliberately Task 2.
- Manual flow grants are for correctness fixtures and future scheduling interfaces, not proof of an optimized per-flow scheduling policy.
