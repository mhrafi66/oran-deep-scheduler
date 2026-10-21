# Task 7 — causal temporal channel / CSI-age integration

**Scope:** Opt-in service-aware downlink adapter. It does not alter Nokia/PPO,
channel generation, legacy PHY, HARQ, packet queues, or the Task-5 runner.

## Reused components

- `simulator.temporal_channel.VelocityWindowFrequencyChannelRuntime`:
  samples correlated Sionna channels at the desired TTI spacing.
- `simulator.temporal_channel.TemporalChannelWindow.tti_slice(t)`:
  passes a view of the current time sample, without a second channel call or
  per-TTI GPU-to-CPU copy. **Short-window/testing mode only**: `generate_window`
  materializes all frequency channels in a window, so do not allocate a huge
  paper-MIMO x 273-PRB x hundreds-of-UE window.
- The existing memory-safe `generate_tti_slice` or another provider can instead
  be passed as a `frame_source(t)`. Note that the current
  `generate_tti_slice` **regenerates CIR coefficients for the whole short
  window per requested offset**; it trades computation for memory. This Task
  does **not** claim to eliminate that cost or implement a persistent cached
  CIR generator across slots. Benchmark a production provider separately.
- `phy.compute_ideal_svd_csi` and rank diagnostic: calculate a CSI report
  **once per configured report period** on the sample corresponding to its
  timestamp. It is still an *ideal-SVD feedback surrogate*, not literal
  3GPP PMI/RI/CQI reporting.
- Task-5 runner, real PHY `evaluate_cell_allocation` and
  `FinalAllocationSINRCapture`: decode on the **current** channel, never on
  the delayed report's channel. The PHY may reuse delayed receive/precoder
  estimates, which should be documented in publications.

## Causal scheduling semantics

For TTI `t`, report period `p` and report delay `d`:

1. Obtain `H_t` from the provided Sionna temporal source **once**.
2. If `t % p == 0`, compute one CSI snapshot dated `t`, keeping it on the
   same GPU. The report includes the ordered global candidate UE IDs.
3. Publish only the latest report dated `s <= t-d`. On startup before any
   such report exists, publish **report 0** with `bootstrap=True`, explicitly
   exposing its actual age. This is a causal **startup convention**, NOT a
   faithful feedback protocol. For strict steady-state delay comparison,
   exclude bootstrap slots from statistics or have the scheduler choose no
   transmission during bootstrap.
4. The scheduler's callback receives only current *pre-service QoS* plus the
   published `CSIReport`, its timestamp and age; **never the current H_t**.
5. The PHY callback receives the exact same final scheduling decision,
   **H_t**, and the dated report (for delayed beamforming/combiner estimates).
6. The Task-5 TB builder, decoder, queues and metrics are unchanged.

Example: `delay=2`, `p=1` => report timestamps for t=0..5 are
`[0,0,0,1,2,3]`; actual PHY always uses `[H0,H1,H2,H3,H4,H5]`.

Explicit candidate-membership mismatch raises an error: never reuse stale
precoding vectors after a handover. The adapter is **per-cell, fixed candidate
membership**; reset/rebuild at handover. For long-running experiments a
separate identity/mobility bridge is required. Existing temporal Sionna
windows evolve Doppler under velocity, but do NOT guarantee geometric/channel
continuity across independently seeded windows.

## Efficiency and memory

- `frame_source` called once/TTI; no extra Sionna channel generation.
- `report_builder` called only once/report period. No conversion of CSI to CPU.
- `O(ceil(delay / period) + 2)` report references; obsolete GPU reports are
  released, and current physical frames are released after whole-grant link
  evaluation. Never store *every* `H_t` in the adapter.
- All source, report and physical callbacks run under `torch.no_grad()`.
- This is **not yet a measured throughput improvement**; assess on A100/H100.

## How to integrate

```python
adapter = ServiceAwareTemporalCSIAdapter(
    frame_source=TemporalWindowFrameSource(temporal_window),
    report_builder=compute_report_once_per_snapshot,  # (H_t, t) -> CSIReport
    policy=select_from_current_qos_and_dated_report,
    physical_link_provider=evaluate_actual_H_t_and_whole_grant,
    delay_ttis=2,
    report_period_ttis=1,
)
runner = ServiceAwareDownlinkCellRunner(
    queues=queues, observer=observer, decoder=decoder, grants=grants,
    policy=adapter.policy, link_provider=adapter.link_provider,
    flow_packer=pack_flows,
)
```

If `runner.step` fails, call `runner.reset(...)` **and** `adapter.reset()`
before episode replay. The adapter intentionally does not monkey-patch
runner.reset. Keep one adapter per runner and do not call it concurrently.

## Acceptance tests on CHPC's allocated GPU node

```bash
conda activate oran_scheduler
cd ~/oran-deep-scheduler-unified
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/simulator/test_service_aware_temporal_csi.py
python scripts/run_service_aware_temporal_csi_smoke.py
python scripts/run_service_aware_temporal_csi_smoke.py --real-sionna --device cuda:0
python scripts/run_service_aware_downlink_smoke.py --real-sionna --device cuda:0
python -m pytest -q
```

Expected markers `SERVICE_AWARE_TASK7_COMPONENT_PASS` and
`SERVICE_AWARE_TASK7_REAL_SIONNA_PASS`. **Do not commit if these fail.**

The real-Sionna smoke uses two UEs in a 21-sector interference geometry,
18 PRBs, **paper antenna arrays**, and two correlated channel samples. It
validates causal CSI/PHY binding, not full 210/420 UE 273 PRB realism.
HARQ's simplified one-process-per-UE diagnostic remains opt-in and is **not**
activated by this smoke. No claim about full NR HARQ or CSI feedback standards.
