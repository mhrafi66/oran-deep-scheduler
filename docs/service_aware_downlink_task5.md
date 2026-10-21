# Task 5 — integrated, opt-in service-aware downlink

**Target repository:** `~/oran-deep-scheduler-unified`, `feature/service-aware-diagnostic`, HEAD `5c78727` (Tasks 1–4 committed). This task changes only two pre-existing files, plus five new files; the original Nokia queue, learned policy, traffic reward and default physical evaluator remain unchanged when no optional PHY capture callback is supplied.

## Scope and exact execution order

The new `ServiceAwareDownlinkCellRunner` is a **per-cell** system-level simulation controller. Each TTI:

1. Calls Task 1 `ServiceAwareTrafficManager.begin_tti` (packet arrivals, queue overflow, deadline expiry).
2. Calls Task 4 `ServiceAwareQoSObserver.observe()` once **before** any current-slot transmission decisions. QoS observations contain only the cell's configured UEs and flows.
3. Calls an externally supplied `policy(observation)` for the actual `[user_slot, RBG]` `CellAllocation` and candidate-to-local-UE mapping. No new classical or RL policy is implemented.
4. Calls an externally supplied `link_provider(observation, decision)` to obtain **whole-grant** `WholeGrantLink` estimates for *exactly* those scheduled candidate/RBG sets. It must come from a compatible physical link abstraction, not a product/mean of per-RBG BLERs.
5. Calls Task 3 `NRGrantAssembler.build()` to size **one complete TB per scheduled UE** with actual PRB geometry and configured PDSCH RE overhead.
6. Runs an externally supplied `flow_packer(obs, decision, sized_tbs)` if the policy has not already specified `planned_flow_bytes`. This makes service multiplexing obey the *actual* sized TB capacity. There is **no implicit QoS flow selection policy**.
7. Task 2 `ServiceAwareTransmissionEngine.execute_tti()` samples each TB decoding outcome once; unsuccessful TBs deliver 0 bytes; Task 1 consumes only successful, explicitly packed flow bytes.
8. Streaming Task 5 metrics assert TTI-level and cumulative packet-byte conservation. `runner.report()` computes final per-flow metrics and packet delay percentiles.

The sequence is strict: no packet decision sees the current slot's later decoding result. The queue owns all packet state; the runner does not maintain second copies. `runner.reset()` explicitly discards an unfinished TTI **only to abandon/replay an entire episode** after a bad policy/provider call. The standalone Task 1 `queues.reset()` remains strict by default: it refuses to reset inside an active TTI. Under exceptions, do **not** continue to the next slot; fix the error, `runner.reset()`, replay from TTI 0.

## Reuse of actual Sionna PHY without re-evaluating channels

`FinalAllocationSINRCapture` is an optional **passive** hook on the *existing* `evaluate_cell_allocation` evaluator. It observes the final scheduled RBG's *already calculated* post-MU-MIMO layer SINRs, layer-to-UE mapping and ranks; it does **not** regenerate the Sionna channel, compute RZF a second time, or reproduce interference calculations. The normal evaluator with `capture_post_precoder_sinr=None` and `capture_only=False` is unchanged. For Task 5 use **`capture_only=True`** with a callback, which avoids redundant per-RBG Sionna ILLA/PHYAbstraction and returns `None` (no legacy per-RBG rate result). This is a separate, explicit API contract: never interpret the capture-only call as a legacy rate evaluation.

For a final UE grant spanning multiple RBGs, the capture gathers only those actually assigned subcarrier SINRs, preserving per-UE rank and stream order, concatenates them across frequency and passes the whole concatenated grant to the same Sionna `InnerLoopLinkAdaptation` and `PHYAbstraction` already used by the existing evaluator. Equal-shaped UE grants are batched; configuration-specific link-adaptation objects are reused. The capture-only mode runs the final evaluator's costly MU-MIMO SINR/precoder computation exactly once per scheduled RBG, **skips redundant per-RBG link adaptation**, and runs whole-grant link adaptation once per group of compatible grants. This supplies *one MCS and one Sionna-estimated TBLER* for each complete UE TB and permits Task 3's whole-grant sizing. Missing/stale RBG samples, inconsistent rank, duplicate streams, invalid samples and stale allocation identities are rejected.

**Scientific limitations:** Sionna's PHY abstraction is a system-level link estimate, **not** bit-accurate 5G NR code-block CRC, HARQ, measured over-the-air BLER, calibrated packet reliability across every payload size, or full DMRS/PDCCH/PUCCH simulation. When a generated channel has only one OFDM symbol, its SINR is expanded across the configured slot, matching the existing research model's temporal assumption. PDSCH data REs are budgeted for whole-TB *size* separately; the SINR-based link abstraction uses the sampled assigned frequency resources rather than simulating an explicit DMRS symbol/RE mask. **Do not claim exact NR BLER / actual real-network packet delivery on this basis.**

The legacy `evaluate_cell_allocation` currently assumes equal subcarrier width for each RBG, while `NRBandwidthPart` itself supports partial first/last RBGs. Therefore the real Sionna integration smoke runs with **18 PRBs = 18 RBGs**, one PRB per RBG. It does **not** claim to physically simulate a full 273-PRB carrier with the original reduced channel. The actual PHY channel PRBs must equal the grant assembler BWP width.

## Metrics definitions and boundaries

* **Arrived / delivered / overflow-dropped / deadline-expired / pending payload bytes:** exact integer conservation for each flow and the entire cell after every completed TTI.
* **Scheduled TB capacity vs decoded capacity vs successful flow bytes:** separate metrics, so neither padding nor failed TBs count as useful delivered payload. No HARQ process is modeled; failed bytes remain queued for a future independently selected attempt.
* **On-time completion ratio:** successfully *completed* on-time packets divided by *all arrivals*, including arrivals rejected by full buffers. If no arrivals, report `None` rather than an artificial 100%. Packets still pending at measurement end are **not** silently counted as successful or missed.
* **Deadline misses:** Task 1 counts each missed packet upon entering the next TTI after its last eligible slot; any finite-horizon pending packets remain explicitly pending. Expiry policy chooses drop vs late retention; no hidden assumptions.
* **Completed-packet delay p50/p95/p99:** exact nearest-rank percentile over *completed packets*, measured as TTI count **including** the arrival TTI. Task 1 updates a compact exact histogram once per completed packet; no list of samples or scan of waiting packets per slot. Incomplete or dropped packets are excluded; interpret such delay percentiles alongside completion and drop ratios (survivorship bias).
* **Throughput:** successful *flow payload bytes* × 8 / elapsed simulated seconds, per cell and per UE.
* **Jain's fairness:** successful throughput across UEs with nonzero offered arrivals during measurement, and `None` if all offered UEs received zero service. Do not interpret as slice-level fairness.
* **GBR measure:** delivered flow bytes divided by `gbr_bps × (slots having pre-service backlog) × TTI seconds / 8`. This is explicitly a **demand-gated diagnostic** (may exceed 1), not formal 3GPP GBR SLA compliance.

Task 5 streaming metrics store per-flow GBR backlogged-slot counters and scalar global totals; final reporting samples Task 1's authoritative flow counters/histograms. No per-TTI trace is saved by default. Performance characteristics still need measurement on CHPC with actual 21-cell Sionna execution.

## Files

- **Modified:** `src/oran_scheduler/simulator/service_aware_traffic.py` — exact completed-packet delay histogram and explicitly guarded active-TTI reset.
- **Modified:** `src/oran_scheduler/phy/schedule_evaluator.py` — optional passive capture callback for already-computed post-MU-MIMO SINR (no effect for legacy calls).
- **New:** `src/oran_scheduler/simulator/service_aware_downlink.py` — TTI controller + conservation metrics and final summaries.
- **New:** `src/oran_scheduler/simulator/service_aware_whole_grant_phy.py` — optional whole-grant Sionna MCS/TBLER from captured SINRs.
- **New:** `tests/simulator/test_service_aware_downlink.py` — QoS, conservation, reset, deadline, multi-flow, flow packer, and PHY capture tests.
- **New:** `scripts/run_service_aware_downlink_smoke.py` — deterministic entire-chain smoke; `--real-sionna` executes actual Sionna channel → existing final allocation PHY → whole-grant Sionna link → decoded packet delivery.
- **New:** `docs/service_aware_downlink_task5.md` — this document.

## CHPC validation

On an **allocated compute node** (not `notchpeak2`), from the worktree with `oran_scheduler` active:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/simulator/test_service_aware_downlink.py
python scripts/run_service_aware_downlink_smoke.py
python scripts/run_service_aware_downlink_smoke.py --real-sionna --device cuda:0
python -m pytest -q
```

Expected markers:

- `SERVICE_AWARE_TASK5_COMPONENT_PASS` — synthetic link for queue-integrated deterministic assertions, **not** realistic PHY.
- `SERVICE_AWARE_TASK5_REAL_SIONNA_PASS` — actual Sionna channel/PHY callback and TB-wide Sionna link abstraction; **must run on CHPC**. Do not call the end-to-end physical integration validated until this command succeeds there.

If any real-Sionna shape/device/MCS mismatch emerges, **do not commit**; preserve the exact traceback and correct the interface rather than hiding it behind synthetic link values. Only run a commit/push after the real-Sionna smoke and full 565+ regression pass.

## Not in scope

New QoS scheduler policy, PPO changes, HARQ buffers/processes, PDCCH scheduling, perfect BLER calibration, all 273 full-band PHY PRBs with variable RBG sizes, packet-flow mobility state migration into this opt-in runtime, and 21-cell real-time performance claims. Existing Nokia reproduction remains independently testable and is not switched to the new packet semantics.
