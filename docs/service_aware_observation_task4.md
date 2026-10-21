# Task 4 — per-cell pre-service QoS observations (opt-in)

Task 4 adds `ServiceAwareQoSObserver` without changing the legacy Nokia baseline scheduler, Sionna PHY, Task 2 decoder, Task 3 grant sizing, or Task 1's packet-queue semantics. It adds only two narrow public **read-only accessors** to Task 1's traffic manager: `active_tti_index` and `configured_flows()`.

## What a future scheduler can observe

`QoSCellObservation` is a frozen, immutable snapshot ordered by configured flow then by local UE index. For **each flow** it exposes queued bytes and packets, HOL delay in TTIs, signed HOL deadline slack, 5QI, priority, optional GBR rate and a bounded shortfall proxy. For **each UE** it exposes aggregate cached backlog, number of nonempty flows, most urgent head deadline, maximum HOL delay, highest configured priority (smaller means higher priority), and sum of actionable GBR shortfalls. UEs with no traffic still appear as zero-backlog observations. No other-cell state or future PHY outcome appears in the observation.

`deadline_slack_ttis = pdb_ttis - hol_delay_ttis` when the head of a nonempty flow has a configured deadline. `0` means the **current** slot is the last eligible service slot, `2` means two slots **after** the current slot remain, `None` means no deadline-bearing head. For `expiry_policy='retain'`, a missed retained HOL is indicated by negative slack. Dropped deadline-expired packets do not appear in backlog. The metric is for the HOL packet, not a fabricated per-flow average of packet deadlines. This formula is correct because Task 1 assigns a fixed PDB per flow and its packets are FIFO ordered.

5QI and priority are **configured metadata**, not an inferred 3GPP-specific 5QI table or scheduler. This module does not calculate PRB allocations, rewards, queue-grant decisions, or priority weights.

## Required lifecycle (no future leakage)

Instantiate the observer once **before TTI 0**, after constructing the per-cell `ServiceAwareTrafficManager`. Every TTI must follow:

```python
queues.begin_tti(tti_index=t, arrivals=arrivals)
pre_service_observation = observer.observe()  # exactly once, BEFORE selecting grants
# schedule using pre_service_observation and separately available radio features
result = decoder.execute_tti(queues=queues, transport_blocks=tbs,
                             planned_flow_bytes=planned_flows)
```

Task 1 does not allow starting TTI `t+1` until `t` is served. `observe()` further rejects duplicate or skipped observations and cannot run after the current TTI's service. Its GBR history only incorporates past, completed successful service from Task 1's **cumulative delivered-byte counters**, not scheduled capacity or failed TB attempts. No extra service-history source is maintained.

To replay an episode, first call `queues.reset()` (and `decoder.reset()`), **then** `observer.reset()` before starting TTI 0. A missed observer reset is detected via the expected TTI sequence.

## Bounded GBR pressure proxy — not a 3GPP compliance oracle

Set `gbr_window_ttis` to the trailing-window width (default 100 slots, i.e. 50 ms at 30 kHz with 0.5 ms slot duration). The window includes the current decision slot and up to `window-1` **previous completed** slots.

- Only a TTI that had queue backlog **after arrivals/expiry and before service** earns a GBR target. Idle slots earn no target, avoiding fictitious deficit after long idle periods.
- `gbr_target_bytes_in_window = gbr_bps * tti_duration_s / 8 * backlogged_slots_in_window`.
- `gbr_window_delivered_bytes` is the **successful** flow payload delivered in the completed slots of the window. This is derived by differencing Task 1's cumulative delivered counters once per TTI.
- `gbr_deficit_bytes = min(current_backlog_bytes, ceil(max(0, target - delivered)))`, with a tiny floating-point roundoff guard.
- GBR flow state stores at most `window-1` `(was_backlogged, delivered_bytes)` records; updates are amortized O(1)/GBR flow. Non-GBR flows require no history.

This deficit is a **local scheduling-pressure feature**, not a claim that a 5QI GBR SLA was violated. It is demand-gated, backlog-capped, and excludes the in-progress slot's eventual delivered bytes. It does not estimate long-term delivered throughput for a flow that has no backlog. The window and target convention are experimental choices and must be held constant across algorithms.

## Cost and layering

`observe()` calls `flow_snapshot` **once per configured flow** and retrieves Task 1's cached UE backlog **once per cell**. No packet-list traversal, PHY re-evaluation, CSI sampling, GPU copies, or cached copy of queued payloads. Time is O(configured flows + local UEs) per TTI. Storage is O(GBR flows × window) with only compact successful-byte/eligibility history, not packet arrays.

The observer's frozen observations can be consumed by classical or learned scheduling implementations independently of how the radio features are provided. **It does not yet connect these features to Nokia's legacy PPO state**, which is intentionally unchanged.

## Tests and acceptance

On a CHPC **compute** node, from `~/oran-deep-scheduler-unified` with `oran_scheduler` active and `PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"`:

```bash
python -m pytest -q tests/simulator/test_service_aware_observation.py
python scripts/run_service_aware_observation_smoke.py
python -m pytest -q
```

Tests cover flow/UE aggregation, end-to-end successful-vs-failed TB service, deadline expiry/retention, finite queues, partial packets, immutability, stale/skipped snapshots, resets, independent cells, sliding GBR windows, idle no-debt behavior and one-snapshot-per-flow rather than per-packet work.

## Unresolved after Task 4

1. This remains an **opt-in** observation layer, not an integrated Nokia scheduling policy or entire live Sionna TTI runner.
2. General multi-RBG whole-TB BLER estimation is still an independent PHY integration prerequisite; Task 3 appropriately refuses to fabricate it.
3. Task 5 needs objective metrics and an actual reproducible full-loop integration smoke before making end-to-end claims.
