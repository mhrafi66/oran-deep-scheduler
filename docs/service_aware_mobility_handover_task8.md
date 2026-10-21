# Task 8 — Mobility / handover / service-aware queue routing

**Purpose:** Connect existing UE kinematics and serving-cell handover decisions to the **new** service-aware per-flow queue identities, without copying queued packets or modifying the Nokia reproduction. This is an opt-in validation/integration bridge. It is not a complete multicell runtime or a claim of 3GPP-compliant RRC procedures.

## Actual modeled behavior

1. The existing `MobilityEngine` computes `UE_pos(t) = UE_pos(0) + velocity * (t * tti_duration_s)` deterministically. Stable global UE ID and initial topology are never resampled by the bridge.
2. A supplied `measurements(snapshot)` callback is invoked once per TTI and returns linear radio powers `[global_UE, BS]` on the same device. Use existing measurements, e.g. `SionnaPathlossHandoverMeasurementProvider(snapshot.tti_index)`, or a report extracted from the already generated physical channel. **Do not generate a second Sionna channel unless explicitly choosing that measurement model.**
3. The existing `HandoverController` applies hysteresis and consecutive-TTI time-to-trigger. "Near enough to a neighbor" by itself is not a handover; its measured signal must satisfy the configured handover condition.
4. A handover updates global serving-cell membership. The authoritative Task-1 queue remains indexed by global UE ID, so the same packets, deadline times, partially served bytes and Task-4 observation history remain intact. This is a **logical reassociation**, not copying local queues. The bridge neither updates nor duplicates packet objects.
5. After handover, `eligible_ue_ids(cell)` exposes the UE in the new serving cell's backlog-positive TDS *eligible* pool, and removes it from the old cell's pool. The TDS policy must still decide whether to schedule it. `cell_observation(global_observation,cell)` prevents the policy from receiving the QoS state of other cells.
6. `membership_generation` increments whenever any UE reassociates. **Do not reuse Task-7 fixed-membership CSI**, combiners, precomputed serving channels or old cell-local candidate indices after that event. Rebuild affected cell-local CSI/policy adapters and reset or reinitialize feedback history with explicit bootstrap handling. The bridge cannot silently rebuild external objects it does not own.

## Strict execution order

For each global TTI `t`:

```
queues.begin_tti(t, arrivals)       # authoritative, global-ID queues
qos = observer.observe()            # pre-service, no future information
routing = bridge.advance(t)          # move UE, measure and decide handover
local = bridge.cell_observation(qos, target_cell)
eligible = bridge.eligible_ue_ids(target_cell)
# independent cell policy selects a feasible RBG allocation for eligible IDs
# current H_t (not stale CSI) is used by whole-TB PHY and decoding
# Task-2/5 successful service updates the SAME global queue owner
```

In the included runner smoke, the mobility bridge is invoked inside the policy after the Task-5 runner admits arrivals and produces its pre-service observation; the link provider is selected by the new serving cell. The deliberately small smoke schedules one cell per slot; it **does not certify simultaneous scheduling of 21 cells**.

## Efficiency and safety

- 1 geometric update, 1 measurement callback and 1 handover state update per TTI. No additional PHY evaluation is performed by the bridge.
- Per-UE backlog is taken from Task 1's cached counters, not a packet scan; packet migration is O(0), because global-ID packet ownership is preserved.
- Global serving-cell membership is materialized on CPU only at initialization and when a handover actually occurs. A membership generation is published to invalidate old CSI data.
- Call the bridge only once in increasing consecutive TTI order, with `num_ues` and `tti_duration_s` matching queues. Construct a new bridge/mobility/handover controller for a new episode; queues/observer/runner also require their documented reset.
- Handover is evaluated on the supplied **power metric**; different choices (path-gain proxy, filtered RSRP, Sionna channel-averaged pathloss) have different physical meanings. Record it in experiments.

## Explicit limitations — not fixed by this task

- **No full geometry/channel coupling:** Existing Task-7 Sionna short windows model Doppler, but independent windows are not guaranteed to preserve all stochastic channel state as UEs move. The bridge accepts external radio measurements, not a new spatially-consistent channel simulator. A mobility-aware frame provider must be validated separately.
- **No automatic Task-7 CSI remapping:** Changing cell membership requires rebuilding the old fixed-member CSI adapter. The generation signal makes stale reuse visible, but a production 21-cell orchestrator is still required to instantiate/dispatch new cell-local CSI callbacks.
- **No full multi-cell PHY runtime:** The included Task-5 smoke routes the UE's service into the new cell using a deterministic whole-TB link; it does NOT calculate a real post-RZF Sionna TB BLER during the handover or schedule 21 cells concurrently.
- **No NR control plane:** Handovers are assumed interruption-free, and no RRC signaling, security procedures, buffer forwarding latency, or HARQ process migration is modeled. If Task 6 HARQ is enabled at handover, outstanding TBs must be explicitly dropped/settled/reinitialized according to the experiment's declared policy; do not silently carry stale local process indices.
- **UE cell eligibility ≠ TDS candidate selection:** Handover makes the UE available to the target cell's TDS; it is not necessarily selected that slot.
- The smoke uses **0 dB hysteresis** and a UE initialized millimeters from a synthetic two-BS boundary to trigger a visible event within five slots. Realistic mobility evaluations should configure appropriate hysteresis, thresholds, durations and geometries.

## Verification on allocated A100 / H100

```bash
conda activate oran_scheduler
cd ~/oran-deep-scheduler-unified
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest -q tests/simulator/test_service_aware_mobility_handover.py
python scripts/run_service_aware_mobility_handover_smoke.py --device cuda:0
python scripts/run_service_aware_temporal_csi_smoke.py --real-sionna --device cuda:0
python scripts/run_service_aware_downlink_smoke.py --real-sionna --device cuda:0
python -m pytest -q
```

Markers: `SERVICE_AWARE_MOBILITY_HANDOVER_COMPONENT_PASS` and existing Task-7/5 Sionna markers. **These are distinct validations**; they are not proof of a unified physically spatial-consistent, continuous 21-cell mobility + PHY + HARQ execution path.
