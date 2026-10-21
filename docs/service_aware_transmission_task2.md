# Task 2: explicit transport-block outcomes (opt-in)

This is a **successful-service accounting** abstraction, not an NR HARQ or bit-accurate receiver implementation. The legacy Nokia traffic, PPO, and rate paths are unchanged.

## Contract and normal TTI call order

1. `queues.begin_tti(tti_index=t, arrivals=...)` handles arrivals and deadline expiry.
2. Existing PHY/scheduler yields **one `DownlinkTB` per served UE**, with whole-TB *payload capacity in bytes* and whole-TB *decoding error probability*. Current research mode supports at most one TB per UE/slot.
3. Caller selects which queued flows' bytes are placed inside each TB as `planned_flow_bytes={(local_ue, flow_id): bytes}`. This module **does not** choose flow QoS priorities. The sum for an individual UE cannot exceed that UE's TB payload capacity or individual flow backlogs.
4. `ServiceAwareTransmissionEngine.execute_tti(...)` samples one independent Bernoulli success per TB. On failure, **zero** flow payload from that TB reaches the Task-1 queue. On success, the *planned bytes*, not necessarily the full TB capacity, become completed service. Unfilled capacity is padding/unutilized payload. The prevalidated Task-1 manager updates FIFO queues, packet completion and deadlines.

```python
queues.begin_tti(tti_index=t, arrivals=arrivals)
result = decoder.execute_tti(
    queues=queues,
    transport_blocks=[DownlinkTB(ue_index=0, payload_capacity_bytes=100, tbler=0.1)],
    planned_flow_bytes={(0, 'urgent'): 70, (0, 'bulk'): 20},
)
```

`result.traffic_service.delivered_bytes` is **successful queued payload**, unlike `result.scheduled_capacity_bytes` (encoded payload capacity), `result.decoded_capacity_bytes` (capacity of decoded TBs), or `result.planned_flow_payload_bytes` (bits selected for transmission regardless of outcome). All numbers are integer **bytes**. **Never feed PHY bits or bps directly to `DownlinkTB`**.

## Physical interface and a non-negotiable guardrail

Existing `CellAllocationEvaluationData` contains **per-candidate, per-RBG** `candidate_nominal_rate_bps` and `candidate_expected_goodput_bps`. With **exactly one RBG assigned per candidate**, the TB can safely reuse these quantities:

- `TBLER = 1 - expected_goodput_bps / nominal_rate_bps`
- `TB_payload_bytes = round(nominal_rate_bps * slot_duration_s / 8)`

`single_rbg_transport_blocks_from_evaluation(...)` implements this conversion with a **single bulk transfer from GPU for the small final-cell result** and **no new Sionna evaluation**. `queue_ue_by_candidate` may be a Python integer sequence or an integer PyTorch tensor (bulk-copied from GPU if necessary). It rejects candidate allocations spanning multiple RBGs. We deliberately refuse to average, multiply or independently sample per-RBG TBLER as a surrogate for decoding **one TB spanning multiple RBGs**. An explicit `DownlinkTB(ue, whole_tb_bytes, whole_tb_tBler)` is already supported for such grants. **Task 3 must establish reliable whole-grant TB sizing, PHY overhead and link abstraction** before general multi-RBG allocations can drive actual packet reliability claims.

This is intentionally narrower than pretending the current per-RBG rate abstraction provides calibrated complete TB decoding probabilities. Also, the source's estimated TBLER is a link abstraction, not an over-the-air validated BLER measurement. Independent Bernoulli draws ignore burst errors/HARQ combining.

## Lifecycle and reproducibility

- RNG is `random.Random(seed)`, independent of global PyTorch RNG/Poisson arrivals.
- One random number per scheduled TB, including endpoint BLERs, so paired scenarios have consistent RNG consumption when the number and order of scheduled TBs match.
- `decoder.reset()` replays the seed; `queues.reset()` resets queue state. Caller resets both at episode boundaries.
- Calls validate all planned flow keys, queue bytes, TB capacities, UE uniqueness and probabilities before consuming RNG or applying service.
- No full per-packet scans, no repeated PHY computation, and no CPU/GPU copy in the TB decoding hot path. The optional single-RBG PHY adapter performs one batched copy per final allocation.

## Tests and limits

Run `python -m pytest -q tests/simulator/test_service_aware_transmission.py` and `python scripts/run_service_aware_transmission_smoke.py`. Then run `python -m pytest -q` on an allocated CHPC compute node. Legacy suite before this task: **446 passed**, 14 deprecation warnings. On top of that, Task 2 adds **33** focused tests (expected full suite **479 tests** if the base remains unchanged).

Explicitly deferred: full-size NR TB calculation and per-grant TBLER across 2+ RBGs; control/DMRS overhead; actual HARQ retransmission processes; MAC flow selection algorithm; policy or RL training; CSI recomputation; packet error correlation. Task 2 is a valid transaction layer but **not yet installed into the legacy Sionna runner**. Integration of whole-grant PHY results and QoS scheduling is separate work. Do not present it as a complete end-to-end 5G simulation yet.
