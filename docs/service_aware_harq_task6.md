# Task 6: opt-in lightweight HARQ diagnostic

## Scope

This is a conservative **one-HARQ-process-per-UE stop-and-wait abstraction**. It
is an experiment/validation tool, not a bit-accurate or performance-faithful NR
HARQ model. Real 5G NR supports multiple parallel HARQ processes, soft
combining and additional MAC/RLC retransmission details. This implementation
**will under-utilize radio resources** when the feedback delay exceeds one
slot; never use its throughput to claim commercial 5G performance.

Use it by passing `ServiceAwareHARQEngine` instead of
`ServiceAwareTransmissionEngine` to the existing
`ServiceAwareDownlinkCellRunner`. One optional preparation hook is added to
the runner, called after `queues.begin_tti` and before its policy; ordinary
Task-5 decoders are unchanged. No channel, CSI, TB-size or queue code is
reimplemented. The entire original Nokia simulation path remains untouched.

## Exact order each TTI

1. Queue `begin_tti`: expire deadline-dropped packets, admit arrivals.
2. Optional HARQ `prepare_tti`: cancel invalid processes, mature feedback.
3. QoS observation, then policy consults `harq.statuses()`.
4. PHY/NR grant construction using existing Task-5 implementation.
5. Flow packing: for a `retransmit` status, provide **the exact same flow
   payload map** as `dict(status.planned_flow_bytes)` and a TB of the
   **same payload capacity**. Use a *new* whole-grant TBLER for the currently
   proposed PHY allocation; no invented soft-combining gain.
6. HARQ decoder samples one outcome per TB, applies successful bytes to the
   authoritative queue, and preserves only lightweight process metadata for
   retransmissions and acknowledgment timing.
7. QoS metrics/conservation are collected by the existing Task-5 runner.

### Pending process states

- `awaiting_feedback`: no TB for that UE is permitted. Its process remains
  outstanding until feedback_ready_tti.
- `retransmit`: previous decoding failed and its NACK matured. The policy
  **may** schedule that UE for retransmission. Delaying the retransmission
  is allowed; retransmission priority must be implemented by the policy.
- Successful decoding removes queued payload bytes immediately, but keeps
  HARQ process occupancy blocked until ACK latency elapses. This preserves
  packet-reception timing. **Queue observations can reveal the successful
  decode before ACK arrives**, so this is not strict feedback-information
  hiding; avoid making 3GPP feedback-timing claims from it.

### Expiry, final failures and packet identity

- Failed TBs do not remove bytes from Task-1 queues. The pending UE cannot
  transmit any fresh TB until feedback/retry completes, protecting FIFO
  payload identity without a duplicated HARQ payload queue.
- If any reserved flow experiences packet expiry, a failed process is
  conservatively canceled. The engine may cancel an otherwise valid TB if
  a *different*, later packet in that same flow expired; false retransmission
  is worse than losing a retransmission opportunity.
- On `max_transmissions` failures, the **HARQ process is discarded**, but
  the bytes remain in the authoritative queue. Whether higher-layer RLC
  retries them or the flow eventually drops them is NOT simulated.
- A packet with `expiry_policy='retain'` may be delivered late, consistent
  with Task-1 semantics. `expiry_policy='drop'` removes packets when their
  deadlines pass; no ghost bytes are delivered from old HARQ processes.

### Efficiency

State scales with at most one small metadata record per pending UE. No
packet copies, GPU tensors, CPU-GPU transfers, waveform decoding, channel
recomputation, or repeated scan of all queued packets. Retransmissions use
existing TB-sizing, Sionna whole-grant link predictions and Task-2 decode
RNG. The additional pre-slot work is O(number of pending UEs and their
reserved flow count); a separate GPU performance measurement would be
needed before claiming a particular runtime overhead.

## Validation

Run directly in a **Slurm-accounted interactive compute shell** on allocated
Notchpeak (not on `notchpeak2`):

```bash
conda activate oran_scheduler
cd ~/oran-deep-scheduler-unified
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/simulator/test_service_aware_harq.py
python scripts/run_service_aware_harq_smoke.py
python scripts/run_service_aware_downlink_smoke.py --real-sionna --device cuda:0
python -m pytest -q
```

The new deterministic HARQ smoke validates queue, retransmission, feedback,
and conservation; the existing Sionna smoke is a **regression of the original
PHY path**, not an end-to-end HARQ-over-Sionna certification. For any HARQ
publication experiment, first add a real-Sionna HARQ loop plus parallel HARQ
process modeling, or document explicitly why stop-and-wait is appropriate.

Only commit and push after CHPC tests pass. Do not merge into the Nokia
reproduction baseline automatically.
