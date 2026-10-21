# Task 3 — NR resource-to-service conversion (opt-in)

This task is deliberately **not** a change to the existing Nokia scheduler, Sionna channel, PF/greedy allocators or existing `phy/rate.py`. It prepares whole-grant `DownlinkTB` objects for the **Task 2** decoder and **Task 1** per-flow queue. It is a supported *capacity/accounting building block*, not yet an automatically connected live downlink PHY pipeline.

## Source audit motivating this implementation

* `simulator/rbg.py` uses `RBGConfig(num_rbgs=18, rbs_per_rbg=1)` in the reduced training configuration. The 18 groups are **18 PRBs**, NOT the 273 PRBs of the paper's full 100 MHz case.
* `phy/rate.py` computes Sionna NR TBS independently for every RBG with all 14 symbols/12 subcarriers, without DMRS/control RE deduction; these independently rounded sizes must **not** be added to create a TB-wide TBS.
* `phy/schedule_evaluator.py` returns per-candidate per-RBG nominal/expected rates, but does **not** expose a TB-level MCS or whole-grant BLER. We cannot infer the latter by multiplying, averaging or independently sampling RBG errors.
* Task 2 already provides `DownlinkTB`, a TB-level seeded decoder and authoritative successful per-flow queue service. This task reuses those unchanged.

## Components and examples

```python
from oran_scheduler.simulator.service_aware_nr_grants import (
    NRBandwidthPart, PDSCHDataRegion, WholeGrantLink, NRGrantAssembler,
)

# Correct full-width *geometry* (a full-width PHY channel must actually be simulated):
full_bwp = NRBandwidthPart(num_prbs=273, rbg_size_prbs=16)
assert full_bwp.prbs_per_rbg == (16,) * 17 + (1,)

# For the existing legacy 18-PRB physical channel:
engine = NRGrantAssembler(
    bandwidth=NRBandwidthPart(num_prbs=18, rbg_size_prbs=1),
    region=PDSCHDataRegion(
        num_pdsch_symbols=14,
        dmrs_re_per_prb=12,
        x_overhead_re_per_prb=0,
        higher_layer_header_bytes=0,
    ),
    channel_num_prbs=18,
)

# Existing cell allocation: shape [scheduler user slot, RBG]. Candidate 0
# occupies RBG 0 and 1, with ONE TB/MCS/rank over the combined grant.
# Crucially, this TB-wide TBLER must be computed separately on the actual
# *complete* allocation, not derived from 2 per-RBG BLER estimates.
link = WholeGrantLink(
    candidate_index=0, assigned_rbgs=(0, 1),
    mcs_index=10, num_layers=2, tbler=0.08,
)
blocks = engine.build(
    allocation=allocation,
    queue_ue_by_candidate=[0, 1, 2],
    whole_links={0: link},  # EXACTLY one link per selected candidate
)
# `blocks` may be passed directly to Task 2's execute_tti(...).
```

## Sizing semantics and guards

1. The `NRBandwidthPart` stores a contiguous BWP geometry with explicit RBG size/alignment. Partial first/last groups are never charged as complete groups. At BWP offset 0, 273 PRBs / 16 gives **17 full RBGs + one 1-PRB tail**, i.e., 18 RBGs. Other BWP offsets shift the boundaries.
2. The constructor `NRGrantAssembler(... channel_num_prbs=...)` **rejects inconsistent simulated channel bandwidth**. A 273-PRB geometry cannot be used against an 18-PRB PHY result.
3. A data region uses the 3GPP-derived `N_RE_per_PRB = min(156,12*N_symbols - DMRS_RE_per_PRB - xOverhead_RE_per_PRB)` rule. The DMRS RE count is an explicit assumption, not inferred from PHY pilots. `xOverhead` is limited to 0,6,12,18 RE/PRB. For 14 symbols, 12 DMRS RE and zero other overhead, data RE/PRB=156. This is **not** a complete scheduling of CORESET/PDCCH/PUCCH or puncturing.
4. One scheduled UE gets **one complete TB per TTI**, supported rank 1–2 and one TB-wide MCS. Distinct UEs may share RBGs through MU-MIMO; a UE may span multiple RBGs; the same UE cannot be duplicated on one RBG. Aggregate total layers/RBG may not exceed the configured bound. The full per-UE RBG assignment must exactly match the `WholeGrantLink` descriptor, guarding against stale PHY estimates.
5. The default TBS backend calls **the same Sionna** `decode_mcs_index` and `calculate_tb_size` helpers already used by `phy/rate.py`, but **once in a batch for uncached final-grant combinations** (`mcs`, total RE, layers). Its bounded LRU-like cache prevents repeated size calculations. Only the small final allocation is copied to CPU once, not the full channel/SINR tensors.
6. `tbs_bits` is byte-aligned Sionna NR TBS. `payload_capacity_bytes = tbs_bits/8 - higher_layer_header_bytes`; the header subtraction is an optional **fixed modeling assumption**, not a packet-by-packet MAC/RLC segmentation model. The capacity feeds `DownlinkTB` and cannot directly be treated as delivered bytes—Task 2 samples decoding, then applies only explicitly allocated per-flow bytes.

## A critical remaining PHY interface

**This task does not produce calibrated multi-RBG whole-TB BLER or choose TB-wide MCS from raw SINR.** The existing final cell evaluator does not retain sufficient whole-grant SINR or link-adaptation results for such a calculation. The caller must supply `WholeGrantLink` from an independently credible *whole-grant* link abstraction. Until that integration exists, real numerical TBS capacity can be validated, but **end-to-end multi-RBG reliability results should NOT be presented as physically predictive**. The smoke uses explicit TBLER endpoints 0/1 only to verify logic and queue conservation. We also do not claim the legacy reduced 18-PRB Sionna channel has been upgraded to 273 PRBs.

It would be misleading to remove this guardrail and silently turn the old per-RBG BLER into a full-TB error probability. The next separately scoped integration should expose/reuse already evaluated final-allocation per-RE SINRs and invoke a **whole-TB** Sionna link abstraction, without rerunning the channel or the entire scheduler search. This should be resolved before evaluating multi-RBG scheduling reliability.

## Run and validation

On a CHPC **GPU compute node** with the `oran_scheduler` environment:

```bash
cd ~/oran-deep-scheduler-unified
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/simulator/test_service_aware_nr_grants.py
python scripts/run_service_aware_nr_grant_smoke.py
python -m pytest -q
```

The new test suite uses an injected deterministic fake TBS backend only for control-path/geometry/queue invariants; a **separate test runs the real Sionna NR TB size** and compares with `phy/rate.py`'s underlying API. Running without Sionna skips just that cross-check; the full legacy suite will also fail to collect without Sionna.

Not included in Task 3: full NR grant construction and signaling, HARQ feedback, dynamic MAC/RLC packet segmentation, learned architecture, QoS ranking policy, or any change to legacy Nokia evaluations.
