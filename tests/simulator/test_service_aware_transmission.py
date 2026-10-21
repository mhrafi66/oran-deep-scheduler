"""Task 2: explicit whole-TB decode outcomes and authoritative queue service."""

from types import SimpleNamespace

import pytest
import torch

from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival,
    QoSFlowSpec,
    ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.service_aware_transmission import (
    DownlinkTB,
    ServiceAwareTransmissionEngine,
    single_rbg_transport_blocks_from_evaluation,
)


def queue(*, n=2, deadline=None):
    specs = [
        QoSFlowSpec(0, "urgent", 7, 20, pdb_ttis=deadline),
        QoSFlowSpec(0, "background", 9, 90),
    ]
    if n > 1:
        specs.append(QoSFlowSpec(1, "background", 9, 90))
    return ServiceAwareTrafficManager(num_ues=n, flows=specs, tti_duration_s=0.0005)


def start(q, t, *arrivals):
    q.begin_tti(tti_index=t, arrivals=[FlowPacketArrival(*a) for a in arrivals])


def eval_rates(nominal, goodput):
    return SimpleNamespace(
        candidate_nominal_rate_bps=torch.tensor(nominal, dtype=torch.float32),
        candidate_expected_goodput_bps=torch.tensor(goodput, dtype=torch.float32),
    )


def test_decode_success_services_two_flows_in_same_ue_one_tb():
    q = queue()
    engine = ServiceAwareTransmissionEngine(seed=3)
    start(q, 0, (0, "urgent", 100), (0, "background", 70), (1, "background", 90))
    result = engine.execute_tti(
        queues=q,
        transport_blocks=[DownlinkTB(0, 180, 0.0), DownlinkTB(1, 90, 0.0)],
        planned_flow_bytes={(0, "urgent"): 100, (0, "background"): 70, (1, "background"): 90},
    )
    assert len(result.tb_outcomes) == 2
    assert all(o.decoded for o in result.tb_outcomes)
    assert (result.scheduled_capacity_bytes, result.decoded_capacity_bytes) == (270, 270)
    assert result.traffic_service.delivered_bytes == 260
    assert result.unused_decoded_capacity_bytes == 10  # TB padding
    assert q.backlog_bytes == 0
    assert result.traffic_service.completed_packets == 3


def test_decode_failure_does_not_change_any_queue_bytes():
    q = queue(n=1)
    engine = ServiceAwareTransmissionEngine(seed=0)
    start(q, 0, (0, "urgent", 100))
    outcome = engine.execute_tti(
        queues=q, transport_blocks=[DownlinkTB(0, 100, 1.0)],
        planned_flow_bytes={(0, "urgent"): 100},
    )
    assert outcome.tb_outcomes[0].decoded is False
    assert outcome.decoded_capacity_bytes == 0
    assert outcome.failed_capacity_bytes == 100
    assert outcome.traffic_service.delivered_bytes == 0
    assert q.backlog_bytes == 100
    start(q, 1)
    next_ = engine.execute_tti(
        queues=q, transport_blocks=[DownlinkTB(0, 100, 0.0)],
        planned_flow_bytes={(0, "urgent"): 100},
    )
    assert next_.traffic_service.delivered_bytes == 100
    assert q.backlog_bytes == 0


def test_failure_then_deadline_expiry_counts_as_miss_not_success():
    q = queue(n=1, deadline=1)
    e = ServiceAwareTransmissionEngine(seed=5)
    start(q, 0, (0, "urgent", 70))
    e.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 70, 1.0)],
                  planned_flow_bytes={(0, "urgent"): 70})
    tti1 = q.begin_tti(tti_index=1, arrivals=[])
    assert (tti1.new_deadline_misses, tti1.expired_dropped_bytes) == (1, 70)
    r = e.execute_tti(queues=q, transport_blocks=[], planned_flow_bytes={})
    assert (r.traffic_service.delivered_bytes, r.traffic_service.backlog_bytes) == (0, 0)
    assert q.flow_snapshot(0, "urgent").on_time_completed_packets == 0


def test_independent_tb_failures_serve_only_successful_ue():
    q = queue()
    start(q, 0, (0, "urgent", 100), (1, "background", 80))
    result = ServiceAwareTransmissionEngine(seed=0).execute_tti(
        queues=q, transport_blocks=[DownlinkTB(0, 100, 1.0), DownlinkTB(1, 80, 0.0)],
        planned_flow_bytes={(0, "urgent"): 100, (1, "background"): 80},
    )
    assert [t.decoded for t in result.tb_outcomes] == [False, True]
    assert result.traffic_service.delivered_bytes_by_ue == (0, 80)
    assert q.ue_backlog_bytes() == (100, 0)
    assert (result.decoded_capacity_bytes, result.failed_capacity_bytes) == (80, 100)


def test_empty_tti_completes_without_any_ue_or_tb():
    q = queue(n=1)
    start(q, 0)
    r = ServiceAwareTransmissionEngine().execute_tti(queues=q, transport_blocks=[], planned_flow_bytes={})
    assert (r.scheduled_capacity_bytes, r.decoded_capacity_bytes, r.traffic_service.delivered_bytes) == (0, 0, 0)
    assert not q.has_active_tti


def test_partial_packet_service_and_unfilled_capacity():
    q = queue(n=1)
    start(q, 0, (0, "urgent", 140))
    e = ServiceAwareTransmissionEngine()
    r = e.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 60, 0.0)],
                      planned_flow_bytes={(0, "urgent"): 60})
    assert (r.traffic_service.delivered_bytes, r.traffic_service.completed_packets, q.backlog_bytes) == (60, 0, 80)
    start(q, 1)
    r = e.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 100, 0.0)],
                      planned_flow_bytes={(0, "urgent"): 80})
    assert (r.traffic_service.delivered_bytes, r.unused_decoded_capacity_bytes) == (80, 20)
    assert q.backlog_bytes == 0
    assert q.flow_snapshot(0, "urgent").completed_packets == 1


@pytest.mark.parametrize("wrong_plan,match", [
    ({(0, "urgent"): 101}, "queued"),
    ({(0, "urgent"): 70, (0, "background"): 60}, "queued"),
    ({(0, "missing"): 10}, "Unknown"),
    ({(0, "urgent"): -1}, "nonnegative"),
    ({(0, "urgent"): 1.5}, "nonnegative"),
])
def test_invalid_flow_claim_fails_atomically(wrong_plan, match):
    q = queue(n=1)
    start(q, 0, (0, "urgent", 100))
    e = ServiceAwareTransmissionEngine(seed=17)
    with pytest.raises(ValueError, match=match):
        e.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 100, 0.0)],
                      planned_flow_bytes=wrong_plan)
    assert q.has_active_tti and q.backlog_bytes == 100
    r = e.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 100, 0.0)],
                      planned_flow_bytes={(0, "urgent"): 100})
    assert r.traffic_service.delivered_bytes == 100


def test_tb_capacity_overclaim_is_rejected_before_mutation():
    q = queue(n=1)
    start(q, 0, (0, "urgent", 100))
    e = ServiceAwareTransmissionEngine()
    with pytest.raises(ValueError, match="exceed TB"):
        e.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 30, 0.0)],
                      planned_flow_bytes={(0, "urgent"): 60})
    assert q.backlog_bytes == 100 and q.has_active_tti
    with pytest.raises(ValueError, match="no TB exists"):
        e.execute_tti(queues=q, transport_blocks=[],
                      planned_flow_bytes={(0, "urgent"): 10})
    assert q.backlog_bytes == 100


def test_duplicate_tb_or_invalid_ue_rejected():
    q = queue(n=1)
    start(q, 0, (0, "urgent", 100))
    e = ServiceAwareTransmissionEngine()
    with pytest.raises(ValueError, match="At most one"):
        e.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 10, 0.1), DownlinkTB(0, 10, 0.1)], planned_flow_bytes={})
    with pytest.raises(ValueError, match="outside"):
        e.execute_tti(queues=q, transport_blocks=[DownlinkTB(1, 10, 0.1)], planned_flow_bytes={})
    assert q.has_active_tti


@pytest.mark.parametrize("value", [-1.0, 1.00001, float("nan"), float("inf"), True])
def test_invalid_tbler_rejected(value):
    with pytest.raises(ValueError, match="probability"):
        DownlinkTB(0, 100, value)


@pytest.mark.parametrize("ue,cap", [(-1, 10), (True, 10), (0, 0), (0, -1), (0, True)])
def test_invalid_tb_identity_or_capacity_rejected(ue, cap):
    with pytest.raises(ValueError):
        DownlinkTB(ue, cap, 0.2)


def test_reproducible_decoding_across_seeds_and_reset():
    e1 = ServiceAwareTransmissionEngine(seed=42)
    e2 = ServiceAwareTransmissionEngine(seed=42)

    def run(engine):
        q = queue(n=1)
        decisions = []
        for t in range(25):
            start(q, t, (0, "urgent", 5000))
            r = engine.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 250, 0.38)],
                                   planned_flow_bytes={(0, "urgent"): 250})
            decisions.append(r.tb_outcomes[0].decoded)
        return decisions

    first = run(e1)
    assert first == run(e2)
    assert 0 < sum(first) < 25
    e1.reset()
    assert first == run(e1)


def test_failed_validation_does_not_consume_rng():
    e1 = ServiceAwareTransmissionEngine(seed=42)
    e2 = ServiceAwareTransmissionEngine(seed=42)
    q = queue(n=1)
    start(q, 0, (0, "urgent", 10))
    with pytest.raises(ValueError):
        e1.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 10, 0.5)],
                       planned_flow_bytes={(0, "urgent"): 11})
    a = e1.execute_tti(queues=q, transport_blocks=[DownlinkTB(0, 10, 0.5)],
                       planned_flow_bytes={(0, "urgent"): 10})
    q2 = queue(n=1)
    start(q2, 0, (0, "urgent", 10))
    b = e2.execute_tti(queues=q2, transport_blocks=[DownlinkTB(0, 10, 0.5)],
                       planned_flow_bytes={(0, "urgent"): 10})
    assert a.tb_outcomes[0].decoded == b.tb_outcomes[0].decoded


def test_no_active_queue_tti_is_rejected():
    q = queue(n=1)
    with pytest.raises(RuntimeError, match="begin_tti"):
        ServiceAwareTransmissionEngine().execute_tti(queues=q, transport_blocks=[], planned_flow_bytes={})


def test_one_rbg_adapter_uses_existing_rates_no_phy_recompute():
    # E.g. 100 B at 0.5 ms => 1,600,000 bps; BLER=0.25 => goodput 1,200,000 bps.
    data = eval_rates([[1_600_000, 0, 0], [0, 2_400_000, 0]],
                      [[1_200_000, 0, 0], [0, 2_400_000, 0]])
    result = single_rbg_transport_blocks_from_evaluation(
        evaluation=data, queue_ue_by_candidate=[1, 0], slot_duration_s=0.0005)
    assert result == (DownlinkTB(1, 100, 0.25), DownlinkTB(0, 150, 0.0))


def test_one_rbg_adapter_ignores_empty_padded_candidate():
    result = single_rbg_transport_blocks_from_evaluation(
        evaluation=eval_rates([[0, 0], [0, 800_000]], [[0, 0], [0, 0]]),
        queue_ue_by_candidate=[-1, 0], slot_duration_s=0.0005)
    assert result == (DownlinkTB(0, 50, 1.0),)


def test_adapter_explicitly_rejects_multi_rbg_tbler_surrogate():
    with pytest.raises(ValueError, match="Multi-RBG"):
        single_rbg_transport_blocks_from_evaluation(
            evaluation=eval_rates([[800_000, 800_000]], [[720_000, 720_000]]),
            queue_ue_by_candidate=[0], slot_duration_s=0.0005)


def test_adapter_rejects_bad_rate_contract():
    with pytest.raises(ValueError, match="exceeds nominal"):
        single_rbg_transport_blocks_from_evaluation(
            evaluation=eval_rates([[800_000]], [[1_000_000]]),
            queue_ue_by_candidate=[0], slot_duration_s=0.0005)


def test_end_to_end_one_rbg_adapter_success_accounting():
    q = queue(n=1)
    start(q, 0, (0, "urgent", 70))
    blocks = single_rbg_transport_blocks_from_evaluation(
        evaluation=eval_rates([[1_600_000]], [[1_600_000]]),
        queue_ue_by_candidate=[0], slot_duration_s=0.0005)
    r = ServiceAwareTransmissionEngine().execute_tti(
        queues=q, transport_blocks=blocks, planned_flow_bytes={(0, "urgent"): 70})
    assert r.traffic_service.delivered_bytes == 70 and q.backlog_bytes == 0
    assert r.unused_decoded_capacity_bytes == 30


def test_adapter_accepts_preexisting_candidate_index_tensor_without_scalar_reads():
    data = eval_rates([[0, 800_000], [800_000, 0]], [[0, 800_000], [600_000, 0]])
    blocks = single_rbg_transport_blocks_from_evaluation(
        evaluation=data,
        queue_ue_by_candidate=torch.tensor([1, 0], dtype=torch.int64),
        slot_duration_s=0.0005,
    )
    assert blocks == (DownlinkTB(1, 50, 0.0), DownlinkTB(0, 50, 0.25))


def test_adapter_rejects_float_ue_mapping():
    with pytest.raises(ValueError, match="integer tensor"):
        single_rbg_transport_blocks_from_evaluation(
            evaluation=eval_rates([[800_000]], [[800_000]]),
            queue_ue_by_candidate=torch.tensor([0.0]),
            slot_duration_s=0.0005,
        )
