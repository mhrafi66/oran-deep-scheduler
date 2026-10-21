"""Task 1: correctness, conservation and efficient per-flow packet accounting."""

import random

import pytest

from oran_scheduler.simulator.service_aware_traffic import (
    FlowPacketArrival,
    QoSFlowSpec,
    ServiceAwareTrafficManager,
)
from oran_scheduler.simulator.traffic import FTP3TrafficConfig


def manager(*flows, num_ues=2, seed=7):
    return ServiceAwareTrafficManager(
        num_ues=num_ues, flows=flows, tti_duration_s=0.0005, seed=seed
    )


def flow(ue=0, name="urgent", **kwargs):
    return QoSFlowSpec(ue, name, five_qi=7, priority=20, **kwargs)


def arrival(ue, name, n):
    return FlowPacketArrival(ue, name, n)


def test_two_flows_one_ue_have_independent_fifo_and_metadata():
    m = manager(
        flow(0, "urgent", pdb_ttis=2, gbr_bps=1_000_000),
        flow(0, "background"), flow(1, "background"),
    )
    start = m.begin_tti(tti_index=0, arrivals=[
        arrival(0, "urgent", 120), arrival(0, "background", 200),
        arrival(1, "background", 70),
    ])
    assert (start.arrived_bytes, start.admitted_bytes, start.backlog_bytes) == (390, 390, 390)
    assert m.ue_backlog_bytes() == (320, 70)
    assert m.flow_snapshot(0, "urgent").spec.gbr_bps == 1_000_000
    result = m.apply_service(
        service_bytes_by_flow={(0, "urgent"): 80, (0, "background"): 20, (1, "background"): 70},
        ue_success_capacity_bytes=[100, 70],
    )
    assert result.delivered_bytes == 170
    assert result.completed_packets == 1
    assert result.on_time_completed_packets == 1
    assert result.delivered_bytes_by_ue == (100, 70)
    assert m.ue_backlog_bytes() == (220, 0)
    snap = m.flow_snapshot(0, "urgent")
    assert (snap.backlog_bytes, snap.queued_packets, snap.hol_delay_ttis) == (40, 1, 1)


def test_partial_packet_service_stays_in_fifo_and_counts_only_completion():
    m = manager(flow(), num_ues=1)
    m.begin_tti(tti_index=0, arrivals=[arrival(0, "urgent", 100), arrival(0, "urgent", 80)])
    s = m.apply_service(service_bytes_by_flow={(0, "urgent"): 120}, ue_success_capacity_bytes=[120])
    assert (s.completed_packets, s.delivered_bytes, s.backlog_bytes) == (1, 120, 60)
    m.begin_tti(tti_index=1, arrivals=[])
    s2 = m.apply_service(service_bytes_by_flow={(0, "urgent"): 60})
    assert (s2.completed_packets, s2.backlog_bytes) == (1, 0)
    assert m.flow_snapshot(0, "urgent").completed_packets == 2


def test_inclusive_deadline_and_drop_only_remaining_bytes():
    m = manager(flow(pdb_ttis=2), num_ues=1)
    m.begin_tti(tti_index=4, arrivals=[arrival(0, "urgent", 100)])
    m.apply_service(service_bytes_by_flow={(0, "urgent"): 30})
    m.begin_tti(tti_index=5, arrivals=[])
    s = m.apply_service(service_bytes_by_flow={(0, "urgent"): 0})
    assert s.backlog_bytes == 70  # deadline at tti 5, still eligible
    start = m.begin_tti(tti_index=6, arrivals=[])
    assert (start.new_deadline_misses, start.expired_dropped_bytes, start.backlog_bytes) == (1, 70, 0)
    final = m.apply_service(service_bytes_by_flow={(0, "urgent"): 500})
    assert (final.delivered_bytes, final.unused_requested_service_bytes) == (0, 500)
    assert m.flow_snapshot(0, "urgent").deadline_misses == 1
    assert m.flow_snapshot(0, "urgent").expired_dropped_packets == 1


def test_retained_expired_packet_can_be_completed_late_once():
    m = manager(flow(pdb_ttis=1, expiry_policy="retain"), num_ues=1)
    m.begin_tti(tti_index=0, arrivals=[arrival(0, "urgent", 80)])
    m.apply_service(service_bytes_by_flow={})
    first = m.begin_tti(tti_index=1, arrivals=[])
    assert (first.new_deadline_misses, first.expired_dropped_bytes, first.backlog_bytes) == (1, 0, 80)
    m.apply_service(service_bytes_by_flow={})
    next_tti = m.begin_tti(tti_index=2, arrivals=[])
    assert next_tti.new_deadline_misses == 0
    result = m.apply_service(service_bytes_by_flow={(0, "urgent"): 80})
    assert (result.completed_packets, result.on_time_completed_packets, result.late_completed_packets) == (1, 0, 1)
    assert m.flow_snapshot(0, "urgent").deadline_misses == 1


def test_deadlines_expire_without_scanning_future_packet_backlog():
    m = manager(flow(pdb_ttis=2, expiry_policy="drop"), num_ues=1)
    m.begin_tti(tti_index=0, arrivals=[arrival(0, "urgent", 10)])
    m.apply_service(service_bytes_by_flow={})
    m.begin_tti(tti_index=1, arrivals=[arrival(0, "urgent", 20)])
    m.apply_service(service_bytes_by_flow={})
    start = m.begin_tti(tti_index=2, arrivals=[])
    assert start.expired_dropped_bytes == 10  # one expired, later arrival still pending
    assert start.backlog_bytes == 20
    m.apply_service(service_bytes_by_flow={(0, "urgent"): 20})
    assert m.flow_snapshot(0, "urgent").on_time_completed_packets == 1


def test_overflow_rejects_whole_packet_and_uses_remaining_backlog():
    m = manager(flow(max_queue_bytes=100), num_ues=1)
    start = m.begin_tti(tti_index=0, arrivals=[arrival(0, "urgent", 70), arrival(0, "urgent", 40)])
    assert (start.admitted_packets, start.overflow_dropped_packets, start.backlog_bytes) == (1, 1, 70)
    m.apply_service(service_bytes_by_flow={(0, "urgent"): 50})
    start = m.begin_tti(tti_index=1, arrivals=[arrival(0, "urgent", 80)])
    assert (start.admitted_packets, start.backlog_bytes) == (1, 100)
    m.apply_service(service_bytes_by_flow={})
    snap = m.flow_snapshot(0, "urgent")
    assert (snap.arrived_bytes, snap.overflow_dropped_bytes, snap.backlog_bytes) == (190, 40, 100)


def test_invalid_multi_flow_capacity_is_atomic():
    m = manager(flow(0, "a"), flow(0, "b"), num_ues=1)
    m.begin_tti(tti_index=0, arrivals=[arrival(0, "a", 20), arrival(0, "b", 20)])
    with pytest.raises(ValueError, match="exceeds"):
        m.apply_service(service_bytes_by_flow={(0, "a"): 10, (0, "b"): 10}, ue_success_capacity_bytes=[15])
    assert m.has_active_tti
    assert m.backlog_bytes == 40
    with pytest.raises(ValueError, match="Unknown"):
        m.apply_service(service_bytes_by_flow={(0, "c"): 15})
    assert m.backlog_bytes == 40
    result = m.apply_service(service_bytes_by_flow={(0, "a"): 10, (0, "b"): 10}, ue_success_capacity_bytes=[20])
    assert (result.delivered_bytes, result.backlog_bytes) == (20, 20)


def test_replayed_arrival_trace_never_calls_poisson(monkeypatch):
    ftp = FTP3TrafficConfig(100, 4000.0, 0.0005)
    m = manager(flow(ftp3=ftp), num_ues=1)
    monkeypatch.setattr(m, "sample_ftp3_arrivals", lambda: (_ for _ in ()).throw(AssertionError("resampled")))
    m.begin_tti(tti_index=0, arrivals=[arrival(0, "urgent", 42)])
    m.apply_service(service_bytes_by_flow={})
    assert m.backlog_bytes == 42


def test_seeded_batched_ftp3_sampling_and_reset_are_reproducible():
    ftp = FTP3TrafficConfig(100, 4000.0, 0.0005)
    specs = [flow(0, "a", ftp3=ftp), flow(1, "b", ftp3=ftp)]
    m1 = manager(*specs, seed=17)
    m2 = manager(*specs, seed=17)
    trace = []
    for t in range(12):
        start1 = m1.begin_tti(tti_index=t)  # auto sample
        start2 = m2.begin_tti(tti_index=t)
        assert start1 == start2
        trace.append(start1.arrived_packets)
        m1.apply_service(service_bytes_by_flow={})
        m2.apply_service(service_bytes_by_flow={})
    assert sum(trace) > 0
    m1.reset()
    replay = []
    for t in range(12):
        replay.append(m1.begin_tti(tti_index=t).arrived_packets)
        m1.apply_service(service_bytes_by_flow={})
    assert replay == trace


def test_invalid_arrival_does_not_mutate_active_or_backlogs():
    m = manager(flow(), num_ues=1)
    with pytest.raises(ValueError, match="Unknown"):
        m.begin_tti(tti_index=0, arrivals=[arrival(0, "urgent", 20), arrival(0, "typo", 10)])
    assert m.backlog_bytes == 0 and not m.has_active_tti
    m.begin_tti(tti_index=0, arrivals=[arrival(0, "urgent", 20)])
    with pytest.raises(RuntimeError, match="Previous TTI"):
        m.begin_tti(tti_index=1, arrivals=[])
    with pytest.raises(RuntimeError, match="middle of a TTI"):
        m.reset()
    m.apply_service(service_bytes_by_flow={})
    with pytest.raises(ValueError, match="contiguous"):
        m.begin_tti(tti_index=2, arrivals=[])


def test_duplicate_flow_and_invalid_metadata_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        manager(flow(), flow(), num_ues=1)
    with pytest.raises(ValueError, match="five_qi"):
        QoSFlowSpec(0, "bad", five_qi=0, priority=5)
    with pytest.raises(ValueError, match="pdb_ttis"):
        flow(pdb_ttis=0)
    with pytest.raises(ValueError, match="GBR|gbr_bps"):
        flow(gbr_bps=float("nan"))
    with pytest.raises(ValueError, match="positive"):
        FlowPacketArrival(0, "urgent", 0)


def test_conservation_over_long_mixed_arrival_service_trace():
    specs = [
        flow(0, "urgent", pdb_ttis=3, max_queue_bytes=900),
        flow(0, "bulk", max_queue_bytes=3000),
        flow(1, "urgent", pdb_ttis=2, expiry_policy="retain", max_queue_bytes=700),
    ]
    m = manager(*specs)
    rng = random.Random(63)
    for t in range(450):
        events = []
        for spec in specs:
            for _ in range(rng.randint(0, 3)):
                events.append(arrival(spec.ue_index, spec.flow_id, rng.randint(10, 110)))
        start = m.begin_tti(tti_index=t, arrivals=events)
        assert start.arrived_bytes == start.admitted_bytes + start.overflow_dropped_bytes
        service = {s.key: rng.randint(0, 180) for s in specs}
        result = m.apply_service(service_bytes_by_flow=service)
        assert m.backlog_bytes == sum(m.ue_backlog_bytes())
        assert result.backlog_bytes >= 0
        for spec in specs:
            snap = m.flow_snapshot(*spec.key)
            assert snap.arrived_bytes == (
                snap.overflow_dropped_bytes + snap.expired_dropped_bytes +
                snap.delivered_bytes + snap.backlog_bytes
            )
            assert snap.queued_packets >= 0
            assert snap.completed_packets == snap.on_time_completed_packets + snap.late_completed_packets
            assert snap.deadline_misses >= snap.late_completed_packets + snap.expired_dropped_packets


def test_many_packets_do_not_require_resumming_backlog():
    m = manager(flow(pdb_ttis=10_000), num_ues=1)
    packets = [arrival(0, "urgent", 12) for _ in range(15_000)]
    start = m.begin_tti(tti_index=0, arrivals=packets)
    assert start.backlog_bytes == 180_000
    assert m.ue_backlog_bytes() == (180_000,)
    result = m.apply_service(service_bytes_by_flow={(0, "urgent"): 6_001})
    assert result.delivered_bytes == 6_001
    assert m.flow_snapshot(0, "urgent").backlog_bytes == 173_999


def test_completion_on_final_allowed_tti_is_not_a_miss():
    m = manager(flow(pdb_ttis=2), num_ues=1)
    m.begin_tti(tti_index=0, arrivals=[arrival(0, "urgent", 80)])
    m.apply_service(service_bytes_by_flow={(0, "urgent"): 10})
    m.begin_tti(tti_index=1, arrivals=[])
    result = m.apply_service(service_bytes_by_flow={(0, "urgent"): 70})
    assert (result.on_time_completed_packets, result.backlog_bytes) == (1, 0)
    start = m.begin_tti(tti_index=2, arrivals=[])
    assert start.new_deadline_misses == 0
    m.apply_service(service_bytes_by_flow={})


def test_invalid_negative_grant_and_cap_do_not_mutate_queues():
    m = manager(flow(), num_ues=1)
    m.begin_tti(tti_index=0, arrivals=[arrival(0, "urgent", 40)])
    with pytest.raises(ValueError, match="non-negative"):
        m.apply_service(service_bytes_by_flow={(0, "urgent"): -1})
    with pytest.raises(ValueError, match="non-negative"):
        m.apply_service(service_bytes_by_flow={(0, "urgent"): 10}, ue_success_capacity_bytes=[-1])
    assert m.backlog_bytes == 40
    m.apply_service(service_bytes_by_flow={(0, "urgent"): 40}, ue_success_capacity_bytes=[40])
    assert m.backlog_bytes == 0
