import pytest

from oran_scheduler.simulator.packet_qos import (
    PacketFIFOQueue,
)


def test_fifo_packet_service_and_delay() -> None:

    queue = PacketFIFOQueue()

    assert queue.enqueue(
        arrival_tti=0,
        size_bits=100.0,
    )

    assert queue.enqueue(
        arrival_tti=1,
        size_bits=50.0,
    )

    result = queue.serve(
        tti_index=2,
        capacity_bits=120.0,
    )

    assert result.delivered_bits == pytest.approx(
        120.0
    )

    assert result.completed_packets == 1

    assert result.completed_delay_ttis == (
        2,
    )

    assert result.queue_bits_after == pytest.approx(
        30.0
    )

    assert result.hol_delay_ttis_after == 1


def test_packet_queue_overflow_drop() -> None:

    queue = PacketFIFOQueue(
        max_queue_bits=100.0
    )

    assert queue.enqueue(
        arrival_tti=0,
        size_bits=80.0,
    )

    assert not queue.enqueue(
        arrival_tti=0,
        size_bits=30.0,
    )

    assert queue.total_dropped_packets == 1

    assert queue.total_dropped_bits == pytest.approx(
        30.0
    )


def test_deadline_miss() -> None:

    queue = PacketFIFOQueue()

    assert queue.enqueue(
        arrival_tti=0,
        size_bits=100.0,
        deadline_tti=2,
    )

    result = queue.serve(
        tti_index=3,
        capacity_bits=100.0,
    )

    assert result.deadline_misses == 1

    assert result.delivered_bits == pytest.approx(
        0.0
    )

    assert result.queue_bits_after == pytest.approx(
        0.0
    )
