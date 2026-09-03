import torch

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
    TrafficBufferManager,
    build_evaluation_ftp3_config,
    build_training_ftp3_config,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def test_paper_ftp3_configurations():
    training = build_training_ftp3_config(
        tti_duration_s=0.0005,
    )

    assert (
        training.packet_size_bytes
        == 1500
    )

    assert (
        training.packet_arrival_rate_per_s
        == 500.0
    )

    assert (
        training.packet_size_bits
        == 12000.0
    )

    assert (
        training.expected_packets_per_tti
        == 0.25
    )

    evaluation = (
        build_evaluation_ftp3_config(
            tti_duration_s=0.0005,
        )
    )

    assert (
        evaluation.packet_size_bytes
        == 500_000
    )

    assert (
        evaluation.packet_arrival_rate_per_s
        == 20.0
    )

    assert (
        evaluation.packet_size_bits
        == 4_000_000.0
    )

    assert (
        evaluation.expected_packets_per_tti
        == 0.01
    )


def test_ftp_arrivals_update_scheduler_buffer():
    device = preferred_device()

    manager = TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=1500,
            packet_arrival_rate_per_s=500.0,
            tti_duration_s=0.001,
        ),
        full_buffer_state_bits=1.0e6,
    )

    start = manager.begin_tti(
        tti_index=0,
        packet_arrivals=torch.tensor(
            [
                0,
                2,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    torch.testing.assert_close(
        start.arrived_bits,
        torch.tensor(
            [
                0.0,
                24000.0,
                12000.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    torch.testing.assert_close(
        start.buffer_for_scheduler_bits,
        torch.tensor(
            [
                1.0e6,
                24000.0,
                12000.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )


def test_ftp_service_is_limited_by_queue():
    device = preferred_device()

    manager = TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=1500,
            packet_arrival_rate_per_s=500.0,
            tti_duration_s=0.001,
        ),
        full_buffer_state_bits=1.0e6,
    )

    manager.begin_tti(
        tti_index=0,
        packet_arrivals=torch.tensor(
            [
                0,
                2,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    service = manager.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [
                    5.0e6,
                    30.0e6,
                    6.0e6,
                ],
                dtype=torch.float32,
                device=device,
            )
        )
    )

    #
    # 1 ms TTI:
    #
    # FB:
    #     capacity = 5000 bits
    #     delivered = 5000
    #
    # FTP UE 1:
    #     capacity = 30000
    #     queue    = 24000
    #     delivered = 24000
    #
    # FTP UE 2:
    #     capacity = 6000
    #     queue    = 12000
    #     delivered = 6000
    #

    torch.testing.assert_close(
        service.delivered_bits,
        torch.tensor(
            [
                5000.0,
                24000.0,
                6000.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    torch.testing.assert_close(
        service.delivered_rate_bps,
        torch.tensor(
            [
                5.0e6,
                24.0e6,
                6.0e6,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    torch.testing.assert_close(
        service.buffer_after_service_bits,
        torch.tensor(
            [
                1.0e6,
                0.0,
                6000.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    torch.testing.assert_close(
        service.unused_service_capacity_bits,
        torch.tensor(
            [
                0.0,
                6000.0,
                0.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )


def test_full_buffer_service_is_not_capped_by_state_proxy():
    device = preferred_device()

    manager = TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                True,
            ],
            dtype=torch.bool,
            device=device,
        ),
        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=1500,
            packet_arrival_rate_per_s=500.0,
            tti_duration_s=0.001,
        ),
        full_buffer_state_bits=1000.0,
    )

    manager.begin_tti(
        tti_index=0,
        packet_arrivals=torch.tensor(
            [
                0,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    result = manager.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [
                    100.0e6,
                ],
                dtype=torch.float32,
                device=device,
            )
        )
    )

    #
    # 100 Mbps * 1 ms = 100,000 bits.
    #
    # This is much larger than the 1000-bit
    # scheduler-state proxy.
    #
    torch.testing.assert_close(
        result.delivered_bits,
        torch.tensor(
            [
                100000.0,
            ],
            device=device,
        ),
    )

    torch.testing.assert_close(
        result.buffer_after_service_bits,
        torch.tensor(
            [
                1000.0,
            ],
            device=device,
        ),
    )


def test_ftp_backlog_carries_to_next_tti():
    device = preferred_device()

    manager = TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=1000,
            packet_arrival_rate_per_s=100.0,
            tti_duration_s=0.001,
        ),
        full_buffer_state_bits=1.0,
    )

    manager.begin_tti(
        tti_index=0,
        packet_arrivals=torch.tensor(
            [
                2,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    #
    # Queue:
    #     2 * 1000 * 8 = 16000 bits
    #
    manager.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [
                    6.0e6,
                ],
                device=device,
            )
        )
    )

    torch.testing.assert_close(
        manager.current_buffer_bits,
        torch.tensor(
            [
                10000.0,
            ],
            device=device,
        ),
    )

    start_1 = manager.begin_tti(
        tti_index=1,
        packet_arrivals=torch.tensor(
            [
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    #
    # old backlog = 10000
    # new packet   = 8000
    #
    torch.testing.assert_close(
        start_1.buffer_for_scheduler_bits,
        torch.tensor(
            [
                18000.0,
            ],
            device=device,
        ),
    )


def test_had_data_to_receive_marks_active_ftp_ues():
    device = preferred_device()

    manager = TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=1000,
            packet_arrival_rate_per_s=100.0,
            tti_duration_s=0.001,
        ),
        full_buffer_state_bits=1000.0,
    )

    manager.begin_tti(
        tti_index=0,
        packet_arrivals=torch.tensor(
            [
                0,
                1,
                0,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    result = manager.apply_service(
        offered_service_capacity_bps=(
            torch.zeros(
                3,
                dtype=torch.float32,
                device=device,
            )
        )
    )

    torch.testing.assert_close(
        result.had_data_to_receive,
        torch.tensor(
            [
                True,
                True,
                False,
            ],
            dtype=torch.bool,
            device=device,
        ),
    )


def test_ftp3_poisson_arrivals_have_expected_mean():
    device = preferred_device()

    num_ues = 20_000

    manager = TrafficBufferManager(
        full_buffer_mask=torch.zeros(
            num_ues,
            dtype=torch.bool,
            device=device,
        ),
        ftp3_config=(
            build_training_ftp3_config(
                tti_duration_s=0.0005,
            )
        ),
        full_buffer_state_bits=1.0,
        seed=1234,
    )

    arrivals = (
        manager.sample_packet_arrivals()
    )

    assert arrivals.dtype == torch.long

    assert arrivals.device == device

    assert torch.all(
        arrivals >= 0
    )

    empirical_mean = (
        arrivals
        .to(
            torch.float32
        )
        .mean()
    )

    #
    # Expected:
    #     500 * 0.0005 = 0.25
    #
    torch.testing.assert_close(
        empirical_mean,
        torch.tensor(
            0.25,
            dtype=torch.float32,
            device=device,
        ),
        atol=0.02,
        rtol=0.0,
    )


