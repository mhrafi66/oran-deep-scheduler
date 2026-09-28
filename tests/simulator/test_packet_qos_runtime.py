from pathlib import Path

import torch

from oran_scheduler.simulator.packet_qos_runtime import (
    PacketQoSTrafficBufferManager,
)
from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
)


def test_packet_shadow_matches_aggregate_queue(
    tmp_path: Path,
) -> None:

    manager = PacketQoSTrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                False,
                False,
            ]
        ),

        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=100,

            packet_arrival_rate_per_s=0.0,

            tti_duration_s=0.001,
        ),

        full_buffer_state_bits=1000.0,

        seed=1,

        packet_qos_enabled=True,

        packet_qos_csv_path=(
            tmp_path / "qos.csv"
        ),

        packet_qos_cell_index=0,

        packet_qos_deadline_ttis=2,
    )

    start = manager.begin_tti(
        tti_index=0,

        packet_arrivals=torch.tensor(
            [
                2,
                1,
            ],
            dtype=torch.long,
        ),
    )

    assert int(
        start.packet_arrivals.sum().item()
    ) == 3

    result = manager.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [
                    800_000.0,
                    400_000.0,
                ]
            )
        )
    )

    assert torch.all(
        result.buffer_after_service_bits
        >= 0.0
    )

    summary = (
        manager
        .last_packet_qos_summary
    )

    assert summary is not None

    assert summary.completed_packets == 1

    assert summary.mean_completion_delay_ttis == 1.0

    assert (
        tmp_path / "qos.csv"
    ).exists()


def test_packet_deadline_miss_is_counted_once(
    tmp_path: Path,
) -> None:

    manager = PacketQoSTrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [
                False,
            ]
        ),

        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=100,

            packet_arrival_rate_per_s=0.0,

            tti_duration_s=0.001,
        ),

        full_buffer_state_bits=1000.0,

        packet_qos_enabled=True,

        packet_qos_csv_path=(
            tmp_path / "qos.csv"
        ),

        packet_qos_deadline_ttis=1,
    )

    manager.begin_tti(
        tti_index=0,

        packet_arrivals=torch.tensor(
            [1],
            dtype=torch.long,
        ),
    )

    manager.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [0.0]
            )
        )
    )

    manager.begin_tti(
        tti_index=1,

        packet_arrivals=torch.tensor(
            [0],
            dtype=torch.long,
        ),
    )

    manager.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [0.0]
            )
        )
    )

    summary = (
        manager
        .last_packet_qos_summary
    )

    assert summary is not None

    assert summary.new_deadline_misses == 1

    assert summary.cumulative_deadline_misses == 1
