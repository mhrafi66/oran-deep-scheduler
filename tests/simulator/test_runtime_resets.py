import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIStateManager,
)
from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
    TrafficBufferManager,
)


def test_pf_history_reset() -> None:

    manager = OneLDSCellTTIStateManager(
        initial_average_throughput_bps=torch.tensor(
            [1.0, 2.0, 3.0]
        ),
        serving_global_ue_indices=torch.tensor(
            [0, 1, -1],
            dtype=torch.long,
        ),
        serving_ue_valid_mask=torch.tensor(
            [True, True, False],
            dtype=torch.bool,
        ),
        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),
        throughput_forgetting_factor=0.9,
    )

    manager.reset_average_throughput_bps(
        5.0
    )

    torch.testing.assert_close(
        manager
        .current_average_throughput_bps,
        torch.tensor(
            [5.0, 5.0, 0.0]
        ),
    )


def test_ftp_queue_reset_preserves_full_buffer_proxy() -> None:

    manager = TrafficBufferManager(
        full_buffer_mask=torch.tensor(
            [True, False, False],
            dtype=torch.bool,
        ),
        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=100,
            packet_arrival_rate_per_s=10.0,
            tti_duration_s=0.001,
        ),
        full_buffer_state_bits=1000.0,
        initial_ftp_buffer_bits=torch.tensor(
            [0.0, 20.0, 30.0]
        ),
    )

    manager.reset_ftp_buffers(
        7.0
    )

    torch.testing.assert_close(
        manager.current_buffer_bits,
        torch.tensor(
            [1000.0, 7.0, 7.0]
        ),
    )
