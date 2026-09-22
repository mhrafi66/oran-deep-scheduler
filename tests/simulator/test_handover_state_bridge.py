import pytest
import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)

from oran_scheduler.simulator.global_ue_state import (
    GlobalUESchedulerStateRegistry,
)

from oran_scheduler.simulator.handover_state_bridge import (
    materialize_cell_local_scheduler_state,
    synchronize_cell_local_scheduler_state,
)

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
)


FULL_BUFFER_BITS = 8000.0


def _registry():
    return GlobalUESchedulerStateRegistry(
        global_ue_indices=torch.tensor(
            [
                10,
                20,
                30,
                40,
            ],
            dtype=torch.long,
        ),

        serving_bs=torch.tensor(
            [
                0,
                0,
                1,
                1,
            ],
            dtype=torch.long,
        ),

        average_throughput_bps=torch.tensor(
            [
                1.0e6,
                2.0e6,
                3.0e6,
                4.0e6,
            ],
            dtype=torch.float32,
        ),

        buffer_bits=torch.tensor(
            [
                FULL_BUFFER_BITS,
                1200.0,
                FULL_BUFFER_BITS,
                3400.0,
            ],
            dtype=torch.float32,
        ),

        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                True,
                False,
            ],
            dtype=torch.bool,
        ),
    )


def _ftp_config():
    return FTP3TrafficConfig(
        packet_size_bytes=100,
        packet_arrival_rate_per_s=0.0,
        tti_duration_s=0.0005,
    )


def _materialize(
    registry,
    cell_index,
):
    return materialize_cell_local_scheduler_state(
        registry=registry,

        cell_index=cell_index,

        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),

        throughput_forgetting_factor=0.9,

        ftp3_config=_ftp_config(),

        full_buffer_state_bits=(
            FULL_BUFFER_BITS
        ),

        traffic_seed=1234 + cell_index,
    )


def test_materialization_preserves_identity_order():
    registry = _registry()

    local = _materialize(
        registry,
        0,
    )

    torch.testing.assert_close(
        local.global_ue_indices,
        torch.tensor(
            [
                10,
                20,
            ],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        local
        .state_manager
        .current_average_throughput_bps,
        torch.tensor(
            [
                1.0e6,
                2.0e6,
            ],
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        local
        .traffic_manager
        .current_buffer_bits,
        torch.tensor(
            [
                FULL_BUFFER_BITS,
                1200.0,
            ],
            dtype=torch.float32,
        ),
    )


def test_local_updates_scatter_back_by_global_identity():
    registry = _registry()

    local = _materialize(
        registry,
        0,
    )

    local.state_manager.reset_average_throughput_bps(
        torch.tensor(
            [
                11.0e6,
                22.0e6,
            ],
            dtype=torch.float32,
        )
    )

    local.traffic_manager.reset_ftp_buffers(
        torch.tensor(
            [
                0.0,
                777.0,
            ],
            dtype=torch.float32,
        )
    )

    synchronize_cell_local_scheduler_state(
        local_state=local,
        registry=registry,
    )

    state = registry.gather(
        torch.tensor(
            [
                20,
                10,
            ],
            dtype=torch.long,
        )
    )

    torch.testing.assert_close(
        state.average_throughput_bps,
        torch.tensor(
            [
                22.0e6,
                11.0e6,
            ],
            dtype=torch.float32,
        ),
    )

    torch.testing.assert_close(
        state.buffer_bits,
        torch.tensor(
            [
                777.0,
                FULL_BUFFER_BITS,
            ],
            dtype=torch.float32,
        ),
    )


def test_handover_then_rematerialization_preserves_state():
    registry = _registry()

    #
    # Simulate cell-0 state evolution first.
    #
    source = _materialize(
        registry,
        0,
    )

    source.state_manager.reset_average_throughput_bps(
        torch.tensor(
            [
                10.0e6,
                20.0e6,
            ],
            dtype=torch.float32,
        )
    )

    source.traffic_manager.reset_ftp_buffers(
        torch.tensor(
            [
                0.0,
                5000.0,
            ],
            dtype=torch.float32,
        )
    )

    synchronize_cell_local_scheduler_state(
        local_state=source,
        registry=registry,
    )

    #
    # UE 20 hands over:
    #
    #     cell 0 -> cell 1
    #
    registry.apply_serving_bs_update(
        new_serving_bs=torch.tensor(
            [
                0,
                1,
                1,
                1,
            ],
            dtype=torch.long,
        ),

        handover_mask=torch.tensor(
            [
                False,
                True,
                False,
                False,
            ],
            dtype=torch.bool,
        ),
    )

    destination = _materialize(
        registry,
        1,
    )

    #
    # Registry ordering is deterministic:
    # 20 joined pre-existing 30,40.
    #
    torch.testing.assert_close(
        destination.global_ue_indices,
        torch.tensor(
            [
                20,
                30,
                40,
            ],
            dtype=torch.long,
        ),
    )

    #
    # UE 20 retains PF history and FTP backlog.
    #
    assert (
        destination
        .state_manager
        .current_average_throughput_bps[
            0
        ]
        .item()
        == pytest.approx(
            20.0e6
        )
    )

    assert (
        destination
        .traffic_manager
        .current_buffer_bits[
            0
        ]
        .item()
        == pytest.approx(
            5000.0
        )
    )


def test_full_buffer_proxy_survives_rematerialization():
    registry = _registry()

    local = _materialize(
        registry,
        0,
    )

    assert (
        local
        .traffic_manager
        .current_buffer_bits[
            0
        ]
        .item()
        == pytest.approx(
            FULL_BUFFER_BITS
        )
    )


def test_empty_cell_is_explicitly_rejected_for_now():
    registry = _registry()

    with pytest.raises(
        ValueError,
        match="empty cell",
    ):
        _materialize(
            registry,
            99,
        )


def test_wrong_full_buffer_proxy_is_rejected():
    registry = _registry()

    with pytest.raises(
        ValueError,
        match="Full-Buffer proxy",
    ):
        materialize_cell_local_scheduler_state(
            registry=registry,

            cell_index=0,

            tds_config=PFTimeDomainConfig(
                num_candidates=2,
            ),

            throughput_forgetting_factor=0.9,

            ftp3_config=_ftp_config(),

            full_buffer_state_bits=9999.0,
        )
