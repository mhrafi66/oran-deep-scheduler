import pytest
import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)

from oran_scheduler.simulator.dynamic_handover_coordinator import (
    DynamicHandoverCoordinator,
)

from oran_scheduler.simulator.dynamic_packet_qos import (
    PacketQoSDynamicHandoverCoordinator,
)

from oran_scheduler.simulator.global_traffic_arrivals import (
    GlobalUETrafficArrivalProcess,
)

from oran_scheduler.simulator.global_ue_state import (
    GlobalUESchedulerStateRegistry,
)

from oran_scheduler.simulator.handover import (
    HandoverConfig,
    HandoverController,
)

from oran_scheduler.simulator.packet_qos_handover import (
    GlobalUEPacketQoSRegistry,
    export_packet_qos_ue_state,
)

from oran_scheduler.simulator.packet_qos_runtime import (
    PacketQoSTrafficBufferManager,
)

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
)


FULL_BUFFER_BITS = 8000.0


def _ftp_config():

    return FTP3TrafficConfig(
        packet_size_bytes=100,

        packet_arrival_rate_per_s=0.0,

        tti_duration_s=0.0005,
    )


def _system(
    *,
    initial_ftp_bits: float = 0.0,
):

    global_ids = torch.tensor(
        [
            10,
            20,
            30,
            40,
        ],
        dtype=torch.long,
    )

    full_buffer_mask = torch.tensor(
        [
            True,
            False,
            True,
            False,
        ],
        dtype=torch.bool,
    )

    registry = (
        GlobalUESchedulerStateRegistry(
            global_ue_indices=(
                global_ids
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

            average_throughput_bps=(
                torch.tensor(
                    [
                        1.0e6,
                        2.0e6,
                        3.0e6,
                        4.0e6,
                    ],
                    dtype=torch.float32,
                )
            ),

            buffer_bits=torch.tensor(
                [
                    FULL_BUFFER_BITS,
                    initial_ftp_bits,
                    FULL_BUFFER_BITS,
                    initial_ftp_bits,
                ],
                dtype=torch.float32,
            ),

            full_buffer_mask=(
                full_buffer_mask
            ),
        )
    )

    handover_controller = (
        HandoverController(
            initial_serving_bs=(
                registry
                .snapshot()
                .serving_bs
            ),

            num_bs=2,

            config=(
                HandoverConfig(
                    hysteresis_db=3.0,

                    time_to_trigger_ttis=1,
                )
            ),
        )
    )

    traffic = (
        GlobalUETrafficArrivalProcess(
            global_ue_indices=(
                global_ids
            ),

            full_buffer_mask=(
                full_buffer_mask
            ),

            ftp3_config=(
                _ftp_config()
            ),

            seed=1234,
        )
    )

    base = DynamicHandoverCoordinator(
        registry=registry,

        handover_controller=(
            handover_controller
        ),

        selected_cell_indices=(
            0,
            1,
        ),

        tds_config=(
            PFTimeDomainConfig(
                num_candidates=2,
            )
        ),

        throughput_forgetting_factor=0.9,

        ftp3_config=_ftp_config(),

        full_buffer_state_bits=(
            FULL_BUFFER_BITS
        ),

        traffic_arrival_process=traffic,

        local_traffic_seed_base=9000,
    )

    packet_registry = (
        GlobalUEPacketQoSRegistry(
            global_ue_indices=(
                global_ids
            ),

            full_buffer_mask=(
                full_buffer_mask
            ),
        )
    )

    wrapped = (
        PacketQoSDynamicHandoverCoordinator(
            base_coordinator=base,

            packet_registry=(
                packet_registry
            ),

            packet_qos_deadline_ttis=2,
        )
    )

    return (
        registry,
        packet_registry,
        wrapped,
    )


def _handover_power():

    return torch.tensor(
        [
            # UE 10 stays on cell 0.
            [
                5.0,
                1.0,
            ],

            # UE 20 moves 0 -> 1.
            [
                1.0,
                5.0,
            ],

            # UE 30 stays on cell 1.
            [
                1.0,
                5.0,
            ],

            # UE 40 stays on cell 1.
            [
                1.0,
                5.0,
            ],
        ],
        dtype=torch.float32,
    )


def test_materialized_managers_are_packet_qos():

    (
        _,
        _,
        coordinator,
    ) = _system()

    states = (
        coordinator
        .materialize_selected_cells()
    )

    assert len(states) == 2

    assert all(
        isinstance(
            state.traffic_manager,
            PacketQoSTrafficBufferManager,
        )
        for state in states
    )


def test_packet_state_synchronizes_to_global_identity():

    (
        _,
        packet_registry,
        coordinator,
    ) = _system()

    states = (
        coordinator
        .materialize_selected_cells()
    )

    source_manager = (
        states[
            0
        ]
        .traffic_manager
    )

    #
    # Local source-cell order:
    #
    #     slot 0 -> UE 10, Full Buffer
    #     slot 1 -> UE 20, FTP
    #
    source_manager.begin_tti(
        tti_index=0,

        packet_arrivals=torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
        ),
    )

    source_manager.apply_service(
        offered_service_capacity_bps=(
            torch.zeros(
                2,
                dtype=torch.float32,
            )
        )
    )

    coordinator.synchronize_selected_cells(
        states
    )

    ue20 = (
        packet_registry
        .state_for(
            20
        )
    )

    assert len(
        ue20.packets
    ) == 1

    assert (
        ue20
        .packets[
            0
        ]
        .arrival_tti
        == 0
    )

    assert (
        ue20
        .packets[
            0
        ]
        .remaining_bits
        == pytest.approx(
            800.0
        )
    )


def test_packet_fifo_follows_ue_across_handover():

    (
        registry,
        packet_registry,
        coordinator,
    ) = _system()

    states = (
        coordinator
        .materialize_selected_cells()
    )

    source_manager = (
        states[
            0
        ]
        .traffic_manager
    )

    source_manager.begin_tti(
        tti_index=0,

        packet_arrivals=torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
        ),
    )

    source_manager.apply_service(
        offered_service_capacity_bps=(
            torch.zeros(
                2,
                dtype=torch.float32,
            )
        )
    )

    transition = (
        coordinator
        .advance_handover(
            tti_index=0,

            link_power=(
                _handover_power()
            ),

            local_states=states,
        )
    )

    torch.testing.assert_close(
        transition
        .moved_global_ue_indices,

        torch.tensor(
            [
                20,
            ],
            dtype=torch.long,
        ),
    )

    #
    # Destination deterministic global ordering:
    #
    #     UE 20, UE 30, UE 40
    #
    destination = (
        transition
        .local_states[
            1
        ]
    )

    torch.testing.assert_close(
        destination
        .global_ue_indices,

        torch.tensor(
            [
                20,
                30,
                40,
            ],
            dtype=torch.long,
        ),
    )

    destination_manager = (
        destination
        .traffic_manager
    )

    assert isinstance(
        destination_manager,
        PacketQoSTrafficBufferManager,
    )

    migrated = (
        export_packet_qos_ue_state(
            manager=(
                destination_manager
            ),

            ue_index=0,
        )
    )

    assert len(
        migrated.packets
    ) == 1

    assert (
        migrated
        .packets[
            0
        ]
        .arrival_tti
        == 0
    )

    assert (
        migrated
        .packets[
            0
        ]
        .remaining_bits
        == pytest.approx(
            800.0
        )
    )

    assert (
        destination_manager
        .current_buffer_bits[
            0
        ]
        .item()
        == pytest.approx(
            800.0
        )
    )

    globally_owned = (
        packet_registry
        .state_for(
            20
        )
    )

    assert globally_owned == migrated

    global_ue20 = registry.gather(
        torch.tensor(
            [
                20,
            ],
            dtype=torch.long,
        )
    )

    assert (
        global_ue20
        .serving_bs[
            0
        ]
        .item()
        == 1
    )


def test_unexplained_initial_aggregate_backlog_is_rejected():

    (
        _,
        _,
        coordinator,
    ) = _system(
        initial_ftp_bits=800.0,
    )

    #
    # Aggregate registry says FTP packets already
    # exist, but packet registry is empty.
    #
    # We deliberately refuse to fabricate an
    # arrival_tti for those unknown packets.
    #
    with pytest.raises(
        RuntimeError,
        match="packet-level global FIFO disagree",
    ):
        coordinator.materialize_selected_cells()
