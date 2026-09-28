import pytest
import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)

from oran_scheduler.simulator.dynamic_handover_coordinator import (
    DynamicHandoverCoordinator,
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

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
)


FULL_BUFFER_BITS = 8000.0


def _ftp_config():
    return FTP3TrafficConfig(
        packet_size_bytes=100,

        packet_arrival_rate_per_s=4000.0,

        tti_duration_s=0.0005,
    )


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


def _coordinator():
    registry = _registry()

    controller = HandoverController(
        initial_serving_bs=(
            registry
            .snapshot()
            .serving_bs
        ),

        num_bs=2,

        config=HandoverConfig(
            hysteresis_db=3.0,
            time_to_trigger_ttis=1,
        ),
    )

    arrivals = GlobalUETrafficArrivalProcess(
        global_ue_indices=(
            registry
            .snapshot()
            .global_ue_indices
        ),

        full_buffer_mask=(
            registry
            .snapshot()
            .full_buffer_mask
        ),

        ftp3_config=_ftp_config(),

        seed=12345,
    )

    coordinator = DynamicHandoverCoordinator(
        registry=registry,

        handover_controller=controller,

        selected_cell_indices=(
            0,
            1,
        ),

        tds_config=PFTimeDomainConfig(
            num_candidates=2,
        ),

        throughput_forgetting_factor=0.9,

        ftp3_config=_ftp_config(),

        full_buffer_state_bits=(
            FULL_BUFFER_BITS
        ),

        traffic_arrival_process=arrivals,
    )

    return (
        registry,
        coordinator,
    )


def test_materialize_selected_cells():
    (
        _,
        coordinator,
    ) = _coordinator()

    states = (
        coordinator
        .materialize_selected_cells()
    )

    assert len(states) == 2

    torch.testing.assert_close(
        states[
            0
        ]
        .global_ue_indices,

        torch.tensor(
            [
                10,
                20,
            ],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        states[
            1
        ]
        .global_ue_indices,

        torch.tensor(
            [
                30,
                40,
            ],
            dtype=torch.long,
        ),
    )


def test_handover_preserves_latest_local_pf_and_queue():
    (
        registry,
        coordinator,
    ) = _coordinator()

    local_states = (
        coordinator
        .materialize_selected_cells()
    )

    #
    # Make UE 20 carry state that exists only in
    # current CELL-LOCAL managers.
    #
    local_states[
        0
    ].state_manager.reset_average_throughput_bps(
        torch.tensor(
            [
                11.0e6,
                22.0e6,
            ],
            dtype=torch.float32,
        )
    )

    local_states[
        0
    ].traffic_manager.reset_ftp_buffers(
        torch.tensor(
            [
                0.0,
                7777.0,
            ],
            dtype=torch.float32,
        )
    )

    #
    # UE 20 moves from BS 0 -> BS 1.
    #
    transition = (
        coordinator
        .advance_handover(
            tti_index=0,

            link_power=torch.tensor(
                [
                    # UE 10 stays at BS 0.
                    [
                        5.0,
                        1.0,
                    ],

                    # UE 20 strongly prefers BS 1.
                    [
                        1.0,
                        5.0,
                    ],

                    # UE 30 stays at BS 1.
                    [
                        1.0,
                        5.0,
                    ],

                    # UE 40 stays at BS 1.
                    [
                        1.0,
                        5.0,
                    ],
                ],
                dtype=torch.float32,
            ),

            local_states=local_states,
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

    assert (
        transition
        .affected_cell_indices
        == (
            0,
            1,
        )
    )

    #
    # Cell 0 lost UE 20.
    #
    torch.testing.assert_close(
        transition
        .local_states[
            0
        ]
        .global_ue_indices,

        torch.tensor(
            [
                10,
            ],
            dtype=torch.long,
        ),
    )

    #
    # Cell 1 gained UE 20.
    #
    torch.testing.assert_close(
        transition
        .local_states[
            1
        ]
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

    destination = (
        transition
        .local_states[
            1
        ]
    )

    #
    # UE 20 is first in deterministic global
    # registry order.
    #
    assert (
        destination
        .state_manager
        .current_average_throughput_bps[
            0
        ]
        .item()
        == pytest.approx(
            22.0e6
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
            7777.0
        )
    )

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


def test_global_arrival_trace_follows_ue_after_handover():
    (
        _,
        coordinator,
    ) = _coordinator()

    before = (
        coordinator
        .materialize_selected_cells()
    )

    #
    # Materialize global traffic TTI 0 while UE 20
    # is still in stream 0.
    #
    source_arrivals = (
        coordinator
        .packet_arrivals_for_stream(
            tti_index=0,

            stream_index=0,

            local_states=before,
        )
    )

    ue20_arrival_before = int(
        source_arrivals[
            1
        ].item()
    )

    transition = (
        coordinator
        .advance_handover(
            tti_index=0,

            link_power=torch.tensor(
                [
                    [
                        5.0,
                        1.0,
                    ],
                    [
                        1.0,
                        5.0,
                    ],
                    [
                        1.0,
                        5.0,
                    ],
                    [
                        1.0,
                        5.0,
                    ],
                ],
                dtype=torch.float32,
            ),

            local_states=before,
        )
    )

    #
    # Same global TTI realization, now UE 20 lives
    # in destination stream 1.
    #
    destination_arrivals = (
        coordinator
        .packet_arrivals_for_stream(
            tti_index=0,

            stream_index=1,

            local_states=(
                transition
                .local_states
            ),
        )
    )

    ue20_arrival_after = int(
        destination_arrivals[
            0
        ].item()
    )

    assert (
        ue20_arrival_after
        == ue20_arrival_before
    )


def test_no_handover_preserves_membership():
    (
        _,
        coordinator,
    ) = _coordinator()

    before = (
        coordinator
        .materialize_selected_cells()
    )

    transition = (
        coordinator
        .advance_handover(
            tti_index=0,

            link_power=torch.tensor(
                [
                    [
                        5.0,
                        1.0,
                    ],
                    [
                        5.0,
                        1.0,
                    ],
                    [
                        1.0,
                        5.0,
                    ],
                    [
                        1.0,
                        5.0,
                    ],
                ],
                dtype=torch.float32,
            ),

            local_states=before,
        )
    )

    assert (
        transition
        .moved_global_ue_indices
        .numel()
        == 0
    )

    assert (
        transition
        .affected_cell_indices
        == tuple()
    )

    torch.testing.assert_close(
        transition
        .local_states[
            0
        ]
        .global_ue_indices,

        torch.tensor(
            [
                10,
                20,
            ],
            dtype=torch.long,
        ),
    )


def test_stale_local_membership_is_rejected():
    (
        registry,
        coordinator,
    ) = _coordinator()

    stale = (
        coordinator
        .materialize_selected_cells()
    )

    #
    # Artificially modify global association without
    # rematerializing local state.
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

    with pytest.raises(
        ValueError,
        match="stale",
    ):
        coordinator.synchronize_selected_cells(
            stale
        )


def test_controller_registry_mismatch_is_rejected():
    registry = _registry()

    wrong_controller = HandoverController(
        initial_serving_bs=torch.tensor(
            [
                1,
                0,
                1,
                1,
            ],
            dtype=torch.long,
        ),

        num_bs=2,

        config=HandoverConfig(
            time_to_trigger_ttis=1,
        ),
    )

    arrivals = GlobalUETrafficArrivalProcess(
        global_ue_indices=(
            registry
            .snapshot()
            .global_ue_indices
        ),

        full_buffer_mask=(
            registry
            .snapshot()
            .full_buffer_mask
        ),

        ftp3_config=_ftp_config(),

        seed=1,
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        DynamicHandoverCoordinator(
            registry=registry,

            handover_controller=(
                wrong_controller
            ),

            selected_cell_indices=(
                0,
                1,
            ),

            tds_config=PFTimeDomainConfig(
                num_candidates=2,
            ),

            throughput_forgetting_factor=0.9,

            ftp3_config=_ftp_config(),

            full_buffer_state_bits=(
                FULL_BUFFER_BITS
            ),

            traffic_arrival_process=(
                arrivals
            ),
        )
