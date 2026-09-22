import pytest
import torch

from oran_scheduler.simulator.packet_qos_handover import (
    GlobalUEPacketQoSRegistry,
    PacketQoSPacketState,
    PacketQoSUEState,
    export_packet_qos_ue_state,
    restore_packet_qos_ue_state,
)

from oran_scheduler.simulator.packet_qos_runtime import (
    PacketQoSTrafficBufferManager,
)

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
)


def _ftp_config():
    return FTP3TrafficConfig(
        packet_size_bytes=100,
        packet_arrival_rate_per_s=0.0,
        tti_duration_s=0.0005,
    )


def _manager(
    *,
    full_buffer_mask,
    deadline_ttis=2,
):
    return PacketQoSTrafficBufferManager(
        full_buffer_mask=(
            torch.tensor(
                full_buffer_mask,
                dtype=torch.bool,
            )
        ),

        ftp3_config=_ftp_config(),

        full_buffer_state_bits=8000.0,

        initial_ftp_buffer_bits=None,

        seed=123,

        packet_qos_enabled=True,

        packet_qos_deadline_ttis=(
            deadline_ttis
        ),
    )


def test_export_restore_preserves_fifo_and_partial_packet():
    source = _manager(
        full_buffer_mask=[
            False,
            False,
        ],
    )

    source.begin_tti(
        tti_index=0,

        packet_arrivals=torch.tensor(
            [
                2,
                1,
            ],
            dtype=torch.long,
        ),
    )

    #
    # 800000 bit/s * 0.0005 s
    # = 400 delivered bits.
    #
    # UE 0 started with:
    #
    #     [800, 800]
    #
    # and therefore ends with:
    #
    #     [400, 800]
    #
    source.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [
                    800000.0,
                    0.0,
                ],
                dtype=torch.float32,
            )
        )
    )

    exported = (
        export_packet_qos_ue_state(
            manager=source,
            ue_index=0,
        )
    )

    assert len(
        exported.packets
    ) == 2

    assert (
        exported
        .packets[
            0
        ]
        .remaining_bits
        == pytest.approx(
            400.0
        )
    )

    assert (
        exported
        .packets[
            1
        ]
        .remaining_bits
        == pytest.approx(
            800.0
        )
    )

    destination = _manager(
        full_buffer_mask=[
            False,
        ],
    )

    restore_packet_qos_ue_state(
        manager=destination,
        ue_index=0,
        state=exported,
    )

    restored = (
        export_packet_qos_ue_state(
            manager=destination,
            ue_index=0,
        )
    )

    assert restored == exported

    assert (
        destination
        .current_buffer_bits[
            0
        ]
        .item()
        == pytest.approx(
            1200.0
        )
    )


def test_deadline_missed_flag_survives_migration():
    source = _manager(
        full_buffer_mask=[
            False,
        ],

        deadline_ttis=1,
    )

    source.begin_tti(
        tti_index=0,

        packet_arrivals=torch.tensor(
            [1],
            dtype=torch.long,
        ),
    )

    source.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [0.0]
            )
        )
    )

    source.begin_tti(
        tti_index=1,

        packet_arrivals=torch.tensor(
            [0],
            dtype=torch.long,
        ),
    )

    source.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [0.0]
            )
        )
    )

    exported = (
        export_packet_qos_ue_state(
            manager=source,
            ue_index=0,
        )
    )

    assert len(
        exported.packets
    ) == 1

    assert (
        exported
        .packets[
            0
        ]
        .arrival_tti
        == 0
    )

    assert (
        exported
        .packets[
            0
        ]
        .deadline_missed
    )

    destination = _manager(
        full_buffer_mask=[
            False,
        ],

        deadline_ttis=1,
    )

    restore_packet_qos_ue_state(
        manager=destination,
        ue_index=0,
        state=exported,
    )

    restored = (
        export_packet_qos_ue_state(
            manager=destination,
            ue_index=0,
        )
    )

    assert restored == exported


def test_global_registry_restores_by_identity_not_slot():
    global_registry = (
        GlobalUEPacketQoSRegistry(
            global_ue_indices=(
                torch.tensor(
                    [
                        10,
                        20,
                        30,
                    ],
                    dtype=torch.long,
                )
            ),

            full_buffer_mask=(
                torch.tensor(
                    [
                        False,
                        False,
                        False,
                    ],
                    dtype=torch.bool,
                )
            ),
        )
    )

    source = _manager(
        full_buffer_mask=[
            False,
            False,
        ],
    )

    source.begin_tti(
        tti_index=5,

        packet_arrivals=torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.long,
        ),
    )

    source.apply_service(
        offered_service_capacity_bps=(
            torch.tensor(
                [
                    0.0,
                    0.0,
                ]
            )
        )
    )

    #
    # Source order:
    #
    #     slot 0 = UE 10
    #     slot 1 = UE 20
    #
    global_registry.synchronize_from_local(
        global_ue_indices=(
            torch.tensor(
                [
                    10,
                    20,
                ],
                dtype=torch.long,
            )
        ),

        manager=source,
    )

    destination = _manager(
        full_buffer_mask=[
            False,
            False,
        ],
    )

    #
    # Destination order after handover:
    #
    #     slot 0 = UE 20
    #     slot 1 = UE 30
    #
    global_registry.restore_to_local(
        global_ue_indices=(
            torch.tensor(
                [
                    20,
                    30,
                ],
                dtype=torch.long,
            )
        ),

        manager=destination,
    )

    ue20 = export_packet_qos_ue_state(
        manager=destination,
        ue_index=0,
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
        == 5
    )

    assert (
        destination
        .current_buffer_bits[
            0
        ]
        .item()
        == pytest.approx(
            800.0
        )
    )

    ue30 = export_packet_qos_ue_state(
        manager=destination,
        ue_index=1,
    )

    assert ue30.packets == tuple()


def test_full_buffer_cannot_receive_packet_fifo():
    manager = _manager(
        full_buffer_mask=[
            True,
        ],
    )

    state = PacketQoSUEState(
        packets=(
            PacketQoSPacketState(
                arrival_tti=0,
                remaining_bits=800.0,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="Full-Buffer",
    ):
        restore_packet_qos_ue_state(
            manager=manager,
            ue_index=0,
            state=state,
        )


def test_unknown_global_ue_is_rejected():
    registry = GlobalUEPacketQoSRegistry(
        global_ue_indices=torch.tensor(
            [
                10,
            ],
            dtype=torch.long,
        ),

        full_buffer_mask=torch.tensor(
            [
                False,
            ],
            dtype=torch.bool,
        ),
    )

    with pytest.raises(
        KeyError,
        match="Unknown global UE",
    ):
        registry.state_for(
            999
        )
