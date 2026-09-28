import pytest
import torch

from oran_scheduler.simulator.global_traffic_arrivals import (
    GlobalUETrafficArrivalProcess,
)

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
)


def _process(
    *,
    seed=1234,
):
    return GlobalUETrafficArrivalProcess(
        global_ue_indices=torch.tensor(
            [
                10,
                20,
                30,
                40,
            ],
            dtype=torch.long,
        ),

        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                False,
                True,
            ],
            dtype=torch.bool,
        ),

        ftp3_config=FTP3TrafficConfig(
            packet_size_bytes=1500,

            #
            # Large rate only to make the tests'
            # random vectors visibly nontrivial.
            #
            packet_arrival_rate_per_s=4000.0,

            tti_duration_s=0.0005,
        ),

        seed=seed,
    )


def test_same_tti_is_idempotent():
    process = _process()

    first = process.snapshot_for_tti(
        0
    )

    second = process.snapshot_for_tti(
        0
    )

    torch.testing.assert_close(
        first.packet_arrivals,
        second.packet_arrivals,
    )


def test_full_buffer_ues_have_zero_arrivals():
    process = _process()

    snapshot = process.snapshot_for_tti(
        0
    )

    assert (
        snapshot
        .packet_arrivals[
            0
        ]
        .item()
        == 0
    )

    assert (
        snapshot
        .packet_arrivals[
            3
        ]
        .item()
        == 0
    )


def test_local_ordering_is_identity_based():
    process = _process()

    global_snapshot = (
        process.snapshot_for_tti(
            0
        )
    )

    gathered = process.arrivals_for(
        tti_index=0,

        global_ue_indices=torch.tensor(
            [
                30,
                10,
                20,
            ],
            dtype=torch.long,
        ),
    )

    expected = torch.stack(
        [
            global_snapshot
            .packet_arrivals[
                2
            ],

            global_snapshot
            .packet_arrivals[
                0
            ],

            global_snapshot
            .packet_arrivals[
                1
            ],
        ]
    )

    torch.testing.assert_close(
        gathered,
        expected,
    )


def test_handover_regrouping_does_not_resample_ue():
    process = _process()

    #
    # TTI 0:
    #
    # UE 20 belongs to the first local cell view.
    #
    cell0_before = process.arrivals_for(
        tti_index=0,

        global_ue_indices=torch.tensor(
            [
                10,
                20,
            ],
            dtype=torch.long,
        ),
    )

    #
    # Same TTI, imagine association changes before a
    # different consumer asks for the realization.
    #
    # UE 20 is now in a different local ordering.
    #
    cell1_after = process.arrivals_for(
        tti_index=0,

        global_ue_indices=torch.tensor(
            [
                20,
                30,
                40,
            ],
            dtype=torch.long,
        ),
    )

    assert (
        cell0_before[
            1
        ].item()
        ==
        cell1_after[
            0
        ].item()
    )


def test_same_seed_reproduces_global_trace():
    first = _process(
        seed=777
    )

    second = _process(
        seed=777
    )

    for tti_index in range(
        5
    ):
        torch.testing.assert_close(
            first
            .snapshot_for_tti(
                tti_index
            )
            .packet_arrivals,

            second
            .snapshot_for_tti(
                tti_index
            )
            .packet_arrivals,
        )


def test_skipped_tti_is_rejected():
    process = _process()

    process.snapshot_for_tti(
        0
    )

    with pytest.raises(
        ValueError,
        match="consecutively",
    ):
        process.snapshot_for_tti(
            2
        )


def test_unknown_global_ue_is_rejected():
    process = _process()

    with pytest.raises(
        KeyError,
        match="Unknown global UE",
    ):
        process.arrivals_for(
            tti_index=0,

            global_ue_indices=torch.tensor(
                [
                    999,
                ],
                dtype=torch.long,
            ),
        )
