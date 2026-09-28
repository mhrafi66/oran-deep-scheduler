import pytest
import torch

from oran_scheduler.simulator.dynamic_serving_population import (
    RegistrySelectedCellMembershipProvider,
)

from oran_scheduler.simulator.global_ue_state import (
    GlobalUESchedulerStateRegistry,
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
                2,
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
                8000.0,
                1000.0,
                2000.0,
                3000.0,
            ],
            dtype=torch.float32,
        ),

        full_buffer_mask=torch.tensor(
            [
                True,
                False,
                False,
                False,
            ],
            dtype=torch.bool,
        ),
    )


def test_membership_tracks_handover():
    registry = _registry()

    provider = (
        RegistrySelectedCellMembershipProvider(
            registry=registry,

            selected_cell_indices=(
                0,
                1,
            ),
        )
    )

    torch.testing.assert_close(
        provider(
            0,
            0,
        ),

        torch.tensor(
            [
                10,
                20,
            ],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        provider(
            0,
            1,
        ),

        torch.tensor(
            [
                30,
            ],
            dtype=torch.long,
        ),
    )

    #
    # UE 20:
    #
    #     cell 0 -> cell 1
    #
    registry.apply_serving_bs_update(
        new_serving_bs=torch.tensor(
            [
                0,
                1,
                1,
                2,
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

    torch.testing.assert_close(
        provider(
            1,
            0,
        ),

        torch.tensor(
            [
                10,
            ],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        provider(
            1,
            1,
        ),

        torch.tensor(
            [
                20,
                30,
            ],
            dtype=torch.long,
        ),
    )


def test_bad_stream_is_rejected():
    provider = (
        RegistrySelectedCellMembershipProvider(
            registry=_registry(),

            selected_cell_indices=(
                0,
                1,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="stream_index",
    ):
        provider(
            0,
            2,
        )
