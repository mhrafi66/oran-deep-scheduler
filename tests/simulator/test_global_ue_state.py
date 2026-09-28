import pytest
import torch

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
                8000.0,
                1200.0,
                8000.0,
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


def test_gather_uses_requested_global_identity_order():
    registry = _registry()

    view = registry.gather(
        torch.tensor(
            [
                40,
                10,
                20,
            ],
            dtype=torch.long,
        )
    )

    torch.testing.assert_close(
        view.global_ue_indices,
        torch.tensor(
            [
                40,
                10,
                20,
            ],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        view.average_throughput_bps,
        torch.tensor(
            [
                4.0e6,
                1.0e6,
                2.0e6,
            ],
        ),
    )

    torch.testing.assert_close(
        view.buffer_bits,
        torch.tensor(
            [
                3400.0,
                8000.0,
                1200.0,
            ],
        ),
    )


def test_handover_preserves_pf_and_queue_state():
    registry = _registry()

    before = registry.snapshot()

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

    after = registry.snapshot()

    assert after.serving_bs[1].item() == 1

    torch.testing.assert_close(
        after.average_throughput_bps,
        before.average_throughput_bps,
    )

    torch.testing.assert_close(
        after.buffer_bits,
        before.buffer_bits,
    )

    torch.testing.assert_close(
        after.full_buffer_mask,
        before.full_buffer_mask,
    )


def test_cell_membership_changes_after_handover():
    registry = _registry()

    torch.testing.assert_close(
        registry.global_ues_for_cell(
            0
        ),
        torch.tensor(
            [
                10,
                20,
            ],
            dtype=torch.long,
        ),
    )

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

    torch.testing.assert_close(
        registry.global_ues_for_cell(
            0
        ),
        torch.tensor(
            [
                10,
            ],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        registry.global_ues_for_cell(
            1
        ),
        torch.tensor(
            [
                20,
                30,
                40,
            ],
            dtype=torch.long,
        ),
    )


def test_pf_update_scatter_is_identity_safe():
    registry = _registry()

    registry.update_pf_history(
        global_ue_indices=torch.tensor(
            [
                40,
                20,
            ],
            dtype=torch.long,
        ),

        average_throughput_bps=torch.tensor(
            [
                44.0e6,
                22.0e6,
            ],
            dtype=torch.float32,
        ),
    )

    state = registry.gather(
        torch.tensor(
            [
                20,
                40,
            ],
            dtype=torch.long,
        )
    )

    torch.testing.assert_close(
        state.average_throughput_bps,
        torch.tensor(
            [
                22.0e6,
                44.0e6,
            ],
            dtype=torch.float32,
        ),
    )


def test_ftp_buffer_update_follows_identity():
    registry = _registry()

    registry.update_buffer_bits(
        global_ue_indices=torch.tensor(
            [
                40,
                20,
            ],
            dtype=torch.long,
        ),

        buffer_bits=torch.tensor(
            [
                9000.0,
                7000.0,
            ],
            dtype=torch.float32,
        ),
    )

    state = registry.gather(
        torch.tensor(
            [
                20,
                40,
            ],
            dtype=torch.long,
        )
    )

    torch.testing.assert_close(
        state.buffer_bits,
        torch.tensor(
            [
                7000.0,
                9000.0,
            ],
            dtype=torch.float32,
        ),
    )


def test_full_buffer_proxy_is_not_overwritten():
    registry = _registry()

    registry.update_buffer_bits(
        global_ue_indices=torch.tensor(
            [
                10,
                20,
            ],
            dtype=torch.long,
        ),

        buffer_bits=torch.tensor(
            [
                1.0,
                999.0,
            ],
            dtype=torch.float32,
        ),
    )

    state = registry.gather(
        torch.tensor(
            [
                10,
                20,
            ],
            dtype=torch.long,
        )
    )

    #
    # UE 10 is Full Buffer:
    # preserve scheduler proxy.
    #
    assert state.buffer_bits[0].item() == 8000.0

    #
    # UE 20 is FTP:
    # real queue update is accepted.
    #
    assert state.buffer_bits[1].item() == 999.0


def test_unknown_global_ue_is_rejected():
    registry = _registry()

    with pytest.raises(
        KeyError,
        match="Unknown global UE",
    ):
        registry.gather(
            torch.tensor(
                [
                    999,
                ],
                dtype=torch.long,
            )
        )


def test_duplicate_global_ue_identity_is_rejected():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        GlobalUESchedulerStateRegistry(
            global_ue_indices=torch.tensor(
                [
                    10,
                    10,
                ],
                dtype=torch.long,
            ),

            serving_bs=torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.long,
            ),

            average_throughput_bps=torch.ones(
                2
            ),

            buffer_bits=torch.zeros(
                2
            ),

            full_buffer_mask=torch.zeros(
                2,
                dtype=torch.bool,
            ),
        )
