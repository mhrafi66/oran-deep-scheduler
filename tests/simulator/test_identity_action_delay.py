import torch

from oran_scheduler.simulator.identity_action_delay import (
    GlobalUEAllocation,
    IdentitySafeActionDelay,
    NO_GLOBAL_UE,
    remap_global_allocation_to_candidates,
)


def _allocation(
    value: int,
) -> GlobalUEAllocation:

    return GlobalUEAllocation(
        global_ue_by_user_slot=(
            torch.tensor(
                [
                    [value, value],
                ],
                dtype=torch.long,
            )
        )
    )


def test_identity_safe_delay_fifo() -> None:

    delay = IdentitySafeActionDelay(
        delay_ttis=2,
        num_user_slots=1,
        num_rbgs=2,
        device="cpu",
    )

    out0 = delay.push(
        _allocation(10)
    )

    out1 = delay.push(
        _allocation(20)
    )

    out2 = delay.push(
        _allocation(30)
    )

    assert torch.all(
        out0.global_ue_by_user_slot
        == NO_GLOBAL_UE
    )

    assert torch.all(
        out1.global_ue_by_user_slot
        == NO_GLOBAL_UE
    )

    assert torch.all(
        out2.global_ue_by_user_slot
        == 10
    )


def test_remap_preserves_identity_not_slot() -> None:

    delayed = GlobalUEAllocation(
        global_ue_by_user_slot=(
            torch.tensor(
                [
                    [100, 200, 300],
                ],
                dtype=torch.long,
            )
        )
    )

    #
    # Candidate ordering changed.
    #
    candidates = torch.tensor(
        [
            300,
            100,
            999,
        ],
        dtype=torch.long,
    )

    valid = torch.tensor(
        [
            True,
            True,
            True,
        ]
    )

    action = (
        remap_global_allocation_to_candidates(
            allocation=delayed,

            candidate_global_ue_indices=(
                candidates
            ),

            candidate_valid_mask=valid,

            no_allocation_action=3,
        )
    )

    #
    # UE 100 -> current candidate 1
    # UE 200 -> gone -> no allocation 3
    # UE 300 -> current candidate 0
    #
    expected = torch.tensor(
        [
            [1, 3, 0],
        ],
        dtype=torch.long,
    )

    torch.testing.assert_close(
        action,
        expected,
    )
