from types import SimpleNamespace

import torch

from oran_scheduler.rl.control_execution import (
    CandidateGatedDelayedPPOExecutionController,
)
from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.schedulers.one_lds_loop import (
    OneLDSScheduleResult,
)


def _preparation(
    ids,
):
    ids = torch.tensor(
        ids,
        dtype=torch.long,
    )

    return SimpleNamespace(
        prepared=SimpleNamespace(
            candidate_global_ue_indices=ids,

            candidate_valid_mask=torch.ones(
                ids.shape,
                dtype=torch.bool,
            ),
        )
    )


def _schedule(
    candidate_indices,
):
    tensor = torch.tensor(
        [
            candidate_indices,
        ],
        dtype=torch.long,
    )

    return OneLDSScheduleResult(
        allocation=CellAllocation(
            candidate_by_user_slot=(
                tensor
            )
        ),

        decisions=(),

        actions=tensor.clone(),
    )


def test_delayed_execution_tracks_global_identity() -> None:

    controller = (
        CandidateGatedDelayedPPOExecutionController(
            delay_ttis=1,
            num_user_slots=1,
            num_rbgs=2,
            device="cpu",
        )
    )

    #
    # TTI 0:
    # candidate slot 0 = UE 100
    #
    out0 = controller(
        0,
        _schedule(
            [0, 0]
        ),
        _preparation(
            [100, 200]
        ),
    )

    assert torch.all(
        out0
        .allocation
        .candidate_by_user_slot
        == -1
    )

    #
    # TTI 1:
    # UE 100 moved to candidate slot 1.
    #
    out1 = controller(
        1,
        _schedule(
            [0, 0]
        ),
        _preparation(
            [300, 100]
        ),
    )

    torch.testing.assert_close(
        out1
        .allocation
        .candidate_by_user_slot,

        torch.tensor(
            [
                [1, 1]
            ],
            dtype=torch.long,
        ),
    )


def test_missing_delayed_ue_becomes_no_allocation() -> None:

    controller = (
        CandidateGatedDelayedPPOExecutionController(
            delay_ttis=1,
            num_user_slots=1,
            num_rbgs=1,
            device="cpu",
        )
    )

    controller(
        0,
        _schedule(
            [0]
        ),
        _preparation(
            [100, 200]
        ),
    )

    output = controller(
        1,
        _schedule(
            [0]
        ),
        _preparation(
            [300, 400]
        ),
    )

    assert int(
        output
        .allocation
        .candidate_by_user_slot[
            0,
            0,
        ].item()
    ) == -1
