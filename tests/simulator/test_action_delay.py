import pytest
import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
)
from oran_scheduler.simulator.action_delay import (
    ActionExecutionDelayBuffer,
)


def _allocation(
    value: int,
) -> CellAllocation:

    return CellAllocation(
        candidate_by_user_slot=(
            torch.tensor(
                [
                    [
                        value,
                        value,
                    ],
                ],
                dtype=torch.long,
            )
        )
    )


def test_zero_delay_executes_current_action() -> None:

    delay = ActionExecutionDelayBuffer(
        delay_ttis=0
    )

    result = delay.push(
        tti_index=3,
        allocation=_allocation(
            2
        ),
    )

    assert result.source_tti == 3

    torch.testing.assert_close(
        result
        .allocation
        .candidate_by_user_slot,
        torch.tensor(
            [
                [
                    2,
                    2,
                ],
            ]
        ),
    )


def test_two_tti_delay_uses_empty_startup() -> None:

    delay = ActionExecutionDelayBuffer(
        delay_ttis=2
    )

    result0 = delay.push(
        tti_index=0,
        allocation=_allocation(
            0
        ),
    )

    result1 = delay.push(
        tti_index=1,
        allocation=_allocation(
            1
        ),
    )

    assert result0.source_tti is None
    assert result1.source_tti is None

    assert torch.all(
        result0
        .allocation
        .candidate_by_user_slot
        == -1
    )

    assert torch.all(
        result1
        .allocation
        .candidate_by_user_slot
        == -1
    )

    result2 = delay.push(
        tti_index=2,
        allocation=_allocation(
            2
        ),
    )

    assert result2.source_tti == 0

    torch.testing.assert_close(
        result2
        .allocation
        .candidate_by_user_slot,
        torch.tensor(
            [
                [
                    0,
                    0,
                ],
            ]
        ),
    )

    result3 = delay.push(
        tti_index=3,
        allocation=_allocation(
            3
        ),
    )

    assert result3.source_tti == 1

    torch.testing.assert_close(
        result3
        .allocation
        .candidate_by_user_slot,
        torch.tensor(
            [
                [
                    1,
                    1,
                ],
            ]
        ),
    )


def test_delay_copies_allocation() -> None:

    delay = ActionExecutionDelayBuffer(
        delay_ttis=1
    )

    source = _allocation(
        0
    )

    delay.push(
        tti_index=0,
        allocation=source,
    )

    source.candidate_by_user_slot[
        :,
        :,
    ] = 9

    result = delay.push(
        tti_index=1,
        allocation=_allocation(
            1
        ),
    )

    torch.testing.assert_close(
        result
        .allocation
        .candidate_by_user_slot,
        torch.tensor(
            [
                [
                    0,
                    0,
                ],
            ]
        ),
    )


def test_negative_delay_is_rejected() -> None:

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):

        ActionExecutionDelayBuffer(
            delay_ttis=-1
        )
