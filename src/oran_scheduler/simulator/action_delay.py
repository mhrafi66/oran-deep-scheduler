from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    CellAllocation,
)


@dataclass(frozen=True)
class ActionDelayResult:
    """
    Result of one scheduler decision entering the
    execution-delay FIFO.

    generated_tti:
        TTI where this allocation was produced.

    executed_tti:
        Current TTI where an allocation is executed.

    source_tti:
        TTI where the executed allocation was generated.

        None means startup produced an empty allocation.

    allocation:
        Allocation executed by the PHY now.
    """

    generated_tti: int

    executed_tti: int

    source_tti: int | None

    allocation: CellAllocation


def clone_cell_allocation(
    allocation: CellAllocation,
) -> CellAllocation:

    return CellAllocation(
        candidate_by_user_slot=(
            allocation
            .candidate_by_user_slot
            .detach()
            .clone()
        )
    )


def empty_like_allocation(
    allocation: CellAllocation,
) -> CellAllocation:

    return CellAllocation(
        candidate_by_user_slot=torch.full_like(
            allocation
            .candidate_by_user_slot,
            fill_value=NO_ALLOCATION,
        )
    )


class ActionExecutionDelayBuffer:
    """
    Delay scheduler ACTION EXECUTION by D TTIs.

    This is distinct from CSI staleness.

    CSI staleness:
        old observation -> decision now -> execute now

    execution delay:
        current observation -> decision now
            -> execute decision D TTIs later

    Startup rule:
        Before D historical actions exist, execute an
        empty allocation.

    This conservative startup policy makes startup loss
    explicit rather than secretly reusing a future action.
    """

    def __init__(
        self,
        *,
        delay_ttis: int,
    ) -> None:

        if delay_ttis < 0:
            raise ValueError(
                "delay_ttis cannot be negative."
            )

        self.delay_ttis = (
            delay_ttis
        )

        self._queue: deque[
            tuple[
                int,
                CellAllocation,
            ]
        ] = deque()

        self._reference_shape: (
            tuple[int, int] | None
        ) = None

        self._reference_device: (
            torch.device | None
        ) = None


    def reset(
        self,
    ) -> None:

        self._queue.clear()

        self._reference_shape = None

        self._reference_device = None


    def push(
        self,
        *,
        tti_index: int,
        allocation: CellAllocation,
    ) -> ActionDelayResult:

        if tti_index < 0:
            raise ValueError(
                "tti_index cannot be negative."
            )

        shape = tuple(
            allocation
            .candidate_by_user_slot
            .shape
        )

        device = (
            allocation
            .candidate_by_user_slot
            .device
        )

        if self._reference_shape is None:

            self._reference_shape = (
                shape
            )

            self._reference_device = (
                device
            )

        else:

            if (
                shape
                != self._reference_shape
            ):
                raise ValueError(
                    "Action allocation shape changed "
                    "inside one delay stream."
                )

            if (
                device
                != self._reference_device
            ):
                raise ValueError(
                    "Action allocation device changed "
                    "inside one delay stream."
                )

        current = clone_cell_allocation(
            allocation
        )

        if self.delay_ttis == 0:

            return ActionDelayResult(
                generated_tti=tti_index,
                executed_tti=tti_index,
                source_tti=tti_index,
                allocation=current,
            )

        self._queue.append(
            (
                tti_index,
                current,
            )
        )

        if (
            len(
                self._queue
            )
            <= self.delay_ttis
        ):

            return ActionDelayResult(
                generated_tti=tti_index,
                executed_tti=tti_index,
                source_tti=None,
                allocation=(
                    empty_like_allocation(
                        allocation
                    )
                ),
            )

        (
            source_tti,
            executed,
        ) = self._queue.popleft()

        expected_source = (
            tti_index
            - self.delay_ttis
        )

        if (
            source_tti
            != expected_source
        ):
            raise RuntimeError(
                "Action-delay FIFO temporal ordering "
                "was violated."
            )

        return ActionDelayResult(
            generated_tti=tti_index,
            executed_tti=tti_index,
            source_tti=source_tti,
            allocation=executed,
        )
