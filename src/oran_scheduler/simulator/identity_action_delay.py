from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import torch


NO_GLOBAL_UE = -1


@dataclass(frozen=True)
class GlobalUEAllocation:
    """
    Scheduling decision represented by persistent
    GLOBAL UE identities rather than candidate slots.

    Shape:
        [user_slot, RBG]

    This fixes the fundamental problem with delaying
    candidate-index actions:

        candidate slot 3 at t
        !=
        candidate slot 3 at t+D.
    """

    global_ue_by_user_slot: torch.Tensor


    def __post_init__(
        self,
    ) -> None:

        tensor = (
            self.global_ue_by_user_slot
        )

        if tensor.ndim != 2:
            raise ValueError(
                "global_ue_by_user_slot must have "
                "shape [user_slot, RBG]."
            )

        if tensor.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError(
                "Global UE allocation must use "
                "integer dtype."
            )

        if torch.any(
            tensor < NO_GLOBAL_UE
        ):
            raise ValueError(
                "Global UE IDs cannot be below -1."
            )


    @property
    def num_user_slots(
        self,
    ) -> int:

        return int(
            self
            .global_ue_by_user_slot
            .shape[0]
        )


    @property
    def num_rbgs(
        self,
    ) -> int:

        return int(
            self
            .global_ue_by_user_slot
            .shape[1]
        )


    @classmethod
    def empty(
        cls,
        *,
        num_user_slots: int,
        num_rbgs: int,
        device: torch.device | str,
    ) -> "GlobalUEAllocation":

        if num_user_slots <= 0:
            raise ValueError(
                "num_user_slots must be positive."
            )

        if num_rbgs <= 0:
            raise ValueError(
                "num_rbgs must be positive."
            )

        return cls(
            global_ue_by_user_slot=(
                torch.full(
                    (
                        num_user_slots,
                        num_rbgs,
                    ),
                    NO_GLOBAL_UE,
                    dtype=torch.long,
                    device=device,
                )
            )
        )


class IdentitySafeActionDelay:
    """
    Delay actions by D TTIs while preserving UE
    identity.

    Startup behavior:
        empty allocation until the FIFO matures.
    """

    def __init__(
        self,
        *,
        delay_ttis: int,
        num_user_slots: int,
        num_rbgs: int,
        device: torch.device | str,
    ) -> None:

        if delay_ttis < 0:
            raise ValueError(
                "delay_ttis must be non-negative."
            )

        self.delay_ttis = delay_ttis

        self.num_user_slots = (
            num_user_slots
        )

        self.num_rbgs = num_rbgs

        self.device = torch.device(
            device
        )

        self._fifo: deque[
            GlobalUEAllocation
        ] = deque()


    def push(
        self,
        allocation: GlobalUEAllocation,
    ) -> GlobalUEAllocation:

        tensor = (
            allocation
            .global_ue_by_user_slot
        )

        if tensor.shape != (
            self.num_user_slots,
            self.num_rbgs,
        ):
            raise ValueError(
                "Allocation shape mismatch."
            )

        cloned = GlobalUEAllocation(
            global_ue_by_user_slot=(
                tensor
                .detach()
                .clone()
                .to(self.device)
            )
        )

        if self.delay_ttis == 0:
            return cloned

        self._fifo.append(cloned)

        if (
            len(self._fifo)
            <= self.delay_ttis
        ):
            return GlobalUEAllocation.empty(
                num_user_slots=(
                    self.num_user_slots
                ),
                num_rbgs=(
                    self.num_rbgs
                ),
                device=self.device,
            )

        return self._fifo.popleft()


def remap_global_allocation_to_candidates(
    *,
    allocation: GlobalUEAllocation,
    candidate_global_ue_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    no_allocation_action: int,
) -> torch.Tensor:
    """
    Convert a delayed GLOBAL-identity allocation to
    CURRENT candidate-slot actions.

    A UE that is no longer present in the current
    candidate set becomes no-allocation.

    Output:
        [user_slot, RBG]
    """

    if candidate_global_ue_indices.ndim != 1:
        raise ValueError(
            "candidate_global_ue_indices must "
            "be rank 1."
        )

    if (
        candidate_valid_mask.shape
        != candidate_global_ue_indices.shape
    ):
        raise ValueError(
            "candidate_valid_mask shape mismatch."
        )

    num_candidates = int(
        candidate_global_ue_indices.numel()
    )

    if (
        no_allocation_action
        < num_candidates
    ):
        raise ValueError(
            "no_allocation_action must not overlap "
            "candidate indices."
        )

    output = torch.full(
        allocation
        .global_ue_by_user_slot
        .shape,

        int(no_allocation_action),

        dtype=torch.long,

        device=(
            allocation
            .global_ue_by_user_slot
            .device
        ),
    )

    candidate_ids = (
        candidate_global_ue_indices
        .to(
            allocation
            .global_ue_by_user_slot
            .device
        )
    )

    valid = (
        candidate_valid_mask
        .to(
            allocation
            .global_ue_by_user_slot
            .device
        )
        .bool()
    )

    for candidate_index in range(
        num_candidates
    ):

        if not bool(
            valid[candidate_index].item()
        ):
            continue

        global_ue = int(
            candidate_ids[
                candidate_index
            ].item()
        )

        match = (
            allocation
            .global_ue_by_user_slot
            == global_ue
        )

        output[match] = (
            candidate_index
        )

    return output
