from collections.abc import Callable
from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
    apply_user_slot_actions,
    build_empty_cell_allocation,
)
from oran_scheduler.state.one_lds import (
    OneLDSStateConfig,
)
from oran_scheduler.state.one_lds_decision import (
    OneLDSDecisionData,
    OneLDSDecisionInputs,
    build_1lds_decision_data,
)


OneLDSActionPolicy = Callable[
    [
        int,
        OneLDSDecisionData,
    ],
    torch.Tensor,
]


@dataclass(frozen=True)
class OneLDSScheduleResult:
    """
    Result of one complete 1LDS scheduling loop
    for one cell and one TTI.

    allocation:
        Final CellAllocation after all UE slots.

    decisions:
        Decision state seen before each UE-slot action.

    actions:
        Actor-space actions with shape:

            [user_slot, RBG]

        Values:
            0 ... K - 1 -> candidate index
            K           -> NO ALLOCATION
    """

    allocation: CellAllocation

    decisions: tuple[
        OneLDSDecisionData,
        ...
    ]

    actions: torch.Tensor

def run_1lds_user_slot_loop(
    *,
    num_user_slots: int,
    inputs: OneLDSDecisionInputs,
    state_config: OneLDSStateConfig,
    action_policy: OneLDSActionPolicy,
    device: str | torch.device,
) -> OneLDSScheduleResult:
    """
    Run all 1LDS MU-MIMO UE-slot decisions for
    one cell and one TTI.

    IMPORTANT:

    A scheduler UE slot is NOT a physical MIMO stream.

    A rank-2 UE selected once occupies one UE slot,
    and is later expanded by the PHY into two
    physical streams.
    """

    if num_user_slots <= 0:
        raise ValueError(
            "num_user_slots must be positive."
        )

    allocation = build_empty_cell_allocation(
        num_user_slots=num_user_slots,
        num_rbgs=state_config.num_rbgs,
        device=device,
    )

    decisions: list[
        OneLDSDecisionData
    ] = []

    actions_by_user_slot: list[
        torch.Tensor
    ] = []

    for user_slot_index in range(
        num_user_slots
    ):
        decision = build_1lds_decision_data(
            allocation=allocation,
            user_slot_index=user_slot_index,
            inputs=inputs,
            config=state_config,
        )

        actions = action_policy(
            user_slot_index,
            decision,
        )

        if not isinstance(
            actions,
            torch.Tensor,
        ):
            raise TypeError(
                "action_policy must return "
                "a torch.Tensor."
            )

        allocation = apply_user_slot_actions(
            allocation=allocation,
            user_slot_index=user_slot_index,
            actions=actions,
            num_candidates=(
                state_config.num_candidates
            ),
            candidate_valid_mask=(
                inputs.candidate_valid_mask
            ),
        )

        decisions.append(
            decision
        )

        actions_by_user_slot.append(
            actions.clone()
        )

    action_tensor = torch.stack(
        actions_by_user_slot,
        dim=0,
    )

    expected_shape = (
        num_user_slots,
        state_config.num_rbgs,
    )

    if tuple(
        action_tensor.shape
    ) != expected_shape:
        raise RuntimeError(
            "Internal 1LDS action-history "
            "shape error."
        )

    return OneLDSScheduleResult(
        allocation=allocation,
        decisions=tuple(
            decisions
        ),
        actions=action_tensor,
    )


