from dataclasses import dataclass

import torch


NO_ALLOCATION = -1


@dataclass
class CellAllocation:
    """
    Candidate-UE allocation for one cell.

    candidate_by_user_slot:
        [user_slot, RBG]

    Each entry contains:

        0 ... K-1
            index into the PF-TDS candidate list

        -1
            no allocation

    Important:
        A scheduler user slot is NOT a physical MIMO stream.

        If a selected UE has rank 2, that one user-slot
        selection later expands into two physical streams.
    """

    candidate_by_user_slot: torch.Tensor


    @property
    def num_user_slots(self) -> int:
        return int(
            self.candidate_by_user_slot.shape[0]
        )

    @property
    def num_rbgs(self) -> int:
        return int(
            self.candidate_by_user_slot.shape[1]
        )

def validate_cell_allocation(
    allocation: CellAllocation,
    num_candidates: int,
) -> None:
    """
    Validate one cell's scheduler allocation.

    Rules:
        - allocation must be [user_slot, RBG]
        - candidate indices are -1 or 0..K-1
        - the same candidate UE cannot be selected twice
          on the same RBG
    """

    actions = (
        allocation.candidate_by_user_slot
    )

    if actions.ndim != 2:
        raise ValueError(
            "candidate_by_user_slot must have shape "
            "[user_slot, RBG]."
        )

    if torch.is_floating_point(
        actions
    ):
        raise ValueError(
            "Scheduler actions must use an integer dtype."
        )

    if num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive."
        )

    if torch.any(
        actions < NO_ALLOCATION
    ):
        raise ValueError(
            "Action contains a value below -1."
        )

    if torch.any(
        actions >= num_candidates
    ):
        raise ValueError(
            "Action contains an invalid candidate index."
        )

    num_rbgs = actions.shape[1]

    for rbg_index in range(
        num_rbgs
    ):

        rbg_actions = actions[
            :,
            rbg_index,
        ]

        allocated = rbg_actions[
            rbg_actions
            != NO_ALLOCATION
        ]

        if allocated.numel() == 0:
            continue

        if torch.unique(
            allocated
        ).numel() != allocated.numel():
            raise ValueError(
                "The same candidate UE is selected "
                "more than once on one RBG."
            )

def selected_candidates_for_rbg(
    allocation: CellAllocation,
    rbg_index: int,
) -> torch.Tensor:
    """
    Return candidate indices scheduled on one RBG.

    Output:
        [num_scheduled_UEs_on_this_RBG]
    """

    if not (
        0
        <= rbg_index
        < allocation.num_rbgs
    ):
        raise ValueError(
            "rbg_index is invalid."
        )

    rbg_actions = (
        allocation
        .candidate_by_user_slot[
            :,
            rbg_index,
        ]
    )

    return rbg_actions[
        rbg_actions
        != NO_ALLOCATION
    ]


def selected_global_ues_for_rbg(
    allocation: CellAllocation,
    candidate_global_ue_indices: torch.Tensor,
    rbg_index: int,
) -> torch.Tensor:
    """
    Map the scheduler's candidate indices to global UE IDs.

    candidate_global_ue_indices:
        [candidate]

    Returns:
        [scheduled_UE]
    """

    if candidate_global_ue_indices.ndim != 1:
        raise ValueError(
            "candidate_global_ue_indices must have "
            "shape [candidate]."
        )

    candidate_indices = (
        selected_candidates_for_rbg(
            allocation=allocation,
            rbg_index=rbg_index,
        )
    )

    if candidate_indices.numel() == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=(
                candidate_global_ue_indices.device
            ),
        )

    return candidate_global_ue_indices[
        candidate_indices
    ]

def build_next_user_slot_action_mask(
    allocation: CellAllocation,
    num_candidates: int,
    user_slot_index: int,
) -> torch.Tensor:
    """
    Build valid actions for the next scheduler user slot.

    Returns:
        [RBG, num_candidates + 1]

    Final column:
        no-allocation action

    Candidate actions already used on an RBG are masked out.
    """

    if not (
        0
        <= user_slot_index
        < allocation.num_user_slots
    ):
        raise ValueError(
            "user_slot_index is invalid."
        )

    num_rbgs = (
        allocation.num_rbgs
    )

    mask = torch.ones(
        (
            num_rbgs,
            num_candidates + 1,
        ),
        dtype=torch.bool,
        device=(
            allocation
            .candidate_by_user_slot
            .device
        ),
    )

    if user_slot_index == 0:
        return mask

    previous_actions = (
        allocation
        .candidate_by_user_slot[
            :user_slot_index,
            :,
        ]
    )

    for rbg_index in range(
        num_rbgs
    ):

        previous_on_rbg = (
            previous_actions[
                :,
                rbg_index,
            ]
        )

        previous_on_rbg = (
            previous_on_rbg[
                previous_on_rbg
                != NO_ALLOCATION
            ]
        )

        if (
            previous_on_rbg.numel()
            == 0
        ):
            continue

        mask[
            rbg_index,
            previous_on_rbg,
        ] = False

    mask[
        :,
        num_candidates,
    ] = True

    return mask






    