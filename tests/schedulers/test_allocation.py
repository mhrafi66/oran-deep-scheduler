import pytest
import torch

from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    CellAllocation,
    build_next_user_slot_action_mask,
    selected_candidates_for_rbg,
    selected_global_ues_for_rbg,
    validate_cell_allocation,
)


def test_selected_candidates_and_global_ues():
    device = "cuda:0"

    actions = torch.tensor(
        [
            [0, 1, -1],
            [2, -1, 3],
            [-1, -1, -1],
        ],
        dtype=torch.long,
        device=device,
    )

    allocation = CellAllocation(
        candidate_by_user_slot=actions
    )

    candidate_global_ues = torch.tensor(
        [10, 20, 30, 40],
        dtype=torch.long,
        device=device,
    )

    validate_cell_allocation(
        allocation=allocation,
        num_candidates=4,
    )

    selected_candidates = (
        selected_candidates_for_rbg(
            allocation=allocation,
            rbg_index=0,
        )
    )

    expected_candidates = torch.tensor(
        [0, 2],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        selected_candidates,
        expected_candidates,
    )

    global_ues = (
        selected_global_ues_for_rbg(
            allocation=allocation,
            candidate_global_ue_indices=(
                candidate_global_ues
            ),
            rbg_index=0,
        )
    )

    expected_global_ues = torch.tensor(
        [10, 30],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        global_ues,
        expected_global_ues,
    )

def test_duplicate_candidate_on_same_rbg_is_invalid():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [1],
                [1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    with pytest.raises(
        ValueError,
        match="same candidate UE",
    ):
        validate_cell_allocation(
            allocation=allocation,
            num_candidates=3,
        )

def test_next_user_slot_action_mask():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0, 1],
                [-1, -1],
                [-1, -1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    mask = build_next_user_slot_action_mask(
        allocation=allocation,
        num_candidates=3,
        user_slot_index=1,
    )

    assert mask.shape == (
        2,
        4,
    )

    assert not bool(
        mask[
            0,
            0,
        ].item()
    )

    assert bool(
        mask[
            0,
            1,
        ].item()
    )

    assert bool(
        mask[
            0,
            2,
        ].item()
    )

    assert bool(
        mask[
            0,
            3,
        ].item()
    )

    assert not bool(
        mask[
            1,
            1,
        ].item()
    )

