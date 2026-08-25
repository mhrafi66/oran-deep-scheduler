import pytest
import torch

from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    CellAllocation,
    build_initial_allocation_from_fds,
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

def test_build_initial_allocation_from_fds():
    device = "cuda:0"

    selected_candidate_by_rbg = torch.tensor(
        [
            2,
            0,
            3,
            1,
        ],
        dtype=torch.long,
        device=device,
    )

    selected_rbg_valid_mask = torch.tensor(
        [
            True,
            True,
            False,
            True,
        ],
        dtype=torch.bool,
        device=device,
    )

    allocation = build_initial_allocation_from_fds(
        selected_candidate_by_rbg=(
            selected_candidate_by_rbg
        ),
        selected_rbg_valid_mask=(
            selected_rbg_valid_mask
        ),
        num_user_slots=4,
    )

    expected = torch.tensor(
        [
            [2, 0, -1, 1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        allocation.candidate_by_user_slot,
        expected,
    )

    assert allocation.num_user_slots == 4
    assert allocation.num_rbgs == 4


def test_action_mask_excludes_invalid_candidates():
    device = "cuda:0"

    allocation = CellAllocation(
        candidate_by_user_slot=torch.tensor(
            [
                [0, 1],
                [-1, -1],
            ],
            dtype=torch.long,
            device=device,
        )
    )

    candidate_valid_mask = torch.tensor(
        [
            True,
            True,
            False,
            False,
        ],
        dtype=torch.bool,
        device=device,
    )

    mask = build_next_user_slot_action_mask(
        allocation=allocation,
        num_candidates=4,
        user_slot_index=1,
        candidate_valid_mask=(
            candidate_valid_mask
        ),
    )

    assert mask.shape == (
        2,
        5,
    )

    assert not bool(
        mask[
            0,
            2,
        ].item()
    )

    assert not bool(
        mask[
            0,
            3,
        ].item()
    )

    assert not bool(
        mask[
            1,
            2,
        ].item()
    )

    assert not bool(
        mask[
            1,
            3,
        ].item()
    )

    assert not bool(
        mask[
            0,
            0,
        ].item()
    )

    assert not bool(
        mask[
            1,
            1,
        ].item()
    )

    assert bool(
        mask[
            0,
            4,
        ].item()
    )

    assert bool(
        mask[
            1,
            4,
        ].item()
    )



