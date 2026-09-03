import pytest
import torch

from oran_scheduler.rl.ppo_candidate_permutation import (
    build_candidate_permutation,
    permute_1lds_state,
    permute_candidate_action_mask,
    permute_ppo_decision_data,
    remap_candidate_actions,
    sample_candidate_permutation,
)

from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
    iter_candidate_permutations,
    permute_ppo_expert_data,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def test_candidate_permutation_builds_inverse():
    device = preferred_device()

    permutation = (
        build_candidate_permutation(
            torch.tensor(
                [
                    2,
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    torch.testing.assert_close(
        permutation.old_to_new,
        torch.tensor(
            [
                1,
                2,
                0,
            ],
            dtype=torch.long,
            device=device,
        ),
    )


def test_state_permutation_moves_whole_ue_segments():
    device = preferred_device()

    #
    # 3 candidates.
    # 4 features per candidate.
    #
    state = torch.tensor(
        [
            10.0,
            11.0,
            12.0,
            13.0,

            20.0,
            21.0,
            22.0,
            23.0,

            30.0,
            31.0,
            32.0,
            33.0,
        ],
        device=device,
    )

    permutation = (
        build_candidate_permutation(
            torch.tensor(
                [
                    2,
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    permuted = permute_1lds_state(
        state=state,
        permutation=permutation,
    )

    expected = torch.tensor(
        [
            30.0,
            31.0,
            32.0,
            33.0,

            10.0,
            11.0,
            12.0,
            13.0,

            20.0,
            21.0,
            22.0,
            23.0,
        ],
        device=device,
    )

    torch.testing.assert_close(
        permuted,
        expected,
    )


def test_candidate_actions_follow_same_users():
    device = preferred_device()

    permutation = (
        build_candidate_permutation(
            torch.tensor(
                [
                    2,
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    #
    # Old:
    #
    # 0=A, 1=B, 2=C, 3=NO ALLOCATION
    #
    actions = torch.tensor(
        [
            0,
            1,
            2,
            3,
        ],
        dtype=torch.long,
        device=device,
    )

    remapped = remap_candidate_actions(
        actions=actions,
        permutation=permutation,
    )

    #
    # New:
    #
    # 0=C, 1=A, 2=B, 3=NO ALLOCATION
    #
    expected = torch.tensor(
        [
            1,
            2,
            0,
            3,
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        remapped,
        expected,
    )


def test_action_mask_candidate_columns_are_permuted():
    device = preferred_device()

    permutation = (
        build_candidate_permutation(
            torch.tensor(
                [
                    2,
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    #
    # Columns:
    #
    # candidate 0
    # candidate 1
    # candidate 2
    # NO ALLOCATION
    #
    mask = torch.tensor(
        [
            [
                True,
                False,
                True,
                True,
            ],
            [
                False,
                True,
                True,
                True,
            ],
        ],
        dtype=torch.bool,
        device=device,
    )

    permuted = (
        permute_candidate_action_mask(
            action_mask=mask,
            permutation=permutation,
        )
    )

    expected = torch.tensor(
        [
            [
                True,   # old candidate 2
                True,   # old candidate 0
                False,  # old candidate 1
                True,   # NO ALLOCATION
            ],
            [
                True,
                False,
                True,
                True,
            ],
        ],
        dtype=torch.bool,
        device=device,
    )

    torch.testing.assert_close(
        permuted,
        expected,
    )

def test_remapped_action_remains_legal():
    device = preferred_device()

    permutation = (
        build_candidate_permutation(
            torch.tensor(
                [
                    2,
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    action_mask = torch.tensor(
        [
            [
                True,
                False,
                True,
                True,
            ],
            [
                False,
                True,
                True,
                True,
            ],
        ],
        dtype=torch.bool,
        device=device,
    )

    #
    # Both old actions are legal.
    #
    actions = torch.tensor(
        [
            2,
            1,
        ],
        dtype=torch.long,
        device=device,
    )

    permuted_mask = (
        permute_candidate_action_mask(
            action_mask=action_mask,
            permutation=permutation,
        )
    )

    remapped_actions = (
        remap_candidate_actions(
            actions=actions,
            permutation=permutation,
        )
    )

    selected_is_legal = (
        permuted_mask
        .gather(
            dim=1,
            index=(
                remapped_actions
                .unsqueeze(
                    1
                )
            ),
        )
        .squeeze(
            1
        )
    )

    assert torch.all(
        selected_is_legal
    )


def test_ppo_and_expert_actions_are_both_remapped():
    device = preferred_device()

    permutation = (
        build_candidate_permutation(
            torch.tensor(
                [
                    2,
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    result = permute_ppo_decision_data(
        state=torch.arange(
            12,
            dtype=torch.float32,
            device=device,
        ),
        actions=torch.tensor(
            [
                0,
                2,
            ],
            dtype=torch.long,
            device=device,
        ),
        expert_actions=torch.tensor(
            [
                1,
                0,
            ],
            dtype=torch.long,
            device=device,
        ),
        action_mask=torch.ones(
            (
                2,
                4,
            ),
            dtype=torch.bool,
            device=device,
        ),
        permutation=permutation,
    )

    torch.testing.assert_close(
        result.actions,
        torch.tensor(
            [
                1,
                0,
            ],
            dtype=torch.long,
            device=device,
        ),
    )

    torch.testing.assert_close(
        result.expert_actions,
        torch.tensor(
            [
                2,
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
    )


def test_permutation_round_trip_restores_data():
    device = preferred_device()

    forward = build_candidate_permutation(
        torch.tensor(
            [
                2,
                0,
                1,
            ],
            dtype=torch.long,
            device=device,
        )
    )

    inverse = build_candidate_permutation(
        forward.old_to_new
    )

    original_state = torch.arange(
        12,
        dtype=torch.float32,
        device=device,
    )

    original_actions = torch.tensor(
        [
            0,
            2,
            3,
        ],
        dtype=torch.long,
        device=device,
    )

    first_state = permute_1lds_state(
        state=original_state,
        permutation=forward,
    )

    restored_state = permute_1lds_state(
        state=first_state,
        permutation=inverse,
    )

    first_actions = remap_candidate_actions(
        actions=original_actions,
        permutation=forward,
    )

    restored_actions = remap_candidate_actions(
        actions=first_actions,
        permutation=inverse,
    )

    torch.testing.assert_close(
        restored_state,
        original_state,
    )

    torch.testing.assert_close(
        restored_actions,
        original_actions,
    )


def test_duplicate_candidate_index_is_rejected():
    device = preferred_device()

    with pytest.raises(
        ValueError,
        match="exactly once",
    ):
        build_candidate_permutation(
            torch.tensor(
                [
                    0,
                    0,
                    2,
                ],
                dtype=torch.long,
                device=device,
            )
        )


def test_sampled_permutation_is_reproducible():
    device = preferred_device()

    generator_a = torch.Generator(
        device="cpu"
    )
    generator_a.manual_seed(
        1234
    )

    generator_b = torch.Generator(
        device="cpu"
    )
    generator_b.manual_seed(
        1234
    )

    permutation_a = (
        sample_candidate_permutation(
            num_candidates=10,
            device=device,
            generator=generator_a,
        )
    )

    permutation_b = (
        sample_candidate_permutation(
            num_candidates=10,
            device=device,
            generator=generator_b,
        )
    )

    torch.testing.assert_close(
        permutation_a.new_to_old,
        permutation_b.new_to_old,
    )


def test_paper_shaped_candidate_permutation():
    device = preferred_device()

    #
    # Paper:
    #
    # K = 10
    # NRBG = 18
    #
    # candidate feature segment:
    #
    #     5 + 2*18 = 41
    #
    # state:
    #
    #     10 * 41 = 410
    #
    state = torch.arange(
        410,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ],
        dtype=torch.long,
        device=device,
    )

    action_mask = torch.ones(
        (
            18,
            11,
        ),
        dtype=torch.bool,
        device=device,
    )

    permutation = (
        sample_candidate_permutation(
            num_candidates=10,
            device=device,
        )
    )

    result = permute_ppo_decision_data(
        state=state,
        actions=actions,
        action_mask=action_mask,
        expert_actions=actions,
        next_state=state + 1000.0,
        permutation=permutation,
    )

    assert tuple(
        result.state.shape
    ) == (
        410,
    )

    assert tuple(
        result.actions.shape
    ) == (
        18,
    )

    assert tuple(
        result.action_mask.shape
    ) == (
        18,
        11,
    )

    assert tuple(
        result.next_state.shape
    ) == (
        410,
    )

    assert tuple(
        result.expert_actions.shape
    ) == (
        18,
    )

    #
    # NO ALLOCATION remains NO ALLOCATION.
    #
    assert int(
        result.actions[10].item()
    ) == 10

def test_augmentation_iterator_count_is_explicit():
    device = preferred_device()

    config = PPOCandidateAugmentationConfig(
        num_permutations=3,
        include_original=True,
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        1234
    )

    representations = list(
        iter_candidate_permutations(
            config=config,
            num_candidates=10,
            device=device,
            generator=generator,
        )
    )

    #
    # original + 3 permutations
    #
    assert len(
        representations
    ) == 4

    assert representations[0] is None

    assert all(
        permutation is not None
        for permutation
        in representations[1:]
    )


def test_augmentation_can_store_only_permutations():
    device = preferred_device()

    config = PPOCandidateAugmentationConfig(
        num_permutations=2,
        include_original=False,
    )

    representations = list(
        iter_candidate_permutations(
            config=config,
            num_candidates=3,
            device=device,
        )
    )

    assert len(
        representations
    ) == 2

    assert all(
        permutation is not None
        for permutation
        in representations
    )


def test_zero_output_augmentation_is_rejected():
    with pytest.raises(
        ValueError,
        match="zero",
    ):
        PPOCandidateAugmentationConfig(
            num_permutations=0,
            include_original=False,
        )


def test_permuted_expert_label_remains_legal():
    device = preferred_device()

    permutation = (
        build_candidate_permutation(
            torch.tensor(
                [
                    2,
                    0,
                    1,
                ],
                dtype=torch.long,
                device=device,
            )
        )
    )

    state = torch.arange(
        12,
        dtype=torch.float32,
        device=device,
    )

    expert_actions = torch.tensor(
        [
            0,
            2,
        ],
        dtype=torch.long,
        device=device,
    )

    action_mask = torch.tensor(
        [
            [
                True,
                False,
                True,
                True,
            ],
            [
                True,
                True,
                True,
                True,
            ],
        ],
        dtype=torch.bool,
        device=device,
    )

    result = permute_ppo_expert_data(
        state=state,
        expert_actions=(
            expert_actions
        ),
        action_mask=action_mask,
        permutation=permutation,
    )

    selected_is_legal = (
        result
        .action_mask
        .gather(
            dim=1,
            index=(
                result
                .expert_actions
                .unsqueeze(
                    1
                )
            ),
        )
        .squeeze(
            1
        )
    )

    assert torch.all(
        selected_is_legal
    )

    #
    # old action 0 -> new action 1
    # old action 2 -> new action 0
    #
    torch.testing.assert_close(
        result.expert_actions,
        torch.tensor(
            [
                1,
                0,
            ],
            dtype=torch.long,
            device=device,
        ),
    )













    