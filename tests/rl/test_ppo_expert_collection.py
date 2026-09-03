import torch

from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertBufferConfig,
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_expert_collection import (
    PPOPFExpertLayerLabel,
    PPOPFExpertTTILabels,
    commit_ppo_pf_expert_tti_labels,
)



def test_expert_commit_multiplies_samples_by_permutations():
    device = (
        torch.device(
            "cuda:0"
        )
        if torch.cuda.is_available()
        else torch.device(
            "cpu"
        )
    )

    buffer = PPOExpertDemonstrationBuffer(
        PPOExpertBufferConfig(
            replacement_mode="fifo",
            capacity=100,
        )
    )

    #
    # K = 3 candidates
    # actions = 4 including NO ALLOCATION
    #
    layer_0 = PPOPFExpertLayerLabel(
        user_slot_index=0,
        state=torch.arange(
            12,
            dtype=torch.float32,
            device=device,
        ),
        expert_actions=torch.tensor(
            [
                0,
                1,
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
        best_pf_sum=torch.tensor(
            [
                5.0,
                6.0,
            ],
            device=device,
        ),
        num_phy_evaluations=8,
    )

    layer_1 = PPOPFExpertLayerLabel(
        user_slot_index=1,
        state=torch.arange(
            12,
            dtype=torch.float32,
            device=device,
        ) + 100.0,
        expert_actions=torch.tensor(
            [
                2,
                3,
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
        best_pf_sum=torch.tensor(
            [
                7.0,
                8.0,
            ],
            device=device,
        ),
        num_phy_evaluations=8,
    )

    labels = PPOPFExpertTTILabels(
        layers=(
            layer_0,
            layer_1,
        ),
        total_num_phy_evaluations=16,
    )

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        1234
    )

    num_added = (
        commit_ppo_pf_expert_tti_labels(
            labels=labels,
            expert_buffer=buffer,
            augmentation_config=(
                PPOCandidateAugmentationConfig(
                    num_permutations=2,
                    include_original=True,
                )
            ),
            augmentation_generator=(
                generator
            ),
        )
    )

    #
    # 2 physical layers
    #
    # each produces:
    #     1 original
    #     2 permuted
    #
    # total:
    #     2 * 3 = 6
    #
    assert num_added == 6

    assert len(
        buffer
    ) == 6

    assert (
        buffer.num_added_total
        == 6
    )

    batch = buffer.sample(
        batch_size=6,
    )

    expert_is_legal = (
        batch
        .action_masks
        .gather(
            dim=2,
            index=(
                batch
                .expert_actions
                .unsqueeze(
                    -1
                )
            ),
        )
        .squeeze(
            -1
        )
    )

    assert torch.all(
        expert_is_legal
    )


def test_expert_commit_without_augmentation_is_unchanged():
    device = (
        torch.device(
            "cuda:0"
        )
        if torch.cuda.is_available()
        else torch.device(
            "cpu"
        )
    )

    buffer = PPOExpertDemonstrationBuffer(
        PPOExpertBufferConfig(
            replacement_mode="fifo",
            capacity=10,
        )
    )

    labels = PPOPFExpertTTILabels(
        layers=(
            PPOPFExpertLayerLabel(
                user_slot_index=0,
                state=torch.zeros(
                    4,
                    device=device,
                ),
                expert_actions=torch.tensor(
                    [
                        0,
                    ],
                    dtype=torch.long,
                    device=device,
                ),
                action_mask=torch.ones(
                    (
                        1,
                        3,
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
                best_pf_sum=torch.ones(
                    1,
                    device=device,
                ),
                num_phy_evaluations=3,
            ),
        ),
        total_num_phy_evaluations=3,
    )

    num_added = (
        commit_ppo_pf_expert_tti_labels(
            labels=labels,
            expert_buffer=buffer,
        )
    )

    assert num_added == 1

    assert len(
        buffer
    ) == 1


