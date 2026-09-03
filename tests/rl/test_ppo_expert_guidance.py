import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertBatch,
    PPOExpertBufferConfig,
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_expert_guidance import (
    PPOExpertGuidanceConfig,
    build_one_hot_expert_distribution,
    compute_ppo_expert_guidance_loss,
    compute_true_jsd,
    perform_ppo_expert_guidance_update,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def test_one_hot_expert_distribution():
    device = preferred_device()

    expert_actions = torch.tensor(
        [
            [
                1,
                0,
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    action_masks = torch.ones(
        (
            1,
            2,
            3,
        ),
        dtype=torch.bool,
        device=device,
    )

    probabilities = (
        build_one_hot_expert_distribution(
            expert_actions=(
                expert_actions
            ),
            action_masks=(
                action_masks
            ),
            dtype=torch.float32,
        )
    )

    expected = torch.tensor(
        [
            [
                [
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                ],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    torch.testing.assert_close(
        probabilities,
        expected,
    )


def test_true_jsd_is_zero_for_identical_distributions():
    device = preferred_device()

    probabilities = torch.tensor(
        [
            [
                [
                    0.2,
                    0.3,
                    0.5,
                ]
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    jsd = compute_true_jsd(
        actor_probabilities=probabilities,
        expert_probabilities=(
            probabilities
        ),
        normalize_to_unit_interval=True,
    )

    torch.testing.assert_close(
        jsd,
        torch.zeros(
            (
                1,
                1,
            ),
            device=device,
        ),
        atol=1.0e-7,
        rtol=0.0,
    )


def test_true_jsd_handles_disjoint_one_hot_distributions():
    device = preferred_device()

    actor = torch.tensor(
        [
            [
                [
                    1.0,
                    0.0,
                ]
            ]
        ],
        device=device,
    )

    expert = torch.tensor(
        [
            [
                [
                    0.0,
                    1.0,
                ]
            ]
        ],
        device=device,
    )

    jsd = compute_true_jsd(
        actor_probabilities=actor,
        expert_probabilities=expert,
        normalize_to_unit_interval=True,
    )

    torch.testing.assert_close(
        jsd,
        torch.ones(
            (
                1,
                1,
            ),
            device=device,
        ),
        atol=1.0e-6,
        rtol=0.0,
    )

    assert torch.isfinite(
        jsd
    ).all()


def test_expert_loss_respects_saved_action_mask():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=4,
            hidden_size=8,
            num_rbgs=1,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    batch = PPOExpertBatch(
        states=torch.zeros(
            (
                1,
                4,
            ),
            device=device,
        ),
        expert_actions=torch.tensor(
            [
                [
                    2,
                ]
            ],
            dtype=torch.long,
            device=device,
        ),
        action_masks=torch.tensor(
            [
                [
                    [
                        True,
                        False,
                        True,
                    ]
                ]
            ],
            dtype=torch.bool,
            device=device,
        ),
    )

    loss = compute_ppo_expert_guidance_loss(
        actor=actor,
        batch=batch,
        config=PPOExpertGuidanceConfig(
            batch_size=1,
            guidance_weight=1.0,
            divergence_mode="true_jsd",
        ),
    )

    assert float(
        loss
        .actor_probabilities[
            0,
            0,
            1,
        ].item()
    ) == 0.0

    torch.testing.assert_close(
        loss.actor_probabilities.sum(
            dim=-1
        ),
        torch.ones(
            (
                1,
                1,
            ),
            device=device,
        ),
    )


def test_expert_guidance_update_increases_expert_action_probability():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=4,
            hidden_size=8,
            num_rbgs=1,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    #
    # Start from a perfectly symmetric actor so the
    # three actions initially have equal probability.
    #
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()

    state = torch.zeros(
        4,
        dtype=torch.float32,
        device=device,
    )

    mask = torch.ones(
        (
            1,
            3,
        ),
        dtype=torch.bool,
        device=device,
    )

    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=10,
            )
        )
    )

    #
    # PF teacher says action 1.
    #
    expert_buffer.add(
        state=state,
        expert_actions=torch.tensor(
            [
                1,
            ],
            dtype=torch.long,
            device=device,
        ),
        action_mask=mask,
    )

    with torch.no_grad():
        (
            distribution_before,
            _,
        ) = actor.build_distribution(
            state=state.unsqueeze(
                0
            ),
            action_mask=mask.unsqueeze(
                0
            ),
        )

        probability_before = float(
            distribution_before
            .probs[
                0,
                0,
                1,
            ]
            .item()
        )

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=0.05,
    )

    update = (
        perform_ppo_expert_guidance_update(
            actor=actor,
            actor_optimizer=(
                actor_optimizer
            ),
            expert_buffer=(
                expert_buffer
            ),
            config=PPOExpertGuidanceConfig(
                batch_size=1,
                guidance_weight=1.0,
                divergence_mode="true_jsd",
            ),
        )
    )

    with torch.no_grad():
        (
            distribution_after,
            _,
        ) = actor.build_distribution(
            state=state.unsqueeze(
                0
            ),
            action_mask=mask.unsqueeze(
                0
            ),
        )

        probability_after = float(
            distribution_after
            .probs[
                0,
                0,
                1,
            ]
            .item()
        )

    assert (
        probability_after
        > probability_before
    )

    assert torch.isfinite(
        update
        .loss_data
        .weighted_loss
    )


def test_printed_symmetric_kl_smoothed_is_finite():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=4,
            hidden_size=8,
            num_rbgs=1,
            num_actions_per_rbg=3,
        )
    ).to(
        device
    )

    batch = PPOExpertBatch(
        states=torch.zeros(
            (
                2,
                4,
            ),
            device=device,
        ),
        expert_actions=torch.tensor(
            [
                [
                    0,
                ],
                [
                    2,
                ],
            ],
            dtype=torch.long,
            device=device,
        ),
        action_masks=torch.ones(
            (
                2,
                1,
                3,
            ),
            dtype=torch.bool,
            device=device,
        ),
    )

    loss = compute_ppo_expert_guidance_loss(
        actor=actor,
        batch=batch,
        config=PPOExpertGuidanceConfig(
            batch_size=2,
            guidance_weight=1.0,
            divergence_mode=(
                "printed_symmetric_kl_smoothed"
            ),
            printed_kl_epsilon=1.0e-6,
        ),
    )

    assert torch.isfinite(
        loss.raw_loss
    )

    assert torch.isfinite(
        loss.divergence_by_rbg
    ).all()


def test_expert_guidance_paper_shaped_batch_on_gpu():
    device = preferred_device()

    actor = OneLDSPPOActor(
        OneLDSPPOActorConfig(
            state_size=410,
            hidden_size=32,
            num_rbgs=18,
            num_actions_per_rbg=11,
        )
    ).to(
        device
    )

    batch_size = 8

    states = torch.zeros(
        (
            batch_size,
            410,
        ),
        dtype=torch.float32,
        device=device,
    )

    expert_actions = torch.zeros(
        (
            batch_size,
            18,
        ),
        dtype=torch.long,
        device=device,
    )

    masks = torch.ones(
        (
            batch_size,
            18,
            11,
        ),
        dtype=torch.bool,
        device=device,
    )

    batch = PPOExpertBatch(
        states=states,
        expert_actions=(
            expert_actions
        ),
        action_masks=masks,
    )

    loss = compute_ppo_expert_guidance_loss(
        actor=actor,
        batch=batch,
        config=PPOExpertGuidanceConfig(
            batch_size=batch_size,
            guidance_weight=1.0,
            divergence_mode="true_jsd",
        ),
    )

    assert tuple(
        loss.actor_probabilities.shape
    ) == (
        batch_size,
        18,
        11,
    )

    assert tuple(
        loss.expert_probabilities.shape
    ) == (
        batch_size,
        18,
        11,
    )

    assert tuple(
        loss.divergence_by_rbg.shape
    ) == (
        batch_size,
        18,
    )

    assert loss.raw_loss.ndim == 0

    assert loss.raw_loss.device == device

    assert torch.isfinite(
        loss.raw_loss
    )


