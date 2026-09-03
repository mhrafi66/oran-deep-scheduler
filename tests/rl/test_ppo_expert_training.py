import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    OneLDSPPOActorConfig,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertBufferConfig,
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_expert_guidance import (
    PPOExpertGuidanceConfig,
)
from oran_scheduler.rl.ppo_expert_training import (
    PPOExpertGuidanceCoordinator,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def test_expert_coordinator_performs_one_guidance_update():
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

    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()

    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=10,
            )
        )
    )

    state = torch.zeros(
        4,
        dtype=torch.float32,
        device=device,
    )

    action_mask = torch.ones(
        (
            1,
            3,
        ),
        dtype=torch.bool,
        device=device,
    )

    #
    # Teacher says action 1.
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
        action_mask=action_mask,
    )

    coordinator = (
        PPOExpertGuidanceCoordinator(
            expert_buffer=expert_buffer,
            guidance_config=(
                PPOExpertGuidanceConfig(
                    batch_size=1,
                    guidance_weight=1.0,
                    divergence_mode="true_jsd",
                )
            ),
        )
    )

    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=0.05,
    )

    with torch.no_grad():
        distribution_before, _ = (
            actor.build_distribution(
                state=state.unsqueeze(
                    0
                ),
                action_mask=(
                    action_mask.unsqueeze(
                        0
                    )
                ),
            )
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

    result = (
        coordinator.perform_after_ppo_update(
            ppo_update_index=1,
            actor=actor,
            actor_optimizer=optimizer,
        )
    )

    with torch.no_grad():
        distribution_after, _ = (
            actor.build_distribution(
                state=state.unsqueeze(
                    0
                ),
                action_mask=(
                    action_mask.unsqueeze(
                        0
                    )
                ),
            )
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

    assert (
        coordinator.num_guidance_updates
        == 1
    )

    assert (
        coordinator.last_update
        is result
    )

    assert result.ppo_update_index == 1

    assert result.expert_buffer_size == 1


def test_expert_coordinator_requires_full_batch():
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

    expert_buffer = (
        PPOExpertDemonstrationBuffer(
            PPOExpertBufferConfig(
                replacement_mode="fifo",
                capacity=10,
                sampling_with_replacement=False,
            )
        )
    )

    expert_buffer.add(
        state=torch.zeros(
            4,
            device=device,
        ),
        expert_actions=torch.zeros(
            1,
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
    )

    coordinator = (
        PPOExpertGuidanceCoordinator(
            expert_buffer=expert_buffer,
            guidance_config=(
                PPOExpertGuidanceConfig(
                    batch_size=2,
                    guidance_weight=1.0,
                    divergence_mode="true_jsd",
                )
            ),
        )
    )

    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=1.0e-4,
    )

    try:
        coordinator.perform_after_ppo_update(
            ppo_update_index=1,
            actor=actor,
            actor_optimizer=optimizer,
        )

    except RuntimeError as error:
        assert "enough demonstrations" in str(
            error
        )

    else:
        raise AssertionError(
            "Expected insufficient expert data "
            "to raise RuntimeError."
        )

    assert (
        coordinator.num_guidance_updates
        == 0
    )


