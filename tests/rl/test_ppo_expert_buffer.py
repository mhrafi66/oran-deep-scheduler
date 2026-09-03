import pytest
import torch

from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertBufferConfig,
    PPOExpertDemonstrationBuffer,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def test_expert_buffer_uses_paper_capacity():
    config = PPOExpertBufferConfig(
        replacement_mode="fifo",
    )

    assert config.capacity == 4000


def test_expert_buffer_stores_paper_shaped_demo():
    device = preferred_device()

    buffer = PPOExpertDemonstrationBuffer(
        PPOExpertBufferConfig(
            replacement_mode="fifo",
        )
    )

    state = torch.zeros(
        410,
        dtype=torch.float32,
        device=device,
    )

    expert_actions = torch.zeros(
        18,
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

    accepted = buffer.add(
        state=state,
        expert_actions=expert_actions,
        action_mask=action_mask,
    )

    assert accepted

    assert len(buffer) == 1

    batch = buffer.sample(
        batch_size=1,
    )

    assert tuple(
        batch.states.shape
    ) == (
        1,
        410,
    )

    assert tuple(
        batch.expert_actions.shape
    ) == (
        1,
        18,
    )

    assert tuple(
        batch.action_masks.shape
    ) == (
        1,
        18,
        11,
    )

    assert batch.states.device == device

    assert (
        batch.expert_actions.device
        == device
    )

    assert (
        batch.action_masks.device
        == device
    )


def test_expert_buffer_rejects_masked_action():
    device = preferred_device()

    buffer = PPOExpertDemonstrationBuffer(
        PPOExpertBufferConfig(
            replacement_mode="fifo",
        )
    )

    state = torch.zeros(
        4,
        dtype=torch.float32,
        device=device,
    )

    expert_actions = torch.tensor(
        [
            1,
            0,
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
            ],
            [
                True,
                True,
                True,
            ],
        ],
        dtype=torch.bool,
        device=device,
    )

    with pytest.raises(
        ValueError,
        match="masked",
    ):
        buffer.add(
            state=state,
            expert_actions=expert_actions,
            action_mask=action_mask,
        )


def test_fifo_replacement_keeps_recent_demos():
    device = preferred_device()

    buffer = PPOExpertDemonstrationBuffer(
        PPOExpertBufferConfig(
            replacement_mode="fifo",
            capacity=2,
        )
    )

    action_mask = torch.ones(
        (
            1,
            2,
        ),
        dtype=torch.bool,
        device=device,
    )

    expert_actions = torch.zeros(
        1,
        dtype=torch.long,
        device=device,
    )

    for value in (
        1.0,
        2.0,
        3.0,
    ):
        buffer.add(
            state=torch.tensor(
                [
                    value,
                ],
                dtype=torch.float32,
                device=device,
            ),
            expert_actions=(
                expert_actions
            ),
            action_mask=action_mask,
        )

    assert len(buffer) == 2

    assert buffer.is_full

    assert (
        buffer.num_added_total
        == 3
    )

    generator = torch.Generator()
    generator.manual_seed(
        1234
    )

    batch = buffer.sample(
        batch_size=2,
        generator=generator,
    )

    stored_values = set(
        batch.states[
            :,
            0,
        ]
        .cpu()
        .tolist()
    )

    assert stored_values == {
        2.0,
        3.0,
    }


def test_reject_when_full_keeps_original_demos():
    device = preferred_device()

    buffer = PPOExpertDemonstrationBuffer(
        PPOExpertBufferConfig(
            replacement_mode=(
                "reject_when_full"
            ),
            capacity=2,
        )
    )

    action_mask = torch.ones(
        (
            1,
            2,
        ),
        dtype=torch.bool,
        device=device,
    )

    expert_actions = torch.zeros(
        1,
        dtype=torch.long,
        device=device,
    )

    assert buffer.add(
        state=torch.tensor(
            [
                1.0,
            ],
            device=device,
        ),
        expert_actions=expert_actions,
        action_mask=action_mask,
    )

    assert buffer.add(
        state=torch.tensor(
            [
                2.0,
            ],
            device=device,
        ),
        expert_actions=expert_actions,
        action_mask=action_mask,
    )

    accepted = buffer.add(
        state=torch.tensor(
            [
                3.0,
            ],
            device=device,
        ),
        expert_actions=expert_actions,
        action_mask=action_mask,
    )

    assert not accepted

    assert len(buffer) == 2

    generator = torch.Generator()
    generator.manual_seed(
        1234
    )

    batch = buffer.sample(
        batch_size=2,
        generator=generator,
    )

    stored_values = set(
        batch.states[
            :,
            0,
        ]
        .cpu()
        .tolist()
    )

    assert stored_values == {
        1.0,
        2.0,
    }


def test_expert_buffer_stores_detached_copies():
    device = preferred_device()

    buffer = PPOExpertDemonstrationBuffer(
        PPOExpertBufferConfig(
            replacement_mode="fifo",
        )
    )

    state = torch.tensor(
        [
            1.0,
            2.0,
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    expert_actions = torch.tensor(
        [
            0,
        ],
        dtype=torch.long,
        device=device,
    )

    action_mask = torch.ones(
        (
            1,
            2,
        ),
        dtype=torch.bool,
        device=device,
    )

    buffer.add(
        state=state,
        expert_actions=expert_actions,
        action_mask=action_mask,
    )

    with torch.no_grad():
        state[
            0
        ] = 999.0

    batch = buffer.sample(
        batch_size=1,
    )

    torch.testing.assert_close(
        batch.states[
            0
        ],
        torch.tensor(
            [
                1.0,
                2.0,
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

    assert not (
        batch.states.requires_grad
    )


def test_sampling_without_replacement_requires_enough_data():
    device = preferred_device()

    buffer = PPOExpertDemonstrationBuffer(
        PPOExpertBufferConfig(
            replacement_mode="fifo",
            capacity=10,
            sampling_with_replacement=False,
        )
    )

    buffer.add(
        state=torch.zeros(
            4,
            device=device,
        ),
        expert_actions=torch.zeros(
            2,
            dtype=torch.long,
            device=device,
        ),
        action_mask=torch.ones(
            (
                2,
                3,
            ),
            dtype=torch.bool,
            device=device,
        ),
    )

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        buffer.sample(
            batch_size=2,
        )


