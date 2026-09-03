import pytest
import torch

from oran_scheduler.rl.ppo_multistream_buffer import (
    PPOMultiStreamBufferConfig,
    PPOMultiStreamTransitionBuffer,
)
from oran_scheduler.rl.ppo_rollout import (
    PPORolloutTransition,
)


def preferred_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda:0"
        )

    return torch.device(
        "cpu"
    )


def build_transition(
    *,
    state_value: float,
    next_state_value: float,
    tti_index: int,
    user_slot_index: int,
    terminated: bool,
    device: torch.device,
) -> PPORolloutTransition:
    state = torch.full(
        (
            4,
        ),
        fill_value=state_value,
        dtype=torch.float32,
        device=device,
    )

    next_state = torch.full(
        (
            4,
        ),
        fill_value=next_state_value,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.tensor(
        [
            0,
        ],
        dtype=torch.long,
        device=device,
    )

    return PPORolloutTransition(
        tti_index=tti_index,
        user_slot_index=user_slot_index,
        state=state,
        actions=actions,
        action_mask=torch.ones(
            (
                1,
                3,
            ),
            dtype=torch.bool,
            device=device,
        ),
        old_log_prob_by_rbg=(
            torch.zeros(
                1,
                device=device,
            )
        ),
        old_joint_log_prob=(
            torch.tensor(
                0.0,
                device=device,
            )
        ),
        reward_by_rbg=torch.ones(
            1,
            device=device,
        ),
        reduced_reward=torch.tensor(
            1.0,
            device=device,
        ),
        old_value=torch.tensor(
            0.0,
            device=device,
        ),
        next_state=next_state,
        terminated=torch.tensor(
            terminated,
            dtype=torch.bool,
            device=device,
        ),
    )


def test_different_streams_are_temporally_independent():
    device = preferred_device()

    buffer = PPOMultiStreamTransitionBuffer(
        config=PPOMultiStreamBufferConfig(
            num_streams=2,
            update_size=128,
        )
    )

    #
    # Cell 0:
    #
    # 0 -> 1
    #
    buffer.add_transition(
        stream_id=0,
        transition=build_transition(
            state_value=0.0,
            next_state_value=1.0,
            tti_index=0,
            user_slot_index=0,
            terminated=False,
            device=device,
        ),
    )

    #
    # Cell 1 starts at state 100.
    #
    # This absolutely does NOT need to equal Cell
    # 0's next_state = 1.
    #
    buffer.add_transition(
        stream_id=1,
        transition=build_transition(
            state_value=100.0,
            next_state_value=101.0,
            tti_index=0,
            user_slot_index=0,
            terminated=False,
            device=device,
        ),
    )

    assert len(
        buffer
    ) == 2

    assert (
        buffer.num_transitions_for_stream(
            0
        )
        == 1
    )

    assert (
        buffer.num_transitions_for_stream(
            1
        )
        == 1
    )


def test_same_stream_must_remain_temporally_contiguous():
    device = preferred_device()

    buffer = PPOMultiStreamTransitionBuffer(
        config=PPOMultiStreamBufferConfig(
            num_streams=2,
        )
    )

    buffer.add_transition(
        stream_id=0,
        transition=build_transition(
            state_value=0.0,
            next_state_value=1.0,
            tti_index=0,
            user_slot_index=0,
            terminated=False,
            device=device,
        ),
    )

    with pytest.raises(
        ValueError,
        match="temporally contiguous",
    ):
        buffer.add_transition(
            stream_id=0,
            transition=build_transition(
                #
                # Should have been 1.0.
                #
                state_value=999.0,
                next_state_value=1000.0,
                tti_index=0,
                user_slot_index=1,
                terminated=False,
                device=device,
            ),
        )

    #
    # Failed insertion must not alter the buffer.
    #
    assert len(
        buffer
    ) == 1


def test_terminal_transition_allows_new_episode():
    device = preferred_device()

    buffer = PPOMultiStreamTransitionBuffer(
        config=PPOMultiStreamBufferConfig(
            num_streams=1,
        )
    )

    buffer.add_transition(
        stream_id=0,
        transition=build_transition(
            state_value=0.0,
            next_state_value=1.0,
            tti_index=0,
            user_slot_index=0,
            terminated=True,
            device=device,
        ),
    )

    #
    # New episode may start anywhere.
    #
    buffer.add_transition(
        stream_id=0,
        transition=build_transition(
            state_value=500.0,
            next_state_value=501.0,
            tti_index=1,
            user_slot_index=0,
            terminated=False,
            device=device,
        ),
    )

    assert len(
        buffer
    ) == 2


def test_multistream_buffer_exposes_update_boundary_crossing():
    device = preferred_device()

    buffer = PPOMultiStreamTransitionBuffer(
        config=PPOMultiStreamBufferConfig(
            num_streams=3,
            update_size=4,
        )
    )

    #
    # Stream 0 contributes 3.
    #
    transitions_0 = (
        build_transition(
            state_value=0.0,
            next_state_value=1.0,
            tti_index=0,
            user_slot_index=0,
            terminated=False,
            device=device,
        ),
        build_transition(
            state_value=1.0,
            next_state_value=2.0,
            tti_index=0,
            user_slot_index=1,
            terminated=False,
            device=device,
        ),
        build_transition(
            state_value=2.0,
            next_state_value=3.0,
            tti_index=0,
            user_slot_index=2,
            terminated=False,
            device=device,
        ),
    )

    buffer.add_stream_transitions(
        stream_id=0,
        transitions=transitions_0,
    )

    assert not (
        buffer.has_reached_update_size
    )

    #
    # Stream 1 contributes two more.
    #
    # Total becomes 5 while M = 4.
    #
    transitions_1 = (
        build_transition(
            state_value=100.0,
            next_state_value=101.0,
            tti_index=0,
            user_slot_index=0,
            terminated=False,
            device=device,
        ),
        build_transition(
            state_value=101.0,
            next_state_value=102.0,
            tti_index=0,
            user_slot_index=1,
            terminated=False,
            device=device,
        ),
    )

    buffer.add_stream_transitions(
        stream_id=1,
        transitions=transitions_1,
    )

    status = buffer.status()

    assert (
        status.num_transitions
        == 5
    )

    assert (
        status.update_size
        == 4
    )

    assert (
        status.has_reached_update_size
    )

    assert not (
        buffer.is_exactly_update_size
    )

    assert (
        status
        .num_transitions_beyond_update_size
        == 1
    )

    assert (
        status.num_transitions_by_stream
        == (
            3,
            2,
            0,
        )
    )


def test_paper_shaped_joint_action_count_exposes_84_per_tti():
    config = PPOMultiStreamBufferConfig(
        num_streams=21,
        update_size=128,
    )

    #
    # Current joint-action interpretation:
    #
    # 4 user-slot decisions
    # per cell
    #
    transitions_per_tti = (
        config.num_streams
        * 4
    )

    assert transitions_per_tti == 84

    #
    # One TTI:
    #
    #     84 < 128
    #
    assert (
        transitions_per_tti
        < config.update_size
    )

    #
    # Two TTIs:
    #
    #     168 > 128
    #
    assert (
        2 * transitions_per_tti
        > config.update_size
    )

    #
    # Therefore exact M=128 does not align with
    # synchronized whole-TTI collection under our
    # current interpretation.
    #


def test_multistream_buffer_uses_paper_update_size():
    config = PPOMultiStreamBufferConfig(
        num_streams=21,
    )

    assert config.update_size == 128