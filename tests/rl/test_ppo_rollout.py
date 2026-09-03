import torch

from oran_scheduler.rl.ppo_rollout import (
    build_pending_ppo_transition,
    finalize_ppo_transition,
    stack_ppo_transitions,
)


def test_pending_transition_detaches_collection_graph():
    state_source = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        requires_grad=True,
    )

    old_log_prob_source = torch.tensor(
        [
            -0.1,
            -0.2,
        ],
        requires_grad=True,
    )

    old_joint_log_prob = (
        old_log_prob_source.sum()
    )

    old_value = (
        2.0
        * state_source.sum()
    )

    pending = build_pending_ppo_transition(
        tti_index=3,
        user_slot_index=1,
        state=state_source,
        actions=torch.tensor(
            [
                0,
                2,
            ],
            dtype=torch.long,
        ),
        action_mask=torch.tensor(
            [
                [
                    True,
                    True,
                    True,
                ],
                [
                    True,
                    False,
                    True,
                ],
            ],
            dtype=torch.bool,
        ),
        old_log_prob_by_rbg=(
            old_log_prob_source
        ),
        old_joint_log_prob=(
            old_joint_log_prob
        ),
        old_value=old_value,
    )

    assert not pending.state.requires_grad

    assert (
        pending
        .old_log_prob_by_rbg
        .grad_fn
        is None
    )

    assert (
        pending
        .old_joint_log_prob
        .grad_fn
        is None
    )

    assert (
        pending
        .old_value
        .grad_fn
        is None
    )

def test_finalize_transition_attaches_delayed_outcome():
    pending = build_pending_ppo_transition(
        tti_index=10,
        user_slot_index=3,
        state=torch.zeros(
            4,
            dtype=torch.float32,
        ),
        actions=torch.tensor(
            [
                1,
                2,
            ],
            dtype=torch.long,
        ),
        action_mask=torch.ones(
            (
                2,
                3,
            ),
            dtype=torch.bool,
        ),
        old_log_prob_by_rbg=torch.tensor(
            [
                -0.4,
                -0.5,
            ],
            dtype=torch.float32,
        ),
        old_joint_log_prob=torch.tensor(
            -0.9,
            dtype=torch.float32,
        ),
        old_value=torch.tensor(
            0.25,
            dtype=torch.float32,
        ),
    )

    next_tti_state = torch.tensor(
        [
            4.0,
            5.0,
            6.0,
            7.0,
        ],
        dtype=torch.float32,
    )

    transition = finalize_ppo_transition(
        pending,
        reward_by_rbg=torch.tensor(
            [
                0.2,
                -0.2,
            ],
            dtype=torch.float32,
        ),
        reduced_reward=torch.tensor(
            0.0,
            dtype=torch.float32,
        ),
        next_state=next_tti_state,
        terminated=torch.tensor(
            False,
            dtype=torch.bool,
        ),
    )

    torch.testing.assert_close(
        transition.next_state,
        next_tti_state,
    )

    assert not bool(
        transition.terminated.item()
    )

    assert (
        transition.user_slot_index
        == 3
    )


def test_stack_ppo_transitions_preserves_joint_and_local_data():
    transitions = []

    for index in range(3):
        pending = build_pending_ppo_transition(
            tti_index=0,
            user_slot_index=index,
            state=torch.full(
                (
                    4,
                ),
                float(index),
            ),
            actions=torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.long,
            ),
            action_mask=torch.ones(
                (
                    2,
                    3,
                ),
                dtype=torch.bool,
            ),
            old_log_prob_by_rbg=torch.tensor(
                [
                    -0.1,
                    -0.2,
                ],
                dtype=torch.float32,
            ),
            old_joint_log_prob=torch.tensor(
                -0.3,
                dtype=torch.float32,
            ),
            old_value=torch.tensor(
                0.1 * index,
                dtype=torch.float32,
            ),
        )

        transitions.append(
            finalize_ppo_transition(
                pending,
                reward_by_rbg=torch.tensor(
                    [
                        0.2,
                        -0.2,
                    ],
                    dtype=torch.float32,
                ),
                reduced_reward=torch.tensor(
                    0.0,
                    dtype=torch.float32,
                ),
                next_state=torch.full(
                    (
                        4,
                    ),
                    float(index + 1),
                ),
                terminated=torch.tensor(
                    False,
                    dtype=torch.bool,
                ),
            )
        )

    batch = stack_ppo_transitions(
        transitions
    )

    assert tuple(
        batch.states.shape
    ) == (
        3,
        4,
    )

    assert tuple(
        batch.actions.shape
    ) == (
        3,
        2,
    )

    assert tuple(
        batch.action_masks.shape
    ) == (
        3,
        2,
        3,
    )

    assert tuple(
        batch.old_log_prob_by_rbg.shape
    ) == (
        3,
        2,
    )

    assert tuple(
        batch.old_joint_log_prob.shape
    ) == (
        3,
    )

    assert tuple(
        batch.reward_by_rbg.shape
    ) == (
        3,
        2,
    )

    assert tuple(
        batch.reduced_reward.shape
    ) == (
        3,
    )

    assert tuple(
        batch.old_value.shape
    ) == (
        3,
    )

    assert tuple(
        batch.next_states.shape
    ) == (
        3,
        4,
    )

    assert tuple(
        batch.terminated.shape
    ) == (
        3,
    )


