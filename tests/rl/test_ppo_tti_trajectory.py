import pytest
import torch

from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_rollout import (
    build_pending_ppo_transition,
)
from oran_scheduler.rl.ppo_tti_trajectory import (
    PPOCellTrajectoryCollector,
    prepare_ppo_trajectory_with_gae,
)


def build_pending(
    *,
    tti_index: int,
    user_slot_index: int,
    state_value: float,
) -> object:
    state = torch.tensor(
        [
            state_value,
        ],
        dtype=torch.float32,
    )

    return build_pending_ppo_transition(
        tti_index=tti_index,
        user_slot_index=user_slot_index,
        state=state,
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
            state_value,
            dtype=torch.float32,
        ),
    )


def test_tti_collector_links_internal_slots_and_delays_last_slot():
    collector = PPOCellTrajectoryCollector(
        num_user_slots=2,
    )

    first_state = torch.tensor(
        [
            0.0,
        ],
        dtype=torch.float32,
    )

    collector.begin_tti(
        tti_index=0,
        first_state=first_state,
    )

    collector.record_action(
        build_pending(
            tti_index=0,
            user_slot_index=0,
            state_value=0.0,
        )
    )

    collector.record_action(
        build_pending(
            tti_index=0,
            user_slot_index=1,
            state_value=1.0,
        )
    )

    collector.finish_tti(
        reward_by_rbg=torch.tensor(
            [
                [
                    0.2,
                    -0.2,
                ],
                [
                    0.2,
                    0.2,
                ],
            ],
            dtype=torch.float32,
        ),
        reduced_reward=torch.tensor(
            [
                0.0,
                0.2,
            ],
            dtype=torch.float32,
        ),
    )

    assert collector.num_ready == 1

    assert (
        collector.has_unresolved_boundary
    )

    internal = (
        collector
        .pop_ready_transitions()
    )

    assert len(internal) == 1

    torch.testing.assert_close(
        internal[0].state,
        torch.tensor(
            [
                0.0,
            ]
        ),
    )

    torch.testing.assert_close(
        internal[0].next_state,
        torch.tensor(
            [
                1.0,
            ]
        ),
    )

    assert not bool(
        internal[0].terminated.item()
    )


def test_next_tti_state_resolves_previous_final_slot():
    collector = PPOCellTrajectoryCollector(
        num_user_slots=2,
    )

    collector.begin_tti(
        tti_index=0,
        first_state=torch.tensor(
            [
                0.0,
            ]
        ),
    )

    collector.record_action(
        build_pending(
            tti_index=0,
            user_slot_index=0,
            state_value=0.0,
        )
    )

    collector.record_action(
        build_pending(
            tti_index=0,
            user_slot_index=1,
            state_value=1.0,
        )
    )

    collector.finish_tti(
        reward_by_rbg=torch.zeros(
            (
                2,
                2,
            ),
            dtype=torch.float32,
        ),
        reduced_reward=torch.zeros(
            2,
            dtype=torch.float32,
        ),
    )

    collector.pop_ready_transitions()

    next_tti_state = torch.tensor(
        [
            2.0,
        ],
        dtype=torch.float32,
    )

    collector.begin_tti(
        tti_index=1,
        first_state=next_tti_state,
    )

    completed = (
        collector
        .pop_ready_transitions()
    )

    assert len(completed) == 1

    assert (
        completed[0].user_slot_index
        == 1
    )

    torch.testing.assert_close(
        completed[0].next_state,
        next_tti_state,
    )

    assert not bool(
        completed[0].terminated.item()
    )


def test_episode_termination_marks_only_final_transition_terminal():
    collector = PPOCellTrajectoryCollector(
        num_user_slots=2,
    )

    collector.begin_tti(
        tti_index=5,
        first_state=torch.tensor(
            [
                5.0,
            ]
        ),
    )

    collector.record_action(
        build_pending(
            tti_index=5,
            user_slot_index=0,
            state_value=5.0,
        )
    )

    collector.record_action(
        build_pending(
            tti_index=5,
            user_slot_index=1,
            state_value=6.0,
        )
    )

    collector.finish_tti(
        reward_by_rbg=torch.zeros(
            (
                2,
                2,
            ),
            dtype=torch.float32,
        ),
        reduced_reward=torch.zeros(
            2,
            dtype=torch.float32,
        ),
    )

    first_part = (
        collector
        .pop_ready_transitions()
    )

    assert len(first_part) == 1

    assert not bool(
        first_part[0]
        .terminated
        .item()
    )

    collector.finalize_episode(
        terminal_state=torch.tensor(
            [
                7.0,
            ],
            dtype=torch.float32,
        )
    )

    last_part = (
        collector
        .pop_ready_transitions()
    )

    assert len(last_part) == 1

    assert bool(
        last_part[0]
        .terminated
        .item()
    )


class FirstFeatureCritic(
    torch.nn.Module
):
    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return state[
            :,
            0,
        ]

def test_gae_uses_real_next_tti_bootstrap_value():
    collector = PPOCellTrajectoryCollector(
        num_user_slots=2,
    )

    collector.begin_tti(
        tti_index=0,
        first_state=torch.tensor(
            [
                0.0,
            ],
            dtype=torch.float32,
        ),
    )

    collector.record_action(
        build_pending(
            tti_index=0,
            user_slot_index=0,
            state_value=0.0,
        )
    )

    collector.record_action(
        build_pending(
            tti_index=0,
            user_slot_index=1,
            state_value=1.0,
        )
    )

    collector.finish_tti(
        reward_by_rbg=torch.zeros(
            (
                2,
                2,
            ),
            dtype=torch.float32,
        ),
        reduced_reward=torch.zeros(
            2,
            dtype=torch.float32,
        ),
    )

    internal = list(
        collector
        .pop_ready_transitions()
    )

    collector.begin_tti(
        tti_index=1,
        first_state=torch.tensor(
            [
                2.0,
            ],
            dtype=torch.float32,
        ),
    )

    boundary = list(
        collector
        .pop_ready_transitions()
    )

    transitions = (
        internal
        + boundary
    )

    result = (
        prepare_ppo_trajectory_with_gae(
            transitions=transitions,
            critic=FirstFeatureCritic(),
            config=PPOGAEConfig(
                gamma=1.0,
                gae_lambda=0.0,
            ),
        )
    )

    torch.testing.assert_close(
        result.bootstrap_value,
        torch.tensor(
            2.0,
            dtype=torch.float32,
        ),
    )

    expected_td_residual = torch.tensor(
        [
            1.0,
            1.0,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        result.gae.td_residual,
        expected_td_residual,
    )

def test_record_action_rejects_out_of_order_user_slots():
    collector = PPOCellTrajectoryCollector(
        num_user_slots=2,
    )

    collector.begin_tti(
        tti_index=0,
        first_state=torch.tensor(
            [
                0.0,
            ],
            dtype=torch.float32,
        ),
    )

    with pytest.raises(
        ValueError
    ):
        collector.record_action(
            build_pending(
                tti_index=0,
                user_slot_index=1,
                state_value=0.0,
            )
        )



