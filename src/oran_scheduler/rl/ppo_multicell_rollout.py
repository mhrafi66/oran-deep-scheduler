from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    compute_joint_action_log_prob,
)
from oran_scheduler.rl.ppo_multistream_buffer import (
    PPOMultiStreamTransitionBuffer,
)
from oran_scheduler.rl.ppo_rollout import (
    build_pending_ppo_transition,
)
from oran_scheduler.rl.ppo_tti_trajectory import (
    PPOCellTrajectoryCollector,
)
from oran_scheduler.state.one_lds_decision import (
    OneLDSDecisionData,
)


@dataclass(frozen=True)
class OneLDSPPOMultiCellRolloutConfig:
    """
    Per-cell trajectory configuration for
    centralized PPO training.

    PAPER-SPECIFIED training setting:
        num_user_slots = 4

    This controller does NOT perform PPO updates.

    Every cell owns an independent temporal
    trajectory, while all finalized transitions are
    sent to one shared multi-stream training buffer.
    """

    num_user_slots: int = 4

    def __post_init__(self) -> None:
        if self.num_user_slots <= 0:
            raise ValueError(
                "num_user_slots must be positive."
            )


class OneLDSPPOMultiCellRolloutController:
    """
    Own one cell's temporal PPO trajectory while
    using a shared actor and critic.

    One instance corresponds to exactly one
    centralized-training stream:

        stream_id == cell index

    Responsibilities:

        actor action sampling
            ->
        old-policy snapshot
            ->
        per-cell pending transitions
            ->
        delayed reward attachment
            ->
        cross-TTI transition resolution
            ->
        shared multi-stream buffer

    Deliberately NOT responsible for:

        PPO optimizer step
        GAE across cells
        expert-guidance optimizer step

    Those are owned by
    PPOMultiStreamTrainingCoordinator.
    """

    def __init__(
        self,
        *,
        stream_id: int,
        actor: OneLDSPPOActor,
        critic: nn.Module,
        transition_buffer: (
            PPOMultiStreamTransitionBuffer
        ),
        config: (
            OneLDSPPOMultiCellRolloutConfig
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = (
                OneLDSPPOMultiCellRolloutConfig()
            )

        if not (
            0
            <= stream_id
            < transition_buffer.config.num_streams
        ):
            raise ValueError(
                "stream_id is outside the shared "
                "multi-stream buffer."
            )

        self.stream_id = stream_id

        self.actor = actor

        self.critic = critic

        self.transition_buffer = (
            transition_buffer
        )

        self.config = config

        self.trajectory = (
            PPOCellTrajectoryCollector(
                num_user_slots=(
                    config.num_user_slots
                )
            )
        )


    @property
    def has_unresolved_tti_boundary(
        self,
    ) -> bool:
        return (
            self
            .trajectory
            .has_unresolved_boundary
        )


    @property
    def has_active_tti(
        self,
    ) -> bool:
        return (
            self
            .trajectory
            .has_active_tti
        )


    @property
    def agent_buffer_size(
        self,
    ) -> int:
        """
        Compatibility-style diagnostic.

        This is the size of the SHARED centralized
        transition buffer, not a private cell buffer.
        """

        return len(
            self.transition_buffer
        )


    def _drain_ready_transitions(
        self,
    ) -> None:
        transitions = (
            self
            .trajectory
            .pop_ready_transitions()
        )

        if len(
            transitions
        ) == 0:
            return

        self.transition_buffer.add_stream_transitions(
            stream_id=self.stream_id,
            transitions=transitions,
        )


    def begin_tti(
        self,
        *,
        tti_index: int,
        first_state: torch.Tensor,
    ) -> None:
        """
        Register this cell's first state for TTI t.

        This resolves the final transition from
        TTI t-1, if one exists.

        IMPORTANT:

            This method does NOT sample an action.

        The synchronized multi-cell runner calls
        begin_tti() for EVERY cell first.

        Only after all cells have done this may the
        shared PPO learner update and the new TTI
        actions be sampled.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if first_state.ndim != 1:
            raise ValueError(
                "first_state must have shape "
                "[state_size]."
            )

        self.trajectory.begin_tti(
            tti_index=tti_index,
            first_state=first_state,
        )

        self._drain_ready_transitions()


    def make_action_policy(
        self,
        *,
        tti_index: int,
    ) -> Callable[
        [
            int,
            OneLDSDecisionData,
        ],
        torch.Tensor,
    ]:
        """
        Return the action callback used AFTER the
        synchronized multi-cell boundary phase.

        Unlike the old single-cell controller,
        slot 0 does NOT begin the TTI here.

        begin_tti() must already have been called by
        the synchronized runner.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        def action_policy(
            user_slot_index: int,
            decision: OneLDSDecisionData,
        ) -> torch.Tensor:
            if not self.trajectory.has_active_tti:
                raise RuntimeError(
                    "Multi-cell PPO TTI must be "
                    "started with begin_tti() before "
                    "sampling any tracked action."
                )

            state = (
                decision
                .state_data
                .state
            )

            action_mask = (
                decision.action_mask
            )

            if state.ndim != 1:
                raise ValueError(
                    "PPO state must have shape "
                    "[state_size]."
                )

            if action_mask.ndim != 2:
                raise ValueError(
                    "PPO action mask must have shape "
                    "[RBG, action]."
                )

            with torch.no_grad():
                sampled = (
                    self.actor.sample_actions(
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

                value_batch = self.critic(
                    state.unsqueeze(
                        0
                    )
                )

            if tuple(
                value_batch.shape
            ) != (
                1,
            ):
                raise ValueError(
                    "PPO critic must return one "
                    "scalar per state."
                )

            actions = (
                sampled.actions[
                    0
                ]
            )

            old_log_prob_by_rbg = (
                sampled
                .log_prob_by_rbg[
                    0
                ]
            )

            old_joint_log_prob = (
                compute_joint_action_log_prob(
                    old_log_prob_by_rbg
                )
            )

            pending = build_pending_ppo_transition(
                tti_index=tti_index,
                user_slot_index=(
                    user_slot_index
                ),
                state=state,
                actions=actions,
                action_mask=action_mask,
                old_log_prob_by_rbg=(
                    old_log_prob_by_rbg
                ),
                old_joint_log_prob=(
                    old_joint_log_prob
                ),
                old_value=(
                    value_batch[
                        0
                    ]
                ),
            )

            self.trajectory.record_action(
                pending
            )

            return (
                actions
                .detach()
                .clone()
            )

        return action_policy
    

    def make_untracked_action_policy(
        self,
    ) -> Callable[
        [
            int,
            OneLDSDecisionData,
        ],
        torch.Tensor,
    ]:
        """
        Training-style stochastic actor sampling
        without creating rollout transitions.

        Used during TTIs 0..99.
        """

        def action_policy(
            user_slot_index: int,
            decision: OneLDSDecisionData,
        ) -> torch.Tensor:
            del user_slot_index

            state = (
                decision
                .state_data
                .state
            )

            action_mask = (
                decision.action_mask
            )

            with torch.no_grad():
                sampled = (
                    self.actor.sample_actions(
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

            return (
                sampled
                .actions[
                    0
                ]
                .detach()
                .clone()
            )

        return action_policy


    def finish_tti(
        self,
        *,
        reward_by_rbg: torch.Tensor,
        reduced_reward: torch.Tensor,
    ) -> None:
        """
        Attach the completed physical reward.

        The first L-1 transitions become ready.

        The last transition remains unresolved until
        begin_tti() receives the next TTI's slot-0
        state.
        """

        self.trajectory.finish_tti(
            reward_by_rbg=reward_by_rbg,
            reduced_reward=reduced_reward,
        )

        self._drain_ready_transitions()


    def finalize_episode(
        self,
        *,
        terminal_state: torch.Tensor,
    ) -> None:
        """
        Resolve the final transition only for a
        genuine environment terminal.
        """

        self.trajectory.finalize_episode(
            terminal_state=terminal_state,
        )

        self._drain_ready_transitions()


