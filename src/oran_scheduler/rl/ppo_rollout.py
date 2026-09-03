from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PendingPPORolloutTransition:
    """
    PPO data available immediately when one 1LDS
    user-layer action is sampled.

    Primary reproduction interpretation:

        one transition
            =
        one joint all-RBG action for one user layer.

    The physical reward is deliberately absent here
    because it is not yet available when the action
    is sampled.

    Shapes:

        state:
            [state_size]

        actions:
            [RBG]

        action_mask:
            [RBG, action]

        old_log_prob_by_rbg:
            [RBG]

        old_joint_log_prob:
            scalar

        old_value:
            scalar
    """

    tti_index: int
    user_slot_index: int

    state: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor

    old_log_prob_by_rbg: torch.Tensor
    old_joint_log_prob: torch.Tensor

    old_value: torch.Tensor


@dataclass(frozen=True)
class PPORolloutTransition:
    """
    Fully resolved PPO transition.

    This object is created only after:

        1. the physical reward is known, and
        2. the actual successor RL state is known.

    For the last user slot of a TTI, next_state must
    be the first RL state of the next TTI unless the
    environment genuinely terminates.

    Shapes:

        state:
            [state_size]

        actions:
            [RBG]

        action_mask:
            [RBG, action]

        old_log_prob_by_rbg:
            [RBG]

        old_joint_log_prob:
            scalar

        reward_by_rbg:
            [RBG]

        reduced_reward:
            scalar

        old_value:
            scalar

        next_state:
            [state_size]

        terminated:
            scalar bool
    """

    tti_index: int
    user_slot_index: int

    state: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor

    old_log_prob_by_rbg: torch.Tensor
    old_joint_log_prob: torch.Tensor

    reward_by_rbg: torch.Tensor
    reduced_reward: torch.Tensor

    old_value: torch.Tensor
    next_state: torch.Tensor

    terminated: torch.Tensor


def _detached_clone(
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Store rollout data without retaining the
    actor/critic collection autograd graph.
    """

    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "Rollout values must be torch.Tensor "
            "objects."
        )

    return (
        value
        .detach()
        .clone()
    )

def validate_ppo_temporal_link(
    previous: PPORolloutTransition,
    current: PPORolloutTransition,
) -> None:
    """
    Validate continuity between two transitions from
    the SAME environment trajectory.

    For a nonterminal transition:

        previous.next_state
            must equal
        current.state

    A genuinely terminated transition may be
    followed by the first state of a new episode.

    IMPORTANT:
        This validator is meaningful only within one
        trajectory stream.

        It must never be used to compare transitions
        from two different gNB/cell streams.
    """

    if bool(
        previous
        .terminated
        .item()
    ):
        return

    if not torch.equal(
        previous.next_state,
        current.state,
    ):
        raise ValueError(
            "Nonterminal PPO transitions from the "
            "same stream must be temporally "
            "contiguous: previous.next_state must "
            "equal current.state."
        )

def build_pending_ppo_transition(
    *,
    tti_index: int,
    user_slot_index: int,
    state: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    old_log_prob_by_rbg: torch.Tensor,
    old_joint_log_prob: torch.Tensor,
    old_value: torch.Tensor,
) -> PendingPPORolloutTransition:
    """
    Snapshot the information available at action time.
    """

    if tti_index < 0:
        raise ValueError(
            "tti_index must be non-negative."
        )

    if user_slot_index < 0:
        raise ValueError(
            "user_slot_index must be non-negative."
        )

    if state.ndim != 1:
        raise ValueError(
            "state must have shape [state_size]."
        )

    if actions.ndim != 1:
        raise ValueError(
            "actions must have shape [RBG]."
        )

    if action_mask.ndim != 2:
        raise ValueError(
            "action_mask must have shape "
            "[RBG, action]."
        )

    num_rbgs = actions.shape[0]

    if action_mask.shape[0] != num_rbgs:
        raise ValueError(
            "actions and action_mask disagree on "
            "the number of RBGs."
        )

    if tuple(
        old_log_prob_by_rbg.shape
    ) != (
        num_rbgs,
    ):
        raise ValueError(
            "old_log_prob_by_rbg must have "
            "shape [RBG]."
        )

    if old_joint_log_prob.ndim != 0:
        raise ValueError(
            "old_joint_log_prob must be scalar."
        )

    if old_value.ndim != 0:
        raise ValueError(
            "old_value must be scalar."
        )

    if action_mask.dtype != torch.bool:
        raise ValueError(
            "action_mask must use torch.bool."
        )

    return PendingPPORolloutTransition(
        tti_index=tti_index,
        user_slot_index=user_slot_index,
        state=_detached_clone(
            state
        ),
        actions=_detached_clone(
            actions
        ),
        action_mask=_detached_clone(
            action_mask
        ),
        old_log_prob_by_rbg=(
            _detached_clone(
                old_log_prob_by_rbg
            )
        ),
        old_joint_log_prob=(
            _detached_clone(
                old_joint_log_prob
            )
        ),
        old_value=_detached_clone(
            old_value
        ),
    )


def finalize_ppo_transition(
    pending: PendingPPORolloutTransition,
    *,
    reward_by_rbg: torch.Tensor,
    reduced_reward: torch.Tensor,
    next_state: torch.Tensor,
    terminated: torch.Tensor,
) -> PPORolloutTransition:
    """
    Complete a pending PPO transition once the
    physical outcome and successor state are known.
    """

    num_rbgs = pending.actions.shape[0]

    if tuple(
        reward_by_rbg.shape
    ) != (
        num_rbgs,
    ):
        raise ValueError(
            "reward_by_rbg must have shape [RBG]."
        )

    if reduced_reward.ndim != 0:
        raise ValueError(
            "reduced_reward must be scalar."
        )

    if tuple(
        next_state.shape
    ) != tuple(
        pending.state.shape
    ):
        raise ValueError(
            "next_state must have the same shape "
            "as state."
        )

    if terminated.ndim != 0:
        raise ValueError(
            "terminated must be scalar."
        )

    if terminated.dtype != torch.bool:
        raise ValueError(
            "terminated must use torch.bool."
        )

    return PPORolloutTransition(
        tti_index=pending.tti_index,
        user_slot_index=(
            pending.user_slot_index
        ),
        state=_detached_clone(
            pending.state
        ),
        actions=_detached_clone(
            pending.actions
        ),
        action_mask=_detached_clone(
            pending.action_mask
        ),
        old_log_prob_by_rbg=(
            _detached_clone(
                pending.old_log_prob_by_rbg
            )
        ),
        old_joint_log_prob=(
            _detached_clone(
                pending.old_joint_log_prob
            )
        ),
        reward_by_rbg=_detached_clone(
            reward_by_rbg
        ),
        reduced_reward=_detached_clone(
            reduced_reward
        ),
        old_value=_detached_clone(
            pending.old_value
        ),
        next_state=_detached_clone(
            next_state
        ),
        terminated=_detached_clone(
            terminated
        ),
    )

@dataclass(frozen=True)
class PPORolloutBatch:
    """
    Stacked finalized PPO transitions.

    For N transitions:

        states:
            [N, state_size]

        actions:
            [N, RBG]

        action_masks:
            [N, RBG, action]

        old_log_prob_by_rbg:
            [N, RBG]

        old_joint_log_prob:
            [N]

        reward_by_rbg:
            [N, RBG]

        reduced_reward:
            [N]

        old_value:
            [N]

        next_states:
            [N, state_size]

        terminated:
            [N]
    """

    states: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor

    old_log_prob_by_rbg: torch.Tensor
    old_joint_log_prob: torch.Tensor

    reward_by_rbg: torch.Tensor
    reduced_reward: torch.Tensor

    old_value: torch.Tensor
    next_states: torch.Tensor

    terminated: torch.Tensor


def stack_ppo_transitions(
    transitions: list[
        PPORolloutTransition
    ],
) -> PPORolloutBatch:
    """
    Stack finalized transitions along one rollout
    dimension.
    """

    if len(transitions) == 0:
        raise ValueError(
            "At least one transition is required."
        )

    return PPORolloutBatch(
        states=torch.stack(
            [
                transition.state
                for transition in transitions
            ],
            dim=0,
        ),
        actions=torch.stack(
            [
                transition.actions
                for transition in transitions
            ],
            dim=0,
        ),
        action_masks=torch.stack(
            [
                transition.action_mask
                for transition in transitions
            ],
            dim=0,
        ),
        old_log_prob_by_rbg=(
            torch.stack(
                [
                    transition
                    .old_log_prob_by_rbg
                    for transition
                    in transitions
                ],
                dim=0,
            )
        ),
        old_joint_log_prob=torch.stack(
            [
                transition
                .old_joint_log_prob
                for transition
                in transitions
            ],
            dim=0,
        ),
        reward_by_rbg=torch.stack(
            [
                transition.reward_by_rbg
                for transition in transitions
            ],
            dim=0,
        ),
        reduced_reward=torch.stack(
            [
                transition.reduced_reward
                for transition in transitions
            ],
            dim=0,
        ),
        old_value=torch.stack(
            [
                transition.old_value
                for transition in transitions
            ],
            dim=0,
        ),
        next_states=torch.stack(
            [
                transition.next_state
                for transition in transitions
            ],
            dim=0,
        ),
        terminated=torch.stack(
            [
                transition.terminated
                for transition in transitions
            ],
            dim=0,
        ),
    )

