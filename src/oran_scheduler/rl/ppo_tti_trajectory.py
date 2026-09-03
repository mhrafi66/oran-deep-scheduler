from collections.abc import Callable
from dataclasses import dataclass

import torch

from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
    PPOGAEResult,
    compute_ppo_gae,
)
from oran_scheduler.rl.ppo_rollout import (
    PPORolloutBatch,
    PPORolloutTransition,
    PendingPPORolloutTransition,
    finalize_ppo_transition,
    stack_ppo_transitions,
)

ValueFunction = Callable[
    [torch.Tensor],
    torch.Tensor,
]


@dataclass(frozen=True)
class _AwaitingTTIBoundary:
    """
    Final user-slot transition of a completed TTI.

    Its reward is already known, but its next_state
    cannot be resolved until the first RL state of
    the next TTI is observed.
    """

    pending: PendingPPORolloutTransition

    reward_by_rbg: torch.Tensor

    reduced_reward: torch.Tensor


@dataclass(frozen=True)
class PPOPreparedTrajectory:
    """
    One temporally ordered rollout together with
    the GAE targets derived from it.

    bootstrap_value:
        V(s_next) for the state following the final
        stored transition.

        If the final transition genuinely terminates
        the environment, this is zero.

    batch:
        Stacked rollout data.

    gae:
        TD residuals, advantages, and target returns.
    """

    batch: PPORolloutBatch

    bootstrap_value: torch.Tensor

    gae: PPOGAEResult


class PPOCellTrajectoryCollector:
    """
    Build temporally correct PPO transitions for
    one cell.

    Primary reproduction interpretation:

        one PPO transition
            =
        one complete 1LDS user-layer decision
        across all RBGs.

    This collector deliberately keeps the last
    transition of each TTI unresolved until the
    first state of the next TTI is observed.

    It does NOT decide:
        - PPO update-buffer size,
        - update cadence,
        - minibatching,
        - GAE lambda,
        - expert/JSD behavior.

    Those remain separate training concerns.
    """

    def __init__(
        self,
        *,
        num_user_slots: int,
    ) -> None:
        if num_user_slots <= 0:
            raise ValueError(
                "num_user_slots must be positive."
            )

        self.num_user_slots = (
            num_user_slots
        )

        self._active_tti_index: (
            int | None
        ) = None

        self._active_first_state: (
            torch.Tensor | None
        ) = None

        self._active_pending: list[
            PendingPPORolloutTransition
        ] = []

        self._awaiting_boundary: (
            _AwaitingTTIBoundary | None
        ) = None

        self._ready: list[
            PPORolloutTransition
        ] = []


    @staticmethod
    def _snapshot(
        value: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                "Trajectory values must be "
                "torch.Tensor objects."
            )

        return (
            value
            .detach()
            .clone()
        )


    @property
    def num_ready(self) -> int:
        return len(
            self._ready
        )


    @property
    def has_unresolved_boundary(
        self,
    ) -> bool:
        return (
            self._awaiting_boundary
            is not None
        )


    @property
    def has_active_tti(
        self,
    ) -> bool:
        return (
            self._active_tti_index
            is not None
        )


    def begin_tti(
        self,
        *,
        tti_index: int,
        first_state: torch.Tensor,
    ) -> None:
        """
        Observe the first RL state of a new TTI.

        IMPORTANT:

        This method does NOT sample or store the new
        TTI's first action.

        Therefore, if observing this state completes
        a PPO rollout, the caller can perform a PPO
        update before sampling the new action.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if self.has_active_tti:
            raise RuntimeError(
                "Cannot begin a new TTI before the "
                "current TTI is finished."
            )

        if first_state.ndim != 1:
            raise ValueError(
                "first_state must have shape "
                "[state_size]."
            )

        if not torch.is_floating_point(
            first_state
        ):
            raise ValueError(
                "first_state must use a "
                "floating-point dtype."
            )

        if not torch.isfinite(
            first_state
        ).all():
            raise ValueError(
                "first_state contains non-finite "
                "values."
            )

        if self._awaiting_boundary is not None:
            boundary = (
                self._awaiting_boundary
            )

            expected_tti_index = (
                boundary
                .pending
                .tti_index
                + 1
            )

            if tti_index != expected_tti_index:
                raise ValueError(
                    "The next TTI index must directly "
                    "follow the TTI awaiting its "
                    "successor state."
                )

            terminated = torch.tensor(
                False,
                dtype=torch.bool,
                device=first_state.device,
            )

            completed = finalize_ppo_transition(
                boundary.pending,
                reward_by_rbg=(
                    boundary.reward_by_rbg
                ),
                reduced_reward=(
                    boundary.reduced_reward
                ),
                next_state=first_state,
                terminated=terminated,
            )

            self._ready.append(
                completed
            )

            self._awaiting_boundary = None

        self._active_tti_index = (
            tti_index
        )

        self._active_first_state = (
            self._snapshot(
                first_state
            )
        )

        self._active_pending = []


    def record_action(
        self,
        pending: PendingPPORolloutTransition,
    ) -> None:
        """
        Record one user-layer action from the
        currently active TTI.
        """

        if not self.has_active_tti:
            raise RuntimeError(
                "begin_tti() must be called before "
                "record_action()."
            )

        if self._awaiting_boundary is not None:
            raise RuntimeError(
                "Previous TTI boundary must be "
                "resolved before recording a new "
                "action."
            )

        if pending.tti_index != (
            self._active_tti_index
        ):
            raise ValueError(
                "Pending transition belongs to the "
                "wrong TTI."
            )

        expected_user_slot = len(
            self._active_pending
        )

        if expected_user_slot >= (
            self.num_user_slots
        ):
            raise RuntimeError(
                "All configured user slots have "
                "already been recorded."
            )

        if pending.user_slot_index != (
            expected_user_slot
        ):
            raise ValueError(
                "User-slot transitions must be "
                "recorded in temporal order."
            )

        if (
            pending.state.shape
            != self._active_first_state.shape
        ):
            raise ValueError(
                "All states in one trajectory must "
                "use the same state shape."
            )

        if expected_user_slot == 0:
            if not torch.equal(
                pending.state,
                self._active_first_state,
            ):
                raise ValueError(
                    "The first action state must "
                    "match the state supplied to "
                    "begin_tti()."
                )

        self._active_pending.append(
            pending
        )

    def finish_tti(
        self,
        *,
        reward_by_rbg: torch.Tensor,
        reduced_reward: torch.Tensor,
    ) -> None:
        """
        Attach the physical rewards for the active
        TTI.

        reward_by_rbg:
            Shape [user_slot, RBG].

        reduced_reward:
            Shape [user_slot].

        All transitions except the final user slot
        can now be finalized because their successor
        state is the next user slot's stored state.

        The final user slot remains unresolved until
        begin_tti() receives the next TTI's first
        state.
        """

        if not self.has_active_tti:
            raise RuntimeError(
                "No active TTI to finish."
            )

        if self._awaiting_boundary is not None:
            raise RuntimeError(
                "An older TTI is still awaiting its "
                "successor state."
            )

        if len(
            self._active_pending
        ) != self.num_user_slots:
            raise RuntimeError(
                "Cannot finish the TTI before all "
                "user-slot actions are recorded."
            )

        first_pending = (
            self._active_pending[0]
        )

        num_rbgs = int(
            first_pending.actions.shape[0]
        )

        expected_reward_shape = (
            self.num_user_slots,
            num_rbgs,
        )

        if tuple(
            reward_by_rbg.shape
        ) != expected_reward_shape:
            raise ValueError(
                "reward_by_rbg must have shape "
                "[user_slot, RBG]."
            )

        if tuple(
            reduced_reward.shape
        ) != (
            self.num_user_slots,
        ):
            raise ValueError(
                "reduced_reward must have shape "
                "[user_slot]."
            )

        if reward_by_rbg.device != (
            first_pending.state.device
        ):
            raise ValueError(
                "reward_by_rbg must be on the same "
                "device as the rollout state."
            )

        if reduced_reward.device != (
            first_pending.state.device
        ):
            raise ValueError(
                "reduced_reward must be on the same "
                "device as the rollout state."
            )

        for user_slot_index in range(
            self.num_user_slots - 1
        ):
            pending = (
                self._active_pending[
                    user_slot_index
                ]
            )

            next_pending = (
                self._active_pending[
                    user_slot_index + 1
                ]
            )

            terminated = torch.tensor(
                False,
                dtype=torch.bool,
                device=pending.state.device,
            )

            completed = finalize_ppo_transition(
                pending,
                reward_by_rbg=(
                    reward_by_rbg[
                        user_slot_index
                    ]
                ),
                reduced_reward=(
                    reduced_reward[
                        user_slot_index
                    ]
                ),
                next_state=(
                    next_pending.state
                ),
                terminated=terminated,
            )

            self._ready.append(
                completed
            )

        final_pending = (
            self._active_pending[-1]
        )

        self._awaiting_boundary = (
            _AwaitingTTIBoundary(
                pending=final_pending,
                reward_by_rbg=(
                    self._snapshot(
                        reward_by_rbg[-1]
                    )
                ),
                reduced_reward=(
                    self._snapshot(
                        reduced_reward[-1]
                    )
                ),
            )
        )

        self._active_tti_index = None
        self._active_first_state = None
        self._active_pending = []


    def finalize_episode(
        self,
        *,
        terminal_state: torch.Tensor,
    ) -> None:
        """
        Resolve the final transition when the
        environment genuinely terminates.

        A normal TTI boundary must NOT use this
        method.
        """

        if self.has_active_tti:
            raise RuntimeError(
                "finish_tti() must be called before "
                "finalize_episode()."
            )

        if self._awaiting_boundary is None:
            raise RuntimeError(
                "No unresolved final transition "
                "exists."
            )

        boundary = (
            self._awaiting_boundary
        )

        terminated = torch.tensor(
            True,
            dtype=torch.bool,
            device=terminal_state.device,
        )

        completed = finalize_ppo_transition(
            boundary.pending,
            reward_by_rbg=(
                boundary.reward_by_rbg
            ),
            reduced_reward=(
                boundary.reduced_reward
            ),
            next_state=terminal_state,
            terminated=terminated,
        )

        self._ready.append(
            completed
        )

        self._awaiting_boundary = None


    def pop_ready_transitions(
        self,
    ) -> tuple[
        PPORolloutTransition,
        ...
    ]:
        """
        Return all fully resolved transitions and
        clear the ready queue.
        """

        transitions = tuple(
            self._ready
        )

        self._ready = []

        return transitions


def _validate_temporal_links(
    batch: PPORolloutBatch,
) -> None:
    """
    Verify that consecutive nonterminal transitions
    actually form one temporal trajectory.
    """

    num_transitions = (
        batch.states.shape[0]
    )

    for transition_index in range(
        num_transitions - 1
    ):
        if bool(
            batch.terminated[
                transition_index
            ].item()
        ):
            continue

        stored_next_state = (
            batch.next_states[
                transition_index
            ]
        )

        following_state = (
            batch.states[
                transition_index + 1
            ]
        )

        if not torch.equal(
            stored_next_state,
            following_state,
        ):
            raise ValueError(
                "Rollout transitions are not "
                "temporally contiguous. A "
                "nonterminal transition's next_state "
                "must equal the following stored "
                "state."
            )
        

def prepare_ppo_trajectory_with_gae(
    *,
    transitions: list[
        PPORolloutTransition
    ]
    | tuple[
        PPORolloutTransition,
        ...
    ],
    critic: ValueFunction,
    config: PPOGAEConfig,
) -> PPOPreparedTrajectory:
    """
    Stack one temporally ordered trajectory and
    compute correctly bootstrapped scalar/joint GAE.

    IMPORTANT:

    This must be called before the critic is updated.
    The stored old_value values and the final
    bootstrap value should correspond to the same
    rollout critic.
    """

    if len(transitions) == 0:
        raise ValueError(
            "At least one transition is required."
        )

    batch = stack_ppo_transitions(
        list(
            transitions
        )
    )

    _validate_temporal_links(
        batch
    )

    final_is_terminal = bool(
        batch.terminated[-1].item()
    )

    if final_is_terminal:
        bootstrap_value = torch.zeros(
            (),
            dtype=batch.old_value.dtype,
            device=batch.old_value.device,
        )

    else:
        final_next_state = (
            batch.next_states[-1]
            .unsqueeze(0)
        )

        with torch.no_grad():
            bootstrap_batch = critic(
                final_next_state
            )

        if tuple(
            bootstrap_batch.shape
        ) != (
            1,
        ):
            raise ValueError(
                "critic must return shape [batch]."
            )

        bootstrap_value = (
            bootstrap_batch[0]
            .detach()
            .clone()
        )

        if bootstrap_value.device != (
            batch.old_value.device
        ):
            raise ValueError(
                "Critic bootstrap value must be on "
                "the rollout device."
            )

        if bootstrap_value.dtype != (
            batch.old_value.dtype
        ):
            raise ValueError(
                "Critic bootstrap value must use "
                "the rollout dtype."
            )

    gae = compute_ppo_gae(
        rewards=batch.reduced_reward,
        values=batch.old_value,
        next_value=bootstrap_value,
        terminated=batch.terminated,
        config=config,
    )

    return PPOPreparedTrajectory(
        batch=batch,
        bootstrap_value=bootstrap_value,
        gae=gae,
    )



