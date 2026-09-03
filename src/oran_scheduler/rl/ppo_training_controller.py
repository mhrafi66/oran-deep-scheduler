from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    compute_joint_action_log_prob,
)
from oran_scheduler.rl.ppo_agent_buffer import (
    PPOAgentUpdateCoordinator,
    PPOAgentBufferUpdateResult,
)
from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
)
from oran_scheduler.rl.ppo_rollout import (
    build_pending_ppo_transition,
)
from oran_scheduler.rl.ppo_tti_trajectory import (
    PPOCellTrajectoryCollector,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizers,
)

from oran_scheduler.rl.ppo_expert_training import (
    PPOExpertGuidanceCoordinator,
)
from oran_scheduler.state.one_lds_decision import (
    OneLDSDecisionData,
)

@dataclass(frozen=True)
class OneLDSPPOTrainingControllerConfig:
    """
    Configuration for single-cell 1LDS PPO
    interaction.

    PAPER-SPECIFIED training setting:
        num_user_slots = 4

    OPEN-REPRODUCTION CONTROL RULE:
        For this single-cell controller, the PPO
        update size must be divisible by the number
        of user-slot decisions per TTI.

    This guarantees that an on-policy update can be
    performed at a TTI boundary rather than in the
    middle of a TTI.
    """

    num_user_slots: int = 4

    def __post_init__(self) -> None:
        if self.num_user_slots <= 0:
            raise ValueError(
                "num_user_slots must be positive."
            )


class OneLDSPPOTrainingController:
    """
    Connect the PPO learning machinery to the
    existing 1LDS scheduler action callback.

    One instance represents one temporally ordered
    cell trajectory.

    Responsibilities:

        1. sample actor actions;
        2. snapshot old policy probabilities;
        3. snapshot rollout critic values;
        4. create pending transitions;
        5. maintain layer/TTI temporal links;
        6. transfer finalized transitions into the
           PPO agent buffer;
        7. perform a PPO update at a safe TTI
           boundary when the buffer reaches M.

    It deliberately does NOT compute the wireless
    reward. The physical simulator/reward code
    remains responsible for that.
    """

    def __init__(
        self,
        *,
        actor: OneLDSPPOActor,
        critic: nn.Module,
        optimizers: PPOOptimizers,
        agent_buffer: PPOAgentUpdateCoordinator,
        gae_config: PPOGAEConfig,
        loss_config: PPOLossConfig,
        config: (
            OneLDSPPOTrainingControllerConfig
            | None
        ) = None,
        expert_guidance_coordinator: (
            PPOExpertGuidanceCoordinator
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = (
                OneLDSPPOTrainingControllerConfig()
            )

        self.actor = actor

        self.critic = critic

        self.optimizers = optimizers

        self.agent_buffer = agent_buffer

        self.gae_config = gae_config

        self.loss_config = loss_config

        self.config = config
        self.expert_guidance_coordinator = (
            expert_guidance_coordinator
        )
        update_size = (
            self.agent_buffer
            .config
            .update_size
        )

        if (
            update_size
            % config.num_user_slots
            != 0
        ):
            raise ValueError(
                "The current single-cell PPO "
                "controller requires update_size "
                "to be divisible by num_user_slots "
                "so updates occur at TTI "
                "boundaries."
            )

        self.trajectory = (
            PPOCellTrajectoryCollector(
                num_user_slots=(
                    config.num_user_slots
                )
            )
        )

        self._num_updates = 0

        self._completed_updates: list[
            PPOAgentBufferUpdateResult
        ] = []

    @property
    def num_updates(
        self,
    ) -> int:
        return self._num_updates


    @property
    def agent_buffer_size(
        self,
    ) -> int:
        return len(
            self.agent_buffer
        )


    @property
    def has_unresolved_tti_boundary(
        self,
    ) -> bool:
        return (
            self.trajectory
            .has_unresolved_boundary
        )

    @property
    def num_expert_guidance_updates(
        self,
    ) -> int:
        if (
            self
            .expert_guidance_coordinator
            is None
        ):
            return 0

        return (
            self
            .expert_guidance_coordinator
            .num_guidance_updates
        )

    def pop_update_results(
        self,
    ) -> tuple[
        PPOAgentBufferUpdateResult,
        ...
    ]:
        results = tuple(
            self._completed_updates
        )

        self._completed_updates = []

        return results


    def _drain_ready_transitions(
        self,
    ) -> None:
        transitions = (
            self.trajectory
            .pop_ready_transitions()
        )

        if len(transitions) == 0:
            return

        self.agent_buffer.add_transitions(
            transitions
        )


    def _update_if_ready(
        self,
    ) -> (
        PPOAgentBufferUpdateResult
        | None
    ):
        result = (
            self.agent_buffer
            .update_if_ready(
                actor=self.actor,
                critic=self.critic,
                optimizers=self.optimizers,
                gae_config=self.gae_config,
                loss_config=self.loss_config,
            )
        )

        if result is not None:
            self._num_updates += 1
            # NEW:
            self._perform_expert_guidance_after_ppo_update()

            # Only after this should execution return to
            # sampling the next PPO action.
            self._completed_updates.append(
                result
            )

        return result


    def _begin_tti_before_first_action(
        self,
        *,
        tti_index: int,
        first_state: torch.Tensor,
    ) -> None:
        """
        Observe s_(t,0), use it to resolve the
        previous TTI's final transition, and perform
        any pending PPO update BEFORE sampling
        a_(t,0).
        """

        self.trajectory.begin_tti(
            tti_index=tti_index,
            first_state=first_state,
        )

        self._drain_ready_transitions()

        self._update_if_ready()

    def _perform_expert_guidance_after_ppo_update(
        self,
    ) -> None:
        """
        Complete the second half of the paper's
        alternating optimization step.

        Must be called:

            AFTER PPO actor/critic update

        and:

            BEFORE the next action is sampled.
        """

        coordinator = (
            self
            .expert_guidance_coordinator
        )

        if coordinator is None:
            return

        coordinator.perform_after_ppo_update(
            ppo_update_index=(
                self.num_updates
            ),
            actor=self.actor,
            actor_optimizer=(
                self
                .optimizers
                .actor
            ),
        )


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
        Return an action callback compatible with:

            run_1lds_user_slot_loop(...)

        The returned callback performs rollout
        collection in addition to actor sampling.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        def action_policy(
            user_slot_index: int,
            decision: OneLDSDecisionData,
        ) -> torch.Tensor:
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
                    "Single-cell PPO state must "
                    "have shape [state_size]."
                )

            if action_mask.ndim != 2:
                raise ValueError(
                    "Single-cell PPO action mask "
                    "must have shape [RBG, action]."
                )

            if user_slot_index == 0:
                self._begin_tti_before_first_action(
                    tti_index=tti_index,
                    first_state=state,
                )

            elif not self.trajectory.has_active_tti:
                raise RuntimeError(
                    "The TTI must be started by "
                    "user slot 0 before later "
                    "user-slot actions."
                )

            state_batch = (
                state.unsqueeze(
                    0
                )
            )

            mask_batch = (
                action_mask.unsqueeze(
                    0
                )
            )

            #
            # Collection should not build an
            # autograd graph. PPO later recomputes
            # these quantities under the current
            # network during optimization.
            #
            with torch.no_grad():
                sampled = (
                    self.actor.sample_actions(
                        state=state_batch,
                        action_mask=mask_batch,
                    )
                )

                value_batch = self.critic(
                    state_batch
                )

            if tuple(
                value_batch.shape
            ) != (
                1,
            ):
                raise ValueError(
                    "The scalar PPO critic must "
                    "return shape [batch]."
                )

            actions = (
                sampled.actions[0]
            )

            old_log_prob_by_rbg = (
                sampled
                .log_prob_by_rbg[0]
            )

            old_joint_log_prob = (
                compute_joint_action_log_prob(
                    old_log_prob_by_rbg
                )
            )

            old_value = (
                value_batch[0]
            )

            pending = (
                build_pending_ppo_transition(
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
                    old_value=old_value,
                )
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
        Sample training-style actor actions without
        creating PPO rollout transitions.

        This is used during the initial simulator
        warm-up period.

        Environment evolution still occurs normally:

            actor action
                ->
            scheduling
                ->
            PHY
                ->
            traffic service
                ->
            throughput history
                ->
            next TTI

        But no PPO experience is collected.

        IMPORTANT:
            This still SAMPLES from the actor.

            It is not evaluation-mode argmax.
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

            if state.ndim != 1:
                raise ValueError(
                    "Single-cell PPO state must "
                    "have shape [state_size]."
                )

            if action_mask.ndim != 2:
                raise ValueError(
                    "Single-cell PPO action mask "
                    "must have shape [RBG, action]."
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
                .actions[0]
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
        Complete the currently active TTI after the
        physical outcome/reward has become known.

        Expected training shapes:

            reward_by_rbg:
                [4, 18]

            reduced_reward:
                [4]

        The first L-1 transitions become immediately
        ready.

        The final transition remains pending until
        the next TTI's slot-0 state exists.
        """

        self.trajectory.finish_tti(
            reward_by_rbg=reward_by_rbg,
            reduced_reward=reduced_reward,
        )

        self._drain_ready_transitions()

        #
        # Under the current exact-boundary
        # single-cell rule, the buffer must NOT
        # become full here.
        #
        # The final transition of this TTI has not
        # yet obtained its real next-TTI state.
        #
        if self.agent_buffer.is_full:
            raise RuntimeError(
                "PPO buffer became full before the "
                "TTI boundary transition was "
                "resolved. This violates the "
                "single-cell exact-boundary "
                "collection rule."
            )


    def finalize_episode(
        self,
        *,
        terminal_state: torch.Tensor,
    ) -> (
        PPOAgentBufferUpdateResult
        | None
    ):
        """
        Resolve the final transition when the
        simulation episode genuinely terminates.

        This method must NOT be used at a normal TTI
        boundary.
        """

        self.trajectory.finalize_episode(
            terminal_state=terminal_state,
        )

        self._drain_ready_transitions()

        return self._update_if_ready()


