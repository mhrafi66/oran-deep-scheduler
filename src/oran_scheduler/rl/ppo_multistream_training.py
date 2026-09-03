from dataclasses import dataclass

import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
)
from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)
from oran_scheduler.rl.ppo_expert_training import (
    PPOExpertAlternatingUpdateData,
    PPOExpertGuidanceCoordinator,
)
from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
)
from oran_scheduler.rl.ppo_multistream_buffer import (
    PPOMultiStreamTransitionBuffer,
)
from oran_scheduler.rl.ppo_multistream_update import (
    PPOMultiStreamPreparedUpdate,
    PPOMultiStreamUpdateConfig,
    PPOPreparedStreamDiagnostics,
    prepare_multistream_ppo_update_if_ready,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizers,
    PPOUpdateResult,
    perform_ppo_update,
)


@dataclass(frozen=True)
class PPOMultiStreamTrainingUpdateResult:
    """
    Diagnostics from one complete centralized
    alternating training update.

    One successful update consists of:

        Teacher 1:
            PPO actor + critic update

        optionally followed by:

        Teacher 2:
            expert-guidance actor-only update

    The multi-stream rollout buffer is cleared only
    after all enabled update stages succeed.
    """

    ppo_update_index: int

    num_real_transitions: int

    num_optimizer_samples: int

    num_transitions_by_stream: tuple[
        int,
        ...
    ]

    stream_diagnostics: tuple[
        PPOPreparedStreamDiagnostics,
        ...
    ]

    ppo_update: PPOUpdateResult

    expert_update: (
        PPOExpertAlternatingUpdateData
        | None
    )


class PPOMultiStreamTrainingCoordinator:
    """
    Own one centralized PPO update process fed by
    several independent temporal streams.

    Responsibilities:

        multi-cell rollout buffer
            ->
        independent per-stream GAE
            ->
        centralized optimizer batch
            ->
        candidate permutation augmentation
            ->
        PPO update
            ->
        expert-guidance update
            ->
        clear rollout streams

    IMPORTANT:
        No new environment action should be sampled
        between the PPO update and the optional
        expert-guidance update.
    """

    def __init__(
        self,
        *,
        transition_buffer: (
            PPOMultiStreamTransitionBuffer
        ),
        update_config: (
            PPOMultiStreamUpdateConfig
        ),
        gae_config: PPOGAEConfig,
        loss_config: PPOLossConfig,
        candidate_augmentation_config: (
            PPOCandidateAugmentationConfig
            | None
        ) = None,
        candidate_augmentation_generator: (
            torch.Generator | None
        ) = None,
        expert_guidance_coordinator: (
            PPOExpertGuidanceCoordinator
            | None
        ) = None,
    ) -> None:
        self.transition_buffer = (
            transition_buffer
        )

        self.update_config = (
            update_config
        )

        self.gae_config = gae_config

        self.loss_config = loss_config

        self.candidate_augmentation_config = (
            candidate_augmentation_config
        )

        self.candidate_augmentation_generator = (
            candidate_augmentation_generator
        )

        self.expert_guidance_coordinator = (
            expert_guidance_coordinator
        )

        self._num_updates = 0

        self._last_update: (
            PPOMultiStreamTrainingUpdateResult
            | None
        ) = None


    @property
    def num_updates(
        self,
    ) -> int:
        return self._num_updates


    @property
    def num_expert_guidance_updates(
        self,
    ) -> int:
        coordinator = (
            self
            .expert_guidance_coordinator
        )

        if coordinator is None:
            return 0

        return (
            coordinator
            .num_guidance_updates
        )


    @property
    def last_update(
        self,
    ) -> (
        PPOMultiStreamTrainingUpdateResult
        | None
    ):
        return self._last_update


    def _prepare_update_if_ready(
        self,
        *,
        actor: OneLDSPPOActor,
        critic: nn.Module,
    ) -> (
        PPOMultiStreamPreparedUpdate
        | None
    ):
        """
        Prepare centralized optimizer data without
        changing model parameters.
        """

        return (
            prepare_multistream_ppo_update_if_ready(
                buffer=(
                    self.transition_buffer
                ),
                actor=actor,
                critic=critic,
                gae_config=(
                    self.gae_config
                ),
                update_config=(
                    self.update_config
                ),
                candidate_augmentation_config=(
                    self
                    .candidate_augmentation_config
                ),
                candidate_augmentation_generator=(
                    self
                    .candidate_augmentation_generator
                ),
            )
        )


    def update_if_ready(
        self,
        *,
        actor: OneLDSPPOActor,
        critic: nn.Module,
        optimizers: PPOOptimizers,
    ) -> (
        PPOMultiStreamTrainingUpdateResult
        | None
    ):
        """
        Perform one complete centralized alternating
        update when enough on-policy data exist.

        Sequence:

            1. prepare independent per-stream GAE

            2. construct centralized PPO batch

            3. perform candidate augmentation

            4. preflight expert-guidance readiness

            5. PPO update:
                   actor + critic

            6. expert-guidance update:
                   actor only

            7. clear all consumed trajectory streams

        If the centralized buffer has not reached its
        configured update boundary, return None.
        """

        prepared = (
            self._prepare_update_if_ready(
                actor=actor,
                critic=critic,
            )
        )

        if prepared is None:
            return None

        expert_coordinator = (
            self
            .expert_guidance_coordinator
        )

        #
        # ------------------------------------------------------
        # Preflight Teacher 2 BEFORE Teacher 1 changes anything.
        # ------------------------------------------------------
        #
        if expert_coordinator is not None:
            expert_coordinator.validate_ready_for_update()

        #
        # ------------------------------------------------------
        # TEACHER 1
        #
        # PPO actor + critic update.
        # ------------------------------------------------------
        #
        ppo_update = perform_ppo_update(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            batch=(
                prepared
                .optimization
                .batch
            ),
            advantage=(
                prepared
                .optimization
                .advantage
            ),
            target_return=(
                prepared
                .optimization
                .target_return
            ),
            loss_config=(
                self.loss_config
            ),
        )

        next_update_index = (
            self._num_updates
            + 1
        )

        #
        # ------------------------------------------------------
        # TEACHER 2
        #
        # Must happen after PPO and before the next
        # on-policy action is sampled.
        # ------------------------------------------------------
        #
        expert_update = None

        if expert_coordinator is not None:
            expert_update = (
                expert_coordinator
                .perform_after_ppo_update(
                    ppo_update_index=(
                        next_update_index
                    ),
                    actor=actor,
                    actor_optimizer=(
                        optimizers.actor
                    ),
                )
            )

        #
        # ------------------------------------------------------
        # Only now is this rollout fully consumed.
        #
        # We deliberately do NOT clear before the
        # enabled optimizer stages succeed.
        # ------------------------------------------------------
        #
        self.transition_buffer.clear()

        self._num_updates = (
            next_update_index
        )

        result = (
            PPOMultiStreamTrainingUpdateResult(
                ppo_update_index=(
                    next_update_index
                ),
                num_real_transitions=(
                    prepared
                    .num_real_transitions
                ),
                num_optimizer_samples=(
                    prepared
                    .num_optimizer_samples
                ),
                num_transitions_by_stream=(
                    prepared
                    .num_transitions_by_stream
                ),
                stream_diagnostics=(
                    prepared
                    .stream_diagnostics
                ),
                ppo_update=(
                    ppo_update
                ),
                expert_update=(
                    expert_update
                ),
            )
        )

        self._last_update = result

        return result


