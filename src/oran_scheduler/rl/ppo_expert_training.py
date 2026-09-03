from dataclasses import dataclass

import torch
from torch.optim import Optimizer

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_expert_guidance import (
    PPOExpertGuidanceConfig,
    PPOExpertGuidanceUpdateData,
    perform_ppo_expert_guidance_update,
)

@dataclass(frozen=True)
class PPOExpertAlternatingUpdateData:
    """
    Diagnostics for one expert-guidance update that
    follows a completed PPO update.

    ppo_update_index:
        Which PPO update this guidance step belongs
        to.

        Example:

            PPO update 1
                ->
            expert update 1

            PPO update 2
                ->
            expert update 2
    """

    ppo_update_index: int

    expert_buffer_size: int

    guidance_update: PPOExpertGuidanceUpdateData


class PPOExpertGuidanceCoordinator:
    """
    Coordinate Teacher-2 updates with PPO updates.

    PAPER-SPECIFIED:
        Each alternating optimization step performs:

            1. PPO update
            2. expert-guidance update

    OPEN-REPRODUCTION CHOICE:
        The expert-guidance update currently reuses
        the PPO actor Adam optimizer.

    Why this choice is explicit:
        The public paper specifies the PPO actor
        optimizer and introduces lambda_JSD, but does
        not fully disclose whether expert guidance
        uses:

            - the same Adam optimizer state,
            - a separate Adam optimizer,
            - or a separate effective learning rate.

    We therefore keep this coordination layer
    isolated so that the optimizer interpretation can
    later be changed in a sensitivity experiment.
    """

    def __init__(
        self,
        *,
        expert_buffer: (
            PPOExpertDemonstrationBuffer
        ),
        guidance_config: (
            PPOExpertGuidanceConfig
        ),
        generator: (
            torch.Generator | None
        ) = None,
    ) -> None:
        self.expert_buffer = (
            expert_buffer
        )

        self.guidance_config = (
            guidance_config
        )

        self.generator = generator

        self._num_guidance_updates = 0

        self._last_update: (
            PPOExpertAlternatingUpdateData
            | None
        ) = None


    @property
    def num_guidance_updates(
        self,
    ) -> int:
        return (
            self._num_guidance_updates
        )


    @property
    def last_update(
        self,
    ) -> (
        PPOExpertAlternatingUpdateData
        | None
    ):
        return self._last_update


    def perform_after_ppo_update(
        self,
        *,
        ppo_update_index: int,
        actor: OneLDSPPOActor,
        actor_optimizer: Optimizer,
    ) -> PPOExpertAlternatingUpdateData:
        """
        Perform exactly one expert-guidance update
        immediately after a PPO update.

        The PPO update must already have completed.

        This method must run BEFORE another on-policy
        action is sampled.
        """

        if ppo_update_index <= 0:
            raise ValueError(
                "ppo_update_index must be positive."
            )

        expert_buffer_size = len(
            self.expert_buffer
        )

        required_batch_size = (
            self
            .guidance_config
            .batch_size
        )

        if (
            not self
            .expert_buffer
            .config
            .sampling_with_replacement
            and expert_buffer_size
            < required_batch_size
        ):
            raise RuntimeError(
                "PPO update is ready, but the "
                "expert buffer does not contain "
                "enough demonstrations for the "
                "configured expert-guidance "
                "mini-batch."
            )

        if expert_buffer_size == 0:
            raise RuntimeError(
                "PPO update is ready, but the "
                "expert buffer is empty."
            )

        guidance_update = (
            perform_ppo_expert_guidance_update(
                actor=actor,
                actor_optimizer=(
                    actor_optimizer
                ),
                expert_buffer=(
                    self.expert_buffer
                ),
                config=(
                    self.guidance_config
                ),
                generator=self.generator,
            )
        )

        self._num_guidance_updates += 1

        result = (
            PPOExpertAlternatingUpdateData(
                ppo_update_index=(
                    ppo_update_index
                ),
                expert_buffer_size=(
                    expert_buffer_size
                ),
                guidance_update=(
                    guidance_update
                ),
            )
        )

        self._last_update = result

        return result


    def perform_after_ppo_update(
        self,
        *,
        ppo_update_index: int,
        actor: OneLDSPPOActor,
        actor_optimizer: Optimizer,
    ) -> PPOExpertAlternatingUpdateData:
        """
        Perform exactly one expert-guidance update
        immediately after a PPO update.

        The PPO update must already have completed.

        This method must run BEFORE another on-policy
        action is sampled.
        """

        if ppo_update_index <= 0:
            raise ValueError(
                "ppo_update_index must be positive."
            )

        expert_buffer_size = (
            self.validate_ready_for_update()
        )

        required_batch_size = (
            self
            .guidance_config
            .batch_size
        )

        if (
            not self
            .expert_buffer
            .config
            .sampling_with_replacement
            and expert_buffer_size
            < required_batch_size
        ):
            raise RuntimeError(
                "PPO update is ready, but the "
                "expert buffer does not contain "
                "enough demonstrations for the "
                "configured expert-guidance "
                "mini-batch."
            )

        if expert_buffer_size == 0:
            raise RuntimeError(
                "PPO update is ready, but the "
                "expert buffer is empty."
            )

        guidance_update = (
            perform_ppo_expert_guidance_update(
                actor=actor,
                actor_optimizer=(
                    actor_optimizer
                ),
                expert_buffer=(
                    self.expert_buffer
                ),
                config=(
                    self.guidance_config
                ),
                generator=self.generator,
            )
        )

        self._num_guidance_updates += 1

        result = (
            PPOExpertAlternatingUpdateData(
                ppo_update_index=(
                    ppo_update_index
                ),
                expert_buffer_size=(
                    expert_buffer_size
                ),
                guidance_update=(
                    guidance_update
                ),
            )
        )

        self._last_update = result

        return result


    def validate_ready_for_update(
        self,
    ) -> int:
        """
        Validate that one expert-guidance update can
        be performed.

        This method has NO optimizer side effects.

        Returns:
            Current expert-buffer size.

        Why this exists:
            A centralized alternating update should
            detect predictable Teacher-2 failures
            BEFORE performing the PPO optimizer
            step.
        """

        expert_buffer_size = len(
            self.expert_buffer
        )

        required_batch_size = (
            self
            .guidance_config
            .batch_size
        )

        if expert_buffer_size == 0:
            raise RuntimeError(
                "PPO update is ready, but the "
                "expert buffer is empty."
            )

        if (
            not self
            .expert_buffer
            .config
            .sampling_with_replacement
            and expert_buffer_size
            < required_batch_size
        ):
            raise RuntimeError(
                "PPO update is ready, but the "
                "expert buffer does not contain "
                "enough demonstrations for the "
                "configured expert-guidance "
                "mini-batch."
            )

        return expert_buffer_size



