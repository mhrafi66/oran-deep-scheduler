from dataclasses import dataclass

import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
)
from oran_scheduler.rl.ppo_gae import (
    PPOGAEConfig,
)
from oran_scheduler.rl.ppo_loss import (
    PPOLossConfig,
)
from oran_scheduler.rl.ppo_rollout import (
    PPORolloutTransition,
    validate_ppo_temporal_link,
)
from oran_scheduler.rl.ppo_tti_trajectory import (
    prepare_ppo_trajectory_with_gae,
)
from oran_scheduler.rl.ppo_update import (
    PPOOptimizers,
    PPOUpdateResult,
    perform_ppo_update,
)

from oran_scheduler.rl.ppo_agent_augmentation import (
    build_candidate_augmented_ppo_update,
)
from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
)

@dataclass(frozen=True)
class PPOAgentBufferConfig:
    """
    Temporary on-policy PPO agent buffer.

    PAPER-SPECIFIED:
        update_size = 128

    Under the current primary reproduction
    interpretation, one stored item corresponds to
    one joint 18-RBG 1LDS user-layer decision.

    IMPORTANT:
        The paper's separate statement about
        4 x 18 x 21 = 1512 samples/TTI remains an
        unresolved interpretation issue.
    """

    update_size: int = 128

    def __post_init__(self) -> None:
        if self.update_size <= 0:
            raise ValueError(
                "update_size must be positive."
            )


@dataclass(frozen=True)
class PPOAgentBufferUpdateResult:
    """
    Detached diagnostics from one completed
    on-policy PPO-buffer update.
    """

    #
    # Number of REAL temporally collected
    # transitions.
    #
    num_transitions: int

    #
    # Number of rows actually presented to the PPO
    # optimizer after candidate augmentation.
    #
    num_optimizer_samples: int

    bootstrap_value: torch.Tensor

    mean_td_residual: torch.Tensor

    mean_advantage: torch.Tensor

    mean_target_return: torch.Tensor

    optimizer: PPOUpdateResult

class PPOAgentUpdateCoordinator:
    """
    Own the temporary on-policy PPO agent buffer and
    trigger one update when it reaches M samples.

    Current primary reproduction semantics:

        one sample
            =
        one joint 1LDS user-layer transition.

    This class deliberately does NOT implement:
        - expert demonstrations,
        - JSD updates,
        - UE permutation augmentation,
        - centralized multi-gNB collection,
        - traffic simulation,
        - reward computation.

    Candidate-permutation augmentation is applied
    only AFTER the real temporal rollout has been
    processed by GAE.

    Permuted copies are therefore optimizer samples,
    not fake temporal transitions.

    Those are separate layers of the reproduction.
    """

    def __init__(
        self,
        *,
        config: PPOAgentBufferConfig
        | None = None,
        candidate_augmentation_config: (
            PPOCandidateAugmentationConfig
            | None
        ) = None,
        candidate_augmentation_generator: (
            torch.Generator | None
        ) = None,
    ) -> None:
        if config is None:
            config = (
                PPOAgentBufferConfig()
            )

        self.config = config

        self.candidate_augmentation_config = (
            candidate_augmentation_config
        )

        self.candidate_augmentation_generator = (
            candidate_augmentation_generator
        )

        self._transitions: list[
            PPORolloutTransition
        ] = []


    def __len__(
        self,
    ) -> int:
        return len(
            self._transitions
        )


    @property
    def is_empty(
        self,
    ) -> bool:
        return (
            len(self._transitions)
            == 0
        )


    @property
    def is_full(
        self,
    ) -> bool:
        return (
            len(self._transitions)
            == self.config.update_size
        )


    @property
    def remaining_capacity(
        self,
    ) -> int:
        return (
            self.config.update_size
            - len(self._transitions)
        )


    @staticmethod
    def _validate_temporal_link(
        previous: PPORolloutTransition,
        current: PPORolloutTransition,
    ) -> None:
        """
        Backward-compatible wrapper around the
        shared rollout continuity validator.
        """

        validate_ppo_temporal_link(
            previous,
            current,
        )

    def add_transition(
        self,
        transition: PPORolloutTransition,
    ) -> None:
        """
        Add one finalized on-policy transition.

        The buffer never silently overflows.

        Once full, the caller must perform the PPO
        update before collecting additional
        current-policy samples.
        """

        if self.is_full:
            raise RuntimeError(
                "PPO agent buffer is full. "
                "Perform the on-policy update before "
                "adding more transitions."
            )

        if len(
            self._transitions
        ) > 0:
            self._validate_temporal_link(
                self._transitions[-1],
                transition,
            )

        self._transitions.append(
            transition
        )


    def add_transitions(
        self,
        transitions: list[
            PPORolloutTransition
        ]
        | tuple[
            PPORolloutTransition,
            ...
        ],
    ) -> None:
        """
        Add several temporally ordered transitions
        atomically.

        If they do not fit, nothing is added.
        """

        if len(transitions) == 0:
            return

        if len(transitions) > (
            self.remaining_capacity
        ):
            raise RuntimeError(
                "Transitions would exceed the PPO "
                "agent update size."
            )

        candidate = list(
            transitions
        )

        if len(
            self._transitions
        ) > 0:
            self._validate_temporal_link(
                self._transitions[-1],
                candidate[0],
            )

        for index in range(
            len(candidate) - 1
        ):
            self._validate_temporal_link(
                candidate[index],
                candidate[index + 1],
            )

        self._transitions.extend(
            candidate
        )

    def clear(
        self,
    ) -> None:
        """
        Explicitly discard all stored on-policy
        transitions.

        Normal training clears automatically only
        after a successful PPO update.
        """

        self._transitions = []


    def update_if_ready(
        self,
        *,
        actor: OneLDSPPOActor,
        critic: nn.Module,
        optimizers: PPOOptimizers,
        gae_config: PPOGAEConfig,
        loss_config: PPOLossConfig,
    ) -> (
        PPOAgentBufferUpdateResult
        | None
    ):
        """
        Perform one complete on-policy PPO update
        when the agent buffer contains exactly M
        transitions.

        Sequence:

            finalized transitions
                ->
            stack rollout
                ->
            critic bootstrap of final next_state
                ->
            GAE
                ->
            PPO actor + critic optimization
                ->
            clear agent buffer

        The buffer is cleared ONLY after a successful
        update.
        """

        if not self.is_full:
            return None

        transitions = tuple(
            self._transitions
        )

        prepared = (
            prepare_ppo_trajectory_with_gae(
                transitions=transitions,
                critic=critic,
                config=gae_config,
            )
        )

        #
        # ------------------------------------------------------
        # Candidate permutation augmentation.
        #
        # IMPORTANT:
        #     GAE has already been computed from the
        #     real temporal trajectory.
        #
        #     We now create alternate candidate-order
        #     representations solely for optimization.
        # ------------------------------------------------------
        #
        optimization_data = (
            build_candidate_augmented_ppo_update(
                actor=actor,
                batch=prepared.batch,
                advantage=(
                    prepared
                    .gae
                    .advantage
                ),
                target_return=(
                    prepared
                    .gae
                    .target_return
                ),
                config=(
                    self
                    .candidate_augmentation_config
                ),
                generator=(
                    self
                    .candidate_augmentation_generator
                ),
            )
        )

        update = perform_ppo_update(
            actor=actor,
            critic=critic,
            optimizers=optimizers,
            batch=(
                optimization_data.batch
            ),
            advantage=(
                optimization_data.advantage
            ),
            target_return=(
                optimization_data
                .target_return
            ),
            loss_config=loss_config,
        )
        result = (
            PPOAgentBufferUpdateResult(
                num_transitions=len(
                    transitions
                ),
                num_optimizer_samples=(
                    optimization_data
                    .num_optimizer_samples
                ),
                bootstrap_value=(
                    prepared
                    .bootstrap_value
                    .detach()
                    .clone()
                ),
                mean_td_residual=(
                    prepared
                    .gae
                    .td_residual
                    .mean()
                    .detach()
                    .clone()
                ),
                mean_advantage=(
                    prepared
                    .gae
                    .advantage
                    .mean()
                    .detach()
                    .clone()
                ),
                mean_target_return=(
                    prepared
                    .gae
                    .target_return
                    .mean()
                    .detach()
                    .clone()
                ),
                optimizer=update,
            )
        )

        #
        # IMPORTANT:
        #
        # Clear only AFTER GAE and optimizer.step()
        # have completed successfully.
        #
        self.clear()

        return result
    

