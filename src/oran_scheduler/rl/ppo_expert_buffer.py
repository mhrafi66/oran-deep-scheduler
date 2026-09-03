from dataclasses import dataclass
from typing import Literal

import torch


PPOExpertBufferReplacementMode = Literal[
    "fifo",
    "reject_when_full",
]


@dataclass(frozen=True)
class PPOExpertBufferConfig:
    """
    Persistent expert-demonstration buffer used by
    the v3 1LDS-PPO expert-guidance path.

    PAPER-SPECIFIED:
        capacity = 4000 state/expert-action pairs.

    PAPER-UNSPECIFIED:
        Exact behavior after the buffer becomes full.

    Therefore replacement_mode is explicit:

        "fifo":
            Replace the oldest demonstrations using
            cyclic FIFO replacement.

        "reject_when_full":
            Keep the original 4000 demonstrations
            and reject later additions.

    sampling_with_replacement is also an explicit
    reproduction choice because Algorithm 1 only
    states that an expert mini-batch is sampled.
    """

    replacement_mode: (
        PPOExpertBufferReplacementMode
    )

    capacity: int = 4000

    sampling_with_replacement: bool = False

    def __post_init__(self) -> None:
        if self.replacement_mode not in (
            "fifo",
            "reject_when_full",
        ):
            raise ValueError(
                "replacement_mode must be "
                "'fifo' or 'reject_when_full'."
            )

        if self.capacity <= 0:
            raise ValueError(
                "capacity must be positive."
            )


@dataclass(frozen=True)
class PPOExpertDemonstration:
    """
    One expert state/action pair.

    state:
        Shape [state_size].

        For paper-shaped 1LDS PPO:
            [410]

    expert_actions:
        Shape [RBG].

        For paper-shaped 1LDS PPO:
            [18]

        One expert action index for every RBG.

    action_mask:
        Shape [RBG, action].

        For paper-shaped 1LDS PPO:
            [18, 11]

    PAPER-SPECIFIED BUFFER CONTENT:
        (state, expert_action)

    REPRODUCTION SUPPORT METADATA:
        action_mask

    We retain the mask because the valid action set
    changes with the current allocation state and
    will be required when evaluating the actor
    distribution for expert-guidance training.
    """

    state: torch.Tensor

    expert_actions: torch.Tensor

    action_mask: torch.Tensor


@dataclass(frozen=True)
class PPOExpertBatch:
    """
    Mini-batch sampled from the expert buffer.

    states:
        [batch, state_size]

    expert_actions:
        [batch, RBG]

    action_masks:
        [batch, RBG, action]
    """

    states: torch.Tensor

    expert_actions: torch.Tensor

    action_masks: torch.Tensor


class PPOExpertDemonstrationBuffer:
    """
    Persistent expert replay buffer for 1LDS-PPO.

    Unlike the temporary on-policy PPO agent buffer,
    this buffer persists across PPO updates.

    It stores only expert supervision data:

        state
        expert action
        action mask

    It does NOT store:
        reward
        next state
        advantage
        return
        termination

    because those quantities are irrelevant to the
    expert imitation/guidance update.
    """

    def __init__(
        self,
        config: PPOExpertBufferConfig,
    ) -> None:
        self.config = config

        self._storage: list[
            PPOExpertDemonstration
        ] = []

        self._next_replacement_index = 0

        self._state_size: int | None = None

        self._num_rbgs: int | None = None

        self._num_actions_per_rbg: (
            int | None
        ) = None

        self._device: (
            torch.device | None
        ) = None

        self._num_added_total = 0


    def __len__(
        self,
    ) -> int:
        return len(
            self._storage
        )


    @property
    def is_full(
        self,
    ) -> bool:
        return (
            len(self._storage)
            >= self.config.capacity
        )


    @property
    def num_added_total(
        self,
    ) -> int:
        """
        Number of accepted demonstrations over the
        lifetime of the buffer.

        Under FIFO replacement this may exceed the
        current buffer length.
        """

        return self._num_added_total


    def _validate_demonstration(
        self,
        *,
        state: torch.Tensor,
        expert_actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        if state.ndim != 1:
            raise ValueError(
                "state must have shape "
                "[state_size]."
            )

        if not torch.is_floating_point(
            state
        ):
            raise ValueError(
                "state must use a floating-point "
                "dtype."
            )

        if not torch.isfinite(
            state
        ).all():
            raise ValueError(
                "state contains non-finite values."
            )

        if expert_actions.ndim != 1:
            raise ValueError(
                "expert_actions must have shape "
                "[RBG]."
            )

        if (
            torch.is_floating_point(
                expert_actions
            )
            or expert_actions.dtype
            == torch.bool
        ):
            raise ValueError(
                "expert_actions must use an "
                "integer dtype."
            )

        if action_mask.ndim != 2:
            raise ValueError(
                "action_mask must have shape "
                "[RBG, action]."
            )

        if action_mask.dtype != torch.bool:
            raise ValueError(
                "action_mask must use torch.bool."
            )

        num_rbgs = int(
            action_mask.shape[0]
        )

        num_actions = int(
            action_mask.shape[1]
        )

        if tuple(
            expert_actions.shape
        ) != (
            num_rbgs,
        ):
            raise ValueError(
                "expert_actions and action_mask "
                "have inconsistent RBG dimensions."
            )

        if num_actions <= 1:
            raise ValueError(
                "At least two actions per RBG are "
                "required."
            )

        device = state.device

        if (
            expert_actions.device != device
            or action_mask.device != device
        ):
            raise ValueError(
                "Expert demonstration tensors must "
                "be on the same device."
            )

        has_legal_action = torch.any(
            action_mask,
            dim=1,
        )

        if not torch.all(
            has_legal_action
        ):
            raise ValueError(
                "Every RBG must have at least one "
                "legal action."
            )

        if torch.any(
            expert_actions < 0
        ):
            raise ValueError(
                "Expert action cannot be negative."
            )

        if torch.any(
            expert_actions >= num_actions
        ):
            raise ValueError(
                "Expert action exceeds the action "
                "space."
            )

        selected_is_legal = (
            action_mask
            .gather(
                dim=1,
                index=(
                    expert_actions
                    .unsqueeze(
                        1
                    )
                ),
            )
            .squeeze(
                1
            )
        )

        if not torch.all(
            selected_is_legal
        ):
            raise ValueError(
                "Expert action selects an action "
                "that is masked as invalid."
            )


    def _validate_buffer_layout(
        self,
        *,
        state: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> None:
        state_size = int(
            state.shape[0]
        )

        num_rbgs = int(
            action_mask.shape[0]
        )

        num_actions = int(
            action_mask.shape[1]
        )

        if self._state_size is None:
            self._state_size = state_size
            self._num_rbgs = num_rbgs
            self._num_actions_per_rbg = (
                num_actions
            )
            self._device = state.device

            return

        if state_size != self._state_size:
            raise ValueError(
                "Expert state size changed within "
                "one demonstration buffer."
            )

        if num_rbgs != self._num_rbgs:
            raise ValueError(
                "Number of RBGs changed within one "
                "demonstration buffer."
            )

        if (
            num_actions
            != self._num_actions_per_rbg
        ):
            raise ValueError(
                "Expert action-space size changed "
                "within one demonstration buffer."
            )

        if state.device != self._device:
            raise ValueError(
                "Expert demonstration device "
                "changed within one buffer."
            )


    def add(
        self,
        *,
        state: torch.Tensor,
        expert_actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> bool:
        """
        Add one expert demonstration.

        Returns:
            True:
                demonstration was accepted.

            False:
                buffer was full and
                replacement_mode="reject_when_full".

        All stored tensors are detached clones.

        Expert guidance must never retain an
        autograd graph from environment interaction.
        """

        self._validate_demonstration(
            state=state,
            expert_actions=expert_actions,
            action_mask=action_mask,
        )

        self._validate_buffer_layout(
            state=state,
            action_mask=action_mask,
        )

        demonstration = (
            PPOExpertDemonstration(
                state=(
                    state
                    .detach()
                    .clone()
                ),
                expert_actions=(
                    expert_actions
                    .detach()
                    .clone()
                ),
                action_mask=(
                    action_mask
                    .detach()
                    .clone()
                ),
            )
        )

        if not self.is_full:
            self._storage.append(
                demonstration
            )

            self._num_added_total += 1

            return True

        if (
            self.config.replacement_mode
            == "reject_when_full"
        ):
            return False

        if (
            self.config.replacement_mode
            == "fifo"
        ):
            self._storage[
                self._next_replacement_index
            ] = demonstration

            self._next_replacement_index = (
                (
                    self
                    ._next_replacement_index
                    + 1
                )
                % self.config.capacity
            )

            self._num_added_total += 1

            return True

        raise RuntimeError(
            "Unexpected expert-buffer replacement "
            "mode."
        )


    def sample(
        self,
        *,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> PPOExpertBatch:
        """
        Sample an expert mini-batch.

        PAPER-SPECIFIED:
            Algorithm 1 samples an expert mini-batch
            after the PPO update.

        PAPER-UNSPECIFIED:
            Exact batch size and whether sampling is
            with replacement.

        Both are therefore controlled externally.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        num_available = len(
            self._storage
        )

        if num_available == 0:
            raise RuntimeError(
                "Cannot sample from an empty expert "
                "buffer."
            )

        if (
            not self.config.sampling_with_replacement
            and batch_size > num_available
        ):
            raise ValueError(
                "batch_size exceeds the number of "
                "expert demonstrations available."
            )

        if self.config.sampling_with_replacement:
            indices = torch.randint(
                low=0,
                high=num_available,
                size=(
                    batch_size,
                ),
                generator=generator,
            )

        else:
            indices = torch.randperm(
                num_available,
                generator=generator,
            )[
                :batch_size
            ]

        selected = [
            self._storage[index]
            for index in indices.tolist()
        ]

        states = torch.stack(
            [
                item.state
                for item in selected
            ],
            dim=0,
        )

        expert_actions = torch.stack(
            [
                item.expert_actions
                for item in selected
            ],
            dim=0,
        )

        action_masks = torch.stack(
            [
                item.action_mask
                for item in selected
            ],
            dim=0,
        )

        return PPOExpertBatch(
            states=states,
            expert_actions=expert_actions,
            action_masks=action_masks,
        )


