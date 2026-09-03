from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class OneLDSPPOActorConfig:
    """
    Architecture of the v3 1LDS PPO actor.

    PAPER-SPECIFIED training dimensions:

        state size:
            410

        hidden layers:
            2

        hidden units:
            32 each

        activation:
            ReLU

        RBGs:
            18

        actions per RBG:
            11
            =
            10 TDS candidates
            +
            NO ALLOCATION

        total output logits:
            18 * 11 = 198
    """

    state_size: int = 410

    hidden_size: int = 32

    num_rbgs: int = 18

    num_actions_per_rbg: int = 11

    def __post_init__(self) -> None:
        if self.state_size <= 0:
            raise ValueError(
                "state_size must be positive."
            )

        if self.hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if self.num_rbgs <= 0:
            raise ValueError(
                "num_rbgs must be positive."
            )

        if self.num_actions_per_rbg <= 1:
            raise ValueError(
                "num_actions_per_rbg must be "
                "greater than one."
            )

    @property
    def output_size(self) -> int:
        return (
            self.num_rbgs
            * self.num_actions_per_rbg
        )


@dataclass(frozen=True)
class OneLDSPPOActionData:
    """
    Result of one PPO actor decision.

    actions:
        Shape [batch, RBG].

        Integer action for every RBG.

    log_prob_by_rbg:
        Shape [batch, RBG].

        Log probability of each selected RBG action.

    entropy_by_rbg:
        Shape [batch, RBG].

        Categorical entropy for every RBG branch.

    masked_logits:
        Shape [batch, RBG, action].

        Actor logits after invalid actions have
        been masked.
    """

    actions: torch.Tensor

    log_prob_by_rbg: torch.Tensor

    entropy_by_rbg: torch.Tensor

    masked_logits: torch.Tensor


class OneLDSPPOActor(nn.Module):
    """
    v3 1LDS PPO actor.

    Architecture:

        state
          |
          v
        Linear(410, 32)
          |
        ReLU
          |
        Linear(32, 32)
          |
        ReLU
          |
        Linear(32, 198)
          |
        reshape
          |
        [18, 11]
    """

    def __init__(
        self,
        config: OneLDSPPOActorConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.network = nn.Sequential(
            nn.Linear(
                config.state_size,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_size,
                config.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_size,
                config.output_size,
            ),
        )


    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute unmasked actor logits.

        Args:
            state:
                Shape [batch, state_size].

        Returns:
            Shape:

                [batch, RBG, action]
        """

        if state.ndim != 2:
            raise ValueError(
                "state must have shape "
                "[batch, state_size]."
            )

        if state.shape[1] != (
            self.config.state_size
        ):
            raise ValueError(
                "Unexpected state feature size."
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

        flat_logits = self.network(
            state
        )

        return flat_logits.reshape(
            state.shape[0],
            self.config.num_rbgs,
            self.config.num_actions_per_rbg,
        )


    def _validate_action_mask(
        self,
        *,
        action_mask: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> None:
        expected_shape = (
            batch_size,
            self.config.num_rbgs,
            self.config.num_actions_per_rbg,
        )

        if tuple(
            action_mask.shape
        ) != expected_shape:
            raise ValueError(
                "action_mask must have shape "
                "[batch, RBG, action]."
            )

        if action_mask.dtype != torch.bool:
            raise ValueError(
                "action_mask must have dtype bool."
            )

        if action_mask.device != device:
            raise ValueError(
                "action_mask and state must be "
                "on the same device."
            )

        has_valid_action = torch.any(
            action_mask,
            dim=-1,
        )

        if not torch.all(
            has_valid_action
        ):
            raise ValueError(
                "Every RBG must have at least one "
                "valid action."
            )


    def build_distribution(
        self,
        *,
        state: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[
        Categorical,
        torch.Tensor,
    ]:
        """
        Build one masked categorical distribution
        for every RBG.
        """

        logits = self.forward(
            state
        )

        self._validate_action_mask(
            action_mask=action_mask,
            batch_size=state.shape[0],
            device=state.device,
        )

        masked_logits = logits.masked_fill(
            ~action_mask,
            float("-inf"),
        )

        distribution = Categorical(
            logits=masked_logits
        )

        return (
            distribution,
            masked_logits,
        )


    def sample_actions(
        self,
        *,
        state: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> OneLDSPPOActionData:
        """
        Sample one action independently for every RBG.
        """

        (
            distribution,
            masked_logits,
        ) = self.build_distribution(
            state=state,
            action_mask=action_mask,
        )

        actions = distribution.sample()

        log_prob_by_rbg = (
            distribution.log_prob(
                actions
            )
        )

        entropy_by_rbg = (
            distribution.entropy()
        )

        return OneLDSPPOActionData(
            actions=actions,
            log_prob_by_rbg=(
                log_prob_by_rbg
            ),
            entropy_by_rbg=(
                entropy_by_rbg
            ),
            masked_logits=(
                masked_logits
            ),
        )


    def deterministic_actions(
        self,
        *,
        state: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select the highest-logit valid action
        independently for every RBG.

        Returns:
            Shape [batch, RBG].
        """

        (
            _,
            masked_logits,
        ) = self.build_distribution(
            state=state,
            action_mask=action_mask,
        )

        return torch.argmax(
            masked_logits,
            dim=-1,
        )


    def evaluate_actions(
        self,
        *,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Evaluate stored actions under the current policy.

        Returns:
            log_prob_by_rbg:
                [batch, RBG]

            entropy_by_rbg:
                [batch, RBG]
        """

        (
            distribution,
            _,
        ) = self.build_distribution(
            state=state,
            action_mask=action_mask,
        )

        expected_action_shape = (
            state.shape[0],
            self.config.num_rbgs,
        )

        if tuple(
            actions.shape
        ) != expected_action_shape:
            raise ValueError(
                "actions must have shape "
                "[batch, RBG]."
            )

        if torch.is_floating_point(
            actions
        ):
            raise ValueError(
                "actions must use an integer dtype."
            )

        if actions.device != state.device:
            raise ValueError(
                "actions and state must be on "
                "the same device."
            )

        selected_is_valid = torch.gather(
            action_mask,
            dim=2,
            index=actions.unsqueeze(
                -1
            ),
        ).squeeze(
            -1
        )

        if not torch.all(
            selected_is_valid
        ):
            raise ValueError(
                "Stored actions include a masked "
                "action."
            )

        return (
            distribution.log_prob(
                actions
            ),
            distribution.entropy(),
        )

    
def compute_joint_action_log_prob(
    log_prob_by_rbg: torch.Tensor,
) -> torch.Tensor:
    """
    Convert factorized RBG log probabilities to the
    log probability of the complete joint action.

    If:

        pi(a | s)
            =
        product_m pi_m(a_m | s)

    then:

        log pi(a | s)
            =
        sum_m log pi_m(a_m | s)

    Input shape:

        [..., RBG]

    Output shape:

        [...]
    """

    if log_prob_by_rbg.ndim < 1:
        raise ValueError(
            "log_prob_by_rbg must have at least "
            "one dimension."
        )

    if log_prob_by_rbg.shape[-1] < 1:
        raise ValueError(
            "The RBG dimension must be non-empty."
        )

    if not torch.is_floating_point(
        log_prob_by_rbg
    ):
        raise ValueError(
            "log_prob_by_rbg must use a "
            "floating-point dtype."
        )

    if not torch.isfinite(
        log_prob_by_rbg
    ).all():
        raise ValueError(
            "log_prob_by_rbg contains non-finite "
            "values."
        )

    return log_prob_by_rbg.sum(
        dim=-1
    )
