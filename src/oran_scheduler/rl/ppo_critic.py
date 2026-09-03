from dataclasses import dataclass

import torch
from torch import nn


"""
State-value critic for the primary 1LDS PPO
reproduction interpretation.

PAPER-SPECIFIED:
    - PPO uses a state-value function V(s).
    - PPO uses two hidden layers with 32 units
      and ReLU activations.
    - The critic learning rate is 2e-4.

PAPER-INFERRED:
    - We interpret V(s) as one scalar value for the
      complete 1LDS layer state and joint all-RBG
      action process.

The public paper does not explicitly state the final
PPO critic output dimensionality. A branchwise
RBG-value interpretation remains a documented
sensitivity variant.
"""

@dataclass(frozen=True)
class OneLDSPPOCriticConfig:
    """
    Configuration for the 1LDS PPO state-value critic.

    PAPER-SPECIFIED PPO network hyperparameters:
        - 2 hidden layers
        - 32 hidden units per layer
        - ReLU activation

    The 1LDS state has 410 features.

    The critic estimates one scalar state value:

        V_phi(s)

    OPEN-REPRODUCTION DETAIL:
        The paper does not fully specify whether actor
        and critic share any feature layers.

        We use a separate critic MLP.
    """

    state_size: int = 410

    hidden_size: int = 32

    def __post_init__(self) -> None:
        if self.state_size <= 0:
            raise ValueError(
                "state_size must be positive."
            )

        if self.hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

class OneLDSPPOCritic(nn.Module):
    """
    State-value network for 1LDS PPO.

    Architecture:

        [410]
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
        Linear(32, 1)
          |
          v
         V(s)

    The critic is V(s), not Q(s, a).

    Therefore the action is NOT an input.
    """

    def __init__(
        self,
        config: OneLDSPPOCriticConfig,
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
                1,
            ),
        )


    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate V(s).

        Args:
            state:
                Shape [batch, state_size].

        Returns:
            Shape [batch].

            One scalar state-value estimate for
            every state in the batch.
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

        value = self.network(
            state
        )

        return value.squeeze(
            -1
        )



