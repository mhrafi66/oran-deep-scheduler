from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PPOGAEConfig:
    """
    Generalized Advantage Estimation configuration.

    PAPER-SPECIFIED:
        gamma = 0.95 for PPO.

    PAPER-SPECIFIED:
        GAE is used to compute PPO advantages.

    PAPER-UNSPECIFIED:
        The exact GAE lambda is not reported.

    Therefore gae_lambda must be supplied explicitly
    by the reproduction rather than silently defaulted.
    """

    gae_lambda: float

    gamma: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(
                "gamma must lie in [0, 1]."
            )

        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(
                "gae_lambda must lie in [0, 1]."
            )


@dataclass(frozen=True)
class PPOGAEConfig:
    """
    Generalized Advantage Estimation configuration.

    PAPER-SPECIFIED:
        gamma = 0.95 for PPO.

    PAPER-SPECIFIED:
        GAE is used to compute PPO advantages.

    PAPER-UNSPECIFIED:
        The exact GAE lambda is not reported.

    Therefore gae_lambda must be supplied explicitly
    by the reproduction rather than silently defaulted.
    """

    gae_lambda: float

    gamma: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(
                "gamma must lie in [0, 1]."
            )

        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(
                "gae_lambda must lie in [0, 1]."
            )

@dataclass(frozen=True)
class PPOGAEResult:
    """
    Outputs of Generalized Advantage Estimation.

    All tensors have shape:

        [transition]

    td_residual:
        One-step TD errors delta_t.

    advantage:
        GAE advantages A_t.

    target_return:
        Critic regression targets:

            J_hat_t = A_t + V(s_t)
    """

    td_residual: torch.Tensor

    advantage: torch.Tensor

    target_return: torch.Tensor

def _validate_transition_tensor(
    *,
    name: str,
    tensor: torch.Tensor,
) -> None:
    if tensor.ndim != 1:
        raise ValueError(
            f"{name} must have shape [transition]."
        )

    if tensor.shape[0] < 1:
        raise ValueError(
            f"{name} must be non-empty."
        )

    if not torch.is_floating_point(
        tensor
    ):
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

def compute_ppo_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    terminated: torch.Tensor,
    config: PPOGAEConfig,
) -> PPOGAEResult:
    """
    Compute scalar/joint PPO GAE.

    Args:
        rewards:
            Shape [transition].

            Scalar reward associated with each
            stored joint PPO transition.

        values:
            Shape [transition].

            V(s_t) for every stored state.

        next_value:
            Scalar tensor.

            V(s_next) for the state immediately
            following the final stored transition.

        terminated:
            Shape [transition], dtype bool.

            True when that transition genuinely ends
            the environment episode.

            A rollout-buffer boundary or TTI boundary
            is NOT automatically a termination.

    Returns:
        TD residuals, GAE advantages, and target
        returns, all with shape [transition].
    """

    _validate_transition_tensor(
        name="rewards",
        tensor=rewards,
    )

    _validate_transition_tensor(
        name="values",
        tensor=values,
    )

    if rewards.shape != values.shape:
        raise ValueError(
            "rewards and values must have "
            "the same shape."
        )

    if rewards.device != values.device:
        raise ValueError(
            "rewards and values must be on "
            "the same device."
        )

    if rewards.dtype != values.dtype:
        raise ValueError(
            "rewards and values must use "
            "the same dtype."
        )

    if next_value.ndim != 0:
        raise ValueError(
            "next_value must be a scalar tensor."
        )

    if not torch.is_floating_point(
        next_value
    ):
        raise ValueError(
            "next_value must use a floating-point "
            "dtype."
        )

    if next_value.device != rewards.device:
        raise ValueError(
            "next_value must be on the same device "
            "as rewards."
        )

    if next_value.dtype != rewards.dtype:
        raise ValueError(
            "next_value must use the same dtype "
            "as rewards."
        )

    if not torch.isfinite(
        next_value
    ):
        raise ValueError(
            "next_value must be finite."
        )

    if terminated.ndim != 1:
        raise ValueError(
            "terminated must have shape "
            "[transition]."
        )

    if terminated.shape != rewards.shape:
        raise ValueError(
            "terminated must have the same shape "
            "as rewards."
        )

    if terminated.dtype != torch.bool:
        raise ValueError(
            "terminated must have dtype bool."
        )

    if terminated.device != rewards.device:
        raise ValueError(
            "terminated must be on the same device "
            "as rewards."
        )

    td_residual = torch.zeros_like(
        rewards
    )

    advantage = torch.zeros_like(
        rewards
    )

    running_advantage = torch.zeros(
        (),
        dtype=rewards.dtype,
        device=rewards.device,
    )

    num_transitions = rewards.shape[0]

    for transition_index in reversed(
        range(
            num_transitions
        )
    ):
        if transition_index == (
            num_transitions - 1
        ):
            following_value = next_value

        else:
            following_value = values[
                transition_index + 1
            ]

        nonterminal = (
            ~terminated[
                transition_index
            ]
        ).to(
            dtype=rewards.dtype
        )

        delta = (
            rewards[
                transition_index
            ]
            + (
                config.gamma
                * nonterminal
                * following_value
            )
            - values[
                transition_index
            ]
        )

        running_advantage = (
            delta
            + (
                config.gamma
                * config.gae_lambda
                * nonterminal
                * running_advantage
            )
        )

        td_residual[
            transition_index
        ] = delta

        advantage[
            transition_index
        ] = running_advantage

    target_return = (
        advantage
        + values
    )

    return PPOGAEResult(
        td_residual=td_residual,
        advantage=advantage,
        target_return=target_return,
    )


