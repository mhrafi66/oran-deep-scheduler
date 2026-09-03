from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    compute_joint_action_log_prob,
)


EntropyReduction = Literal[
    "sum",
    "mean",
]


@dataclass(frozen=True)
class PPOLossConfig:
    """
    PPO loss configuration for the primary 1LDS
    scalar/joint reproduction interpretation.

    PAPER-SPECIFIED:
        clip_epsilon = 0.2

    PAPER-UNSPECIFIED:
        entropy_coefficient

    PAPER-INFERRED:
        Because the complete 1LDS policy is treated
        as the product of the per-RBG categorical
        policies, its joint entropy is the sum of
        the per-RBG entropies.

    OPEN-REPRODUCTION SENSITIVITY:
        entropy_reduction="mean" is retained as an
        alternative scaling convention.
    """

    entropy_coefficient: float

    clip_epsilon: float = 0.2

    entropy_reduction: EntropyReduction = "sum"

    def __post_init__(self) -> None:
        if not math.isfinite(
            self.entropy_coefficient
        ):
            raise ValueError(
                "entropy_coefficient must be finite."
            )

        if self.entropy_coefficient < 0.0:
            raise ValueError(
                "entropy_coefficient must be "
                "non-negative."
            )

        if not math.isfinite(
            self.clip_epsilon
        ):
            raise ValueError(
                "clip_epsilon must be finite."
            )

        if not (
            0.0
            < self.clip_epsilon
            < 1.0
        ):
            raise ValueError(
                "clip_epsilon must lie in (0, 1)."
            )

        if self.entropy_reduction not in (
            "sum",
            "mean",
        ):
            raise ValueError(
                "entropy_reduction must be "
                "'sum' or 'mean'."
            )


@dataclass(frozen=True)
class PPOClippedPolicyResult:
    """
    Intermediate PPO clipped-policy quantities.

    All vector-valued tensors have shape:

        [transition]
    """

    log_ratio: torch.Tensor

    ratio: torch.Tensor

    clipped_ratio: torch.Tensor

    unclipped_surrogate: torch.Tensor

    clipped_surrogate: torch.Tensor

    surrogate: torch.Tensor

    objective: torch.Tensor


@dataclass(frozen=True)
class PPOActorLossResult:
    """
    Actor-side PPO loss and diagnostics.

    current_log_prob_by_rbg:
        [transition, RBG]

    current_joint_log_prob:
        [transition]

    entropy_by_rbg:
        [transition, RBG]

    entropy_per_transition:
        [transition]

    policy.objective:
        scalar

    entropy_objective:
        scalar

    loss:
        scalar
    """

    current_log_prob_by_rbg: torch.Tensor

    current_joint_log_prob: torch.Tensor

    entropy_by_rbg: torch.Tensor

    entropy_per_transition: torch.Tensor

    policy: PPOClippedPolicyResult

    entropy_objective: torch.Tensor

    loss: torch.Tensor


@dataclass(frozen=True)
class PPOCriticLossResult:
    """
    Critic-side PPO regression quantities.

    predicted_value:
        [transition]

    squared_error:
        [transition]

    loss:
        scalar
    """

    predicted_value: torch.Tensor

    squared_error: torch.Tensor

    loss: torch.Tensor


def _validate_float_vector(
    *,
    name: str,
    value: torch.Tensor,
) -> None:
    if value.ndim != 1:
        raise ValueError(
            f"{name} must have shape "
            "[transition]."
        )

    if value.shape[0] < 1:
        raise ValueError(
            f"{name} must be non-empty."
        )

    if not torch.is_floating_point(
        value
    ):
        raise ValueError(
            f"{name} must use a floating-point "
            "dtype."
        )

    if not torch.isfinite(
        value
    ).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )


def compute_ppo_clipped_policy_objective(
    *,
    current_joint_log_prob: torch.Tensor,
    old_joint_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
) -> PPOClippedPolicyResult:
    """
    Compute the PPO clipped policy objective.

    rho_t =
        exp(
            log pi_new(a_t | s_t)
            -
            log pi_old(a_t | s_t)
        )

    objective_t =
        min(
            rho_t * A_t,
            clip(rho_t) * A_t,
        )

    The signed advantage is intentionally preserved.

    No advantage normalization is applied because
    the v3 paper does not publicly specify such a
    step.
    """

    _validate_float_vector(
        name="current_joint_log_prob",
        value=current_joint_log_prob,
    )

    _validate_float_vector(
        name="old_joint_log_prob",
        value=old_joint_log_prob,
    )

    _validate_float_vector(
        name="advantage",
        value=advantage,
    )

    if (
        current_joint_log_prob.shape
        != old_joint_log_prob.shape
    ):
        raise ValueError(
            "Current and old joint log "
            "probabilities must have the same shape."
        )

    if (
        current_joint_log_prob.shape
        != advantage.shape
    ):
        raise ValueError(
            "Joint log probability and advantage "
            "must have the same shape."
        )

    if (
        current_joint_log_prob.device
        != old_joint_log_prob.device
        or current_joint_log_prob.device
        != advantage.device
    ):
        raise ValueError(
            "PPO policy tensors must be on the "
            "same device."
        )

    if (
        current_joint_log_prob.dtype
        != old_joint_log_prob.dtype
        or current_joint_log_prob.dtype
        != advantage.dtype
    ):
        raise ValueError(
            "PPO policy tensors must use the "
            "same dtype."
        )

    if not (
        0.0
        < clip_epsilon
        < 1.0
    ):
        raise ValueError(
            "clip_epsilon must lie in (0, 1)."
        )

    old_reference = (
        old_joint_log_prob
        .detach()
    )

    advantage_reference = (
        advantage
        .detach()
    )

    log_ratio = (
        current_joint_log_prob
        - old_reference
    )

    ratio = torch.exp(
        log_ratio
    )

    if not torch.isfinite(
        ratio
    ).all():
        raise ValueError(
            "PPO probability ratio became "
            "non-finite."
        )

    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - clip_epsilon,
        max=1.0 + clip_epsilon,
    )

    unclipped_surrogate = (
        ratio
        * advantage_reference
    )

    clipped_surrogate = (
        clipped_ratio
        * advantage_reference
    )

    surrogate = torch.minimum(
        unclipped_surrogate,
        clipped_surrogate,
    )

    objective = surrogate.mean()

    return PPOClippedPolicyResult(
        log_ratio=log_ratio,
        ratio=ratio,
        clipped_ratio=clipped_ratio,
        unclipped_surrogate=(
            unclipped_surrogate
        ),
        clipped_surrogate=(
            clipped_surrogate
        ),
        surrogate=surrogate,
        objective=objective,
    )


def reduce_ppo_entropy_by_rbg(
    entropy_by_rbg: torch.Tensor,
    *,
    reduction: EntropyReduction,
) -> torch.Tensor:
    """
    Reduce per-RBG categorical entropies to one
    entropy value per joint 1LDS transition.

    Input:
        [transition, RBG]

    Output:
        [transition]
    """

    if entropy_by_rbg.ndim != 2:
        raise ValueError(
            "entropy_by_rbg must have shape "
            "[transition, RBG]."
        )

    if entropy_by_rbg.shape[0] < 1:
        raise ValueError(
            "The transition dimension must be "
            "non-empty."
        )

    if entropy_by_rbg.shape[1] < 1:
        raise ValueError(
            "The RBG dimension must be non-empty."
        )

    if not torch.is_floating_point(
        entropy_by_rbg
    ):
        raise ValueError(
            "entropy_by_rbg must use a "
            "floating-point dtype."
        )

    if not torch.isfinite(
        entropy_by_rbg
    ).all():
        raise ValueError(
            "entropy_by_rbg contains non-finite "
            "values."
        )

    if reduction == "sum":
        return entropy_by_rbg.sum(
            dim=-1
        )

    if reduction == "mean":
        return entropy_by_rbg.mean(
            dim=-1
        )

    raise ValueError(
        "reduction must be 'sum' or 'mean'."
    )

def compute_ppo_actor_loss(
    *,
    actor: OneLDSPPOActor,
    state: torch.Tensor,
    action_mask: torch.Tensor,
    actions: torch.Tensor,
    old_log_prob_by_rbg: torch.Tensor,
    old_joint_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    config: PPOLossConfig,
) -> PPOActorLossResult:
    """
    Re-evaluate stored rollout actions under the
    current actor and compute the PPO actor loss.

    IMPORTANT:

    old log probabilities and advantages are treated
    as fixed rollout targets.

    Gradients flow only through the current policy.
    """

    (
        current_log_prob_by_rbg,
        entropy_by_rbg,
    ) = actor.evaluate_actions(
        state=state,
        action_mask=action_mask,
        actions=actions,
    )

    if (
        old_log_prob_by_rbg.shape
        != current_log_prob_by_rbg.shape
    ):
        raise ValueError(
            "old_log_prob_by_rbg has an "
            "unexpected shape."
        )

    if not torch.is_floating_point(
        old_log_prob_by_rbg
    ):
        raise ValueError(
            "old_log_prob_by_rbg must use a "
            "floating-point dtype."
        )

    if not torch.isfinite(
        old_log_prob_by_rbg
    ).all():
        raise ValueError(
            "old_log_prob_by_rbg contains "
            "non-finite values."
        )

    if (
        old_log_prob_by_rbg.device
        != current_log_prob_by_rbg.device
    ):
        raise ValueError(
            "Old and current RBG log probabilities "
            "must be on the same device."
        )

    if (
        old_log_prob_by_rbg.dtype
        != current_log_prob_by_rbg.dtype
    ):
        raise ValueError(
            "Old and current RBG log probabilities "
            "must use the same dtype."
        )

    current_joint_log_prob = (
        compute_joint_action_log_prob(
            current_log_prob_by_rbg
        )
    )

    stored_joint_from_branches = (
        compute_joint_action_log_prob(
            old_log_prob_by_rbg.detach()
        )
    )

    _validate_float_vector(
        name="old_joint_log_prob",
        value=old_joint_log_prob,
    )

    if not torch.allclose(
        stored_joint_from_branches,
        old_joint_log_prob.detach(),
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError(
            "Stored old_joint_log_prob does not "
            "match the sum of "
            "old_log_prob_by_rbg."
        )

    policy = (
        compute_ppo_clipped_policy_objective(
            current_joint_log_prob=(
                current_joint_log_prob
            ),
            old_joint_log_prob=(
                old_joint_log_prob
            ),
            advantage=advantage,
            clip_epsilon=(
                config.clip_epsilon
            ),
        )
    )

    entropy_per_transition = (
        reduce_ppo_entropy_by_rbg(
            entropy_by_rbg,
            reduction=(
                config.entropy_reduction
            ),
        )
    )

    entropy_objective = (
        entropy_per_transition.mean()
    )

    loss = (
        -policy.objective
        - (
            config.entropy_coefficient
            * entropy_objective
        )
    )

    if not torch.isfinite(
        loss
    ):
        raise ValueError(
            "PPO actor loss became non-finite."
        )

    return PPOActorLossResult(
        current_log_prob_by_rbg=(
            current_log_prob_by_rbg
        ),
        current_joint_log_prob=(
            current_joint_log_prob
        ),
        entropy_by_rbg=(
            entropy_by_rbg
        ),
        entropy_per_transition=(
            entropy_per_transition
        ),
        policy=policy,
        entropy_objective=(
            entropy_objective
        ),
        loss=loss,
    )

def compute_ppo_critic_loss(
    *,
    critic: nn.Module,
    state: torch.Tensor,
    target_return: torch.Tensor,
) -> PPOCriticLossResult:
    """
    Compute the PPO critic regression loss:

        1/2 * mean(
            (V(s) - J_hat)^2
        )

    target_return is detached so the critic update
    cannot backpropagate through GAE or rollout
    collection.
    """

    _validate_float_vector(
        name="target_return",
        value=target_return,
    )

    predicted_value = critic(
        state
    )

    _validate_float_vector(
        name="predicted_value",
        value=predicted_value,
    )

    if (
        predicted_value.shape
        != target_return.shape
    ):
        raise ValueError(
            "predicted_value and target_return "
            "must have the same shape."
        )

    if (
        predicted_value.device
        != target_return.device
    ):
        raise ValueError(
            "Critic prediction and target must be "
            "on the same device."
        )

    if (
        predicted_value.dtype
        != target_return.dtype
    ):
        raise ValueError(
            "Critic prediction and target must use "
            "the same dtype."
        )

    target_reference = (
        target_return
        .detach()
    )

    squared_error = (
        predicted_value
        - target_reference
    ) ** 2

    loss = (
        0.5
        * squared_error.mean()
    )

    if not torch.isfinite(
        loss
    ):
        raise ValueError(
            "PPO critic loss became non-finite."
        )

    return PPOCriticLossResult(
        predicted_value=(
            predicted_value
        ),
        squared_error=(
            squared_error
        ),
        loss=loss,
    )

