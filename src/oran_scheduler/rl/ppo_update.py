from dataclasses import dataclass
import math

import torch
from torch import nn

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
)
from oran_scheduler.rl.ppo_loss import (
    PPOActorLossResult,
    PPOCriticLossResult,
    PPOLossConfig,
    compute_ppo_actor_loss,
    compute_ppo_critic_loss,
)
from oran_scheduler.rl.ppo_rollout import (
    PPORolloutBatch,
)

@dataclass(frozen=True)
class PPOOptimizerConfig:
    """
    Optimizer configuration for 1LDS PPO.

    PAPER-SPECIFIED:
        actor_learning_rate = 1e-4
        critic_learning_rate = 2e-4
        optimizer = Adam

    OPEN-REPRODUCTION ASSUMPTION:
        The public paper does not specify Adam's
        beta values, epsilon, weight decay, or
        AMSGrad setting.

        The defaults below match standard PyTorch
        Adam defaults and remain explicitly
        configurable for sensitivity experiments.
    """

    actor_learning_rate: float = 1.0e-4

    critic_learning_rate: float = 2.0e-4

    adam_beta1: float = 0.9

    adam_beta2: float = 0.999

    adam_epsilon: float = 1.0e-8

    weight_decay: float = 0.0

    amsgrad: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(
                self.actor_learning_rate
            )
            or self.actor_learning_rate <= 0.0
        ):
            raise ValueError(
                "actor_learning_rate must be "
                "positive and finite."
            )

        if (
            not math.isfinite(
                self.critic_learning_rate
            )
            or self.critic_learning_rate <= 0.0
        ):
            raise ValueError(
                "critic_learning_rate must be "
                "positive and finite."
            )

        if not (
            0.0
            <= self.adam_beta1
            < 1.0
        ):
            raise ValueError(
                "adam_beta1 must lie in [0, 1)."
            )

        if not (
            0.0
            <= self.adam_beta2
            < 1.0
        ):
            raise ValueError(
                "adam_beta2 must lie in [0, 1)."
            )

        if (
            not math.isfinite(
                self.adam_epsilon
            )
            or self.adam_epsilon <= 0.0
        ):
            raise ValueError(
                "adam_epsilon must be positive "
                "and finite."
            )

        if (
            not math.isfinite(
                self.weight_decay
            )
            or self.weight_decay < 0.0
        ):
            raise ValueError(
                "weight_decay must be "
                "non-negative and finite."
            )


@dataclass
class PPOOptimizers:
    """
    Separate actor and critic Adam optimizers.

    The current reproduction uses separate actor
    and critic MLPs.
    """

    actor: torch.optim.Adam

    critic: torch.optim.Adam


def create_ppo_optimizers(
    *,
    actor: OneLDSPPOActor,
    critic: nn.Module,
    config: PPOOptimizerConfig,
) -> PPOOptimizers:
    """
    Create the paper-shaped PPO optimizers.

    PAPER-SPECIFIED:
        Adam
        actor LR  = 1e-4
        critic LR = 2e-4

    Adam internal hyperparameters that are absent
    from the public paper come from config and are
    therefore explicit reproduction choices.
    """

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=config.actor_learning_rate,
        betas=(
            config.adam_beta1,
            config.adam_beta2,
        ),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=config.critic_learning_rate,
        betas=(
            config.adam_beta1,
            config.adam_beta2,
        ),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    )

    return PPOOptimizers(
        actor=actor_optimizer,
        critic=critic_optimizer,
    )


@dataclass(frozen=True)
class PPOUpdateResult:
    """
    Detached diagnostics from one PPO optimization
    step.

    These values describe the losses that generated
    the parameter update.

    They deliberately do not retain an autograd
    graph.
    """

    actor_loss: torch.Tensor

    critic_loss: torch.Tensor

    policy_objective: torch.Tensor

    entropy_objective: torch.Tensor

    mean_probability_ratio: torch.Tensor

    mean_advantage: torch.Tensor

    mean_target_return: torch.Tensor


def _module_device(
    module: nn.Module,
) -> torch.device:
    parameters = list(
        module.parameters()
    )

    if len(parameters) == 0:
        raise ValueError(
            "Module must contain trainable "
            "parameters."
        )

    device = parameters[0].device

    for parameter in parameters[1:]:
        if parameter.device != device:
            raise ValueError(
                "All module parameters must be on "
                "one device."
            )

    return device


def _validate_ppo_update_inputs(
    *,
    actor: OneLDSPPOActor,
    critic: nn.Module,
    batch: PPORolloutBatch,
    advantage: torch.Tensor,
    target_return: torch.Tensor,
) -> None:
    actor_device = _module_device(
        actor
    )

    critic_device = _module_device(
        critic
    )

    if actor_device != critic_device:
        raise ValueError(
            "Actor and critic must be on the same "
            "device."
        )

    num_transitions = int(
        batch.states.shape[0]
    )

    if num_transitions < 1:
        raise ValueError(
            "PPO update batch must be non-empty."
        )

    if tuple(
        advantage.shape
    ) != (
        num_transitions,
    ):
        raise ValueError(
            "advantage must have shape "
            "[transition]."
        )

    if tuple(
        target_return.shape
    ) != (
        num_transitions,
    ):
        raise ValueError(
            "target_return must have shape "
            "[transition]."
        )

    tensors = {
        "states": batch.states,
        "actions": batch.actions,
        "action_masks": batch.action_masks,
        "old_log_prob_by_rbg": (
            batch.old_log_prob_by_rbg
        ),
        "old_joint_log_prob": (
            batch.old_joint_log_prob
        ),
        "advantage": advantage,
        "target_return": target_return,
    }

    for name, tensor in tensors.items():
        if tensor.device != actor_device:
            raise ValueError(
                f"{name} is on {tensor.device}, "
                f"but PPO models are on "
                f"{actor_device}."
            )

    if not torch.isfinite(
        advantage
    ).all():
        raise ValueError(
            "advantage contains non-finite values."
        )

    if not torch.isfinite(
        target_return
    ).all():
        raise ValueError(
            "target_return contains non-finite "
            "values."
        )


def _validate_finite_gradients(
    *,
    module: nn.Module,
    name: str,
) -> None:
    saw_gradient = False

    for parameter in module.parameters():
        if parameter.grad is None:
            continue

        saw_gradient = True

        if not torch.isfinite(
            parameter.grad
        ).all():
            raise FloatingPointError(
                f"{name} produced a non-finite "
                "gradient."
            )

    if not saw_gradient:
        raise RuntimeError(
            f"{name} produced no gradients."
        )


def perform_ppo_update(
    *,
    actor: OneLDSPPOActor,
    critic: nn.Module,
    optimizers: PPOOptimizers,
    batch: PPORolloutBatch,
    advantage: torch.Tensor,
    target_return: torch.Tensor,
    loss_config: PPOLossConfig,
) -> PPOUpdateResult:
    """
    Perform one PPO actor update and one critic
    update.

    This is the first layer of the implementation
    that actually changes neural-network parameters.

    The function intentionally does NOT:
        - normalize advantages,
        - shuffle transitions,
        - split mini-batches,
        - repeat epochs,
        - clip gradients,
        - generate GAE,
        - collect rollout data,
        - perform expert/JSD optimization.

    Those responsibilities remain outside this
    primitive.
    """

    _validate_ppo_update_inputs(
        actor=actor,
        critic=critic,
        batch=batch,
        advantage=advantage,
        target_return=target_return,
    )

    actor.train()

    critic.train()

    # --------------------------------------------------------------
    # Actor update
    # --------------------------------------------------------------

    optimizers.actor.zero_grad(
        set_to_none=True
    )

    actor_result: PPOActorLossResult = (
        compute_ppo_actor_loss(
            actor=actor,
            state=batch.states,
            action_mask=batch.action_masks,
            actions=batch.actions,
            old_log_prob_by_rbg=(
                batch.old_log_prob_by_rbg
            ),
            old_joint_log_prob=(
                batch.old_joint_log_prob
            ),
            advantage=advantage,
            config=loss_config,
        )
    )

    actor_result.loss.backward()

    _validate_finite_gradients(
        module=actor,
        name="PPO actor",
    )

    optimizers.actor.step()

    # --------------------------------------------------------------
    # Critic update
    # --------------------------------------------------------------

    optimizers.critic.zero_grad(
        set_to_none=True
    )

    critic_result: PPOCriticLossResult = (
        compute_ppo_critic_loss(
            critic=critic,
            state=batch.states,
            target_return=target_return,
        )
    )

    critic_result.loss.backward()

    _validate_finite_gradients(
        module=critic,
        name="PPO critic",
    )

    optimizers.critic.step()

    return PPOUpdateResult(
        actor_loss=(
            actor_result
            .loss
            .detach()
            .clone()
        ),
        critic_loss=(
            critic_result
            .loss
            .detach()
            .clone()
        ),
        policy_objective=(
            actor_result
            .policy
            .objective
            .detach()
            .clone()
        ),
        entropy_objective=(
            actor_result
            .entropy_objective
            .detach()
            .clone()
        ),
        mean_probability_ratio=(
            actor_result
            .policy
            .ratio
            .mean()
            .detach()
            .clone()
        ),
        mean_advantage=(
            advantage
            .mean()
            .detach()
            .clone()
        ),
        mean_target_return=(
            target_return
            .mean()
            .detach()
            .clone()
        ),
    )

