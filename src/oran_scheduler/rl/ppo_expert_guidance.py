from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch.optim import Optimizer

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
)
from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertBatch,
    PPOExpertDemonstrationBuffer,
)


PPOExpertDivergenceMode = Literal[
    "true_jsd",
    "printed_symmetric_kl_smoothed",
]


PPOExpertRBGReduction = Literal[
    "mean",
    "sum",
]


@dataclass(frozen=True)
class PPOExpertGuidanceConfig:
    """
    Expert-guidance update configuration.

    PAPER-SPECIFIED:
        - expert guidance follows a PPO update
        - a mini-batch is sampled from D_expert
        - the actor is updated using a JSD-style loss

    PAPER-UNSPECIFIED / AMBIGUOUS:
        - expert mini-batch size b'
        - numerical value / exact role of lambda_JSD
        - exact expert probability representation
        - exact reduction across the 18 RBG branches
        - printed Eq. (5) versus mathematically
          standard Jensen-Shannon divergence

    Current primary reproduction:
        divergence_mode="true_jsd"
        one-hot expert action labels
        mean across RBGs and batch
        guidance_weight explicitly supplied
    """

    batch_size: int

    guidance_weight: float

    divergence_mode: PPOExpertDivergenceMode

    rbg_reduction: PPOExpertRBGReduction = "mean"

    normalize_true_jsd_to_unit_interval: bool = True

    printed_kl_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if (
            not math.isfinite(
                self.guidance_weight
            )
            or self.guidance_weight < 0.0
        ):
            raise ValueError(
                "guidance_weight must be "
                "non-negative and finite."
            )

        if self.divergence_mode not in (
            "true_jsd",
            "printed_symmetric_kl_smoothed",
        ):
            raise ValueError(
                "Unsupported divergence_mode."
            )

        if self.rbg_reduction not in (
            "mean",
            "sum",
        ):
            raise ValueError(
                "rbg_reduction must be "
                "'mean' or 'sum'."
            )

        if (
            not math.isfinite(
                self.printed_kl_epsilon
            )
            or self.printed_kl_epsilon <= 0.0
        ):
            raise ValueError(
                "printed_kl_epsilon must be "
                "positive and finite."
            )




@dataclass(frozen=True)
class PPOExpertGuidanceLossData:
    """
    Detailed Teacher-2 loss information.

    actor_probabilities:
        [batch, RBG, action]

    expert_probabilities:
        [batch, RBG, action]

        Current reproduction converts each stored
        deterministic PF expert action into a one-hot
        distribution.

    divergence_by_rbg:
        [batch, RBG]

    raw_loss:
        scalar divergence after RBG and batch
        reduction.

    weighted_loss:
        scalar value used for backward():

            guidance_weight * raw_loss
    """

    actor_probabilities: torch.Tensor

    expert_probabilities: torch.Tensor

    divergence_by_rbg: torch.Tensor

    raw_loss: torch.Tensor

    weighted_loss: torch.Tensor


@dataclass(frozen=True)
class PPOExpertGuidanceUpdateData:
    """
    Diagnostics from one actual expert-guidance
    optimizer step.
    """

    loss_data: PPOExpertGuidanceLossData

    batch_size: int


def build_one_hot_expert_distribution(
    *,
    expert_actions: torch.Tensor,
    action_masks: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert deterministic PF expert labels into a
    one-hot probability distribution.

    Inputs:

        expert_actions:
            [batch, RBG]

        action_masks:
            [batch, RBG, action]

    Output:

        expert_probabilities:
            [batch, RBG, action]

    PAPER-INFERRED:
        Algorithm 1 stores an expert ACTION rather
        than a full expert probability vector.

        Therefore the most direct reconstruction is
        a deterministic one-hot expert policy.
    """

    if expert_actions.ndim != 2:
        raise ValueError(
            "expert_actions must have shape "
            "[batch, RBG]."
        )

    if action_masks.ndim != 3:
        raise ValueError(
            "action_masks must have shape "
            "[batch, RBG, action]."
        )

    if action_masks.dtype != torch.bool:
        raise ValueError(
            "action_masks must use torch.bool."
        )

    if tuple(
        expert_actions.shape
    ) != tuple(
        action_masks.shape[:2]
    ):
        raise ValueError(
            "Expert action and mask dimensions "
            "are inconsistent."
        )

    if (
        expert_actions.device
        != action_masks.device
    ):
        raise ValueError(
            "Expert actions and masks must be on "
            "the same device."
        )

    num_actions = int(
        action_masks.shape[-1]
    )

    if (
        torch.is_floating_point(
            expert_actions
        )
        or expert_actions.dtype
        == torch.bool
    ):
        raise ValueError(
            "expert_actions must use an integer "
            "dtype."
        )

    if torch.any(
        expert_actions < 0
    ):
        raise ValueError(
            "Expert actions cannot be negative."
        )

    if torch.any(
        expert_actions >= num_actions
    ):
        raise ValueError(
            "Expert action exceeds action space."
        )

    expert_is_legal = (
        action_masks
        .gather(
            dim=2,
            index=(
                expert_actions
                .unsqueeze(
                    -1
                )
            ),
        )
        .squeeze(
            -1
        )
    )

    if not torch.all(
        expert_is_legal
    ):
        raise ValueError(
            "Expert action is masked as illegal."
        )

    probabilities = torch.zeros(
        action_masks.shape,
        dtype=dtype,
        device=action_masks.device,
    )

    probabilities.scatter_(
        dim=2,
        index=(
            expert_actions
            .unsqueeze(
                -1
            )
        ),
        value=1.0,
    )

    return probabilities


def _kl_divergence(
    *,
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate:

        KL(p || q)

    over the final dimension.

    Terms for which p == 0 contribute exactly zero.

    The caller must ensure q > 0 wherever p > 0.
    """

    positive_p = (
        p > 0.0
    )

    safe_p = torch.where(
        positive_p,
        p,
        torch.ones_like(
            p
        ),
    )

    safe_q = torch.where(
        positive_p,
        q,
        torch.ones_like(
            q
        ),
    )

    terms = torch.where(
        positive_p,
        p
        * (
            torch.log(
                safe_p
            )
            - torch.log(
                safe_q
            )
        ),
        torch.zeros_like(
            p
        ),
    )

    return terms.sum(
        dim=-1
    )


def compute_true_jsd(
    *,
    actor_probabilities: torch.Tensor,
    expert_probabilities: torch.Tensor,
    normalize_to_unit_interval: bool,
) -> torch.Tensor:
    """
    Standard Jensen-Shannon divergence:

        M = 0.5 * (P + Q)

        JSD(P,Q)
            =
        0.5 KL(P || M)
        +
        0.5 KL(Q || M)

    Output:
        [batch, RBG]

    This is finite even when the deterministic
    expert assigns zero probability to most actions.

    If normalize_to_unit_interval=True, divide by
    ln(2), giving a maximum of 1 when natural logs
    are used.

    This matches the paper's textual claim that JSD
    is bounded between 0 and 1.
    """

    mixture = (
        0.5
        * (
            actor_probabilities
            + expert_probabilities
        )
    )

    actor_kl = _kl_divergence(
        p=actor_probabilities,
        q=mixture,
    )

    expert_kl = _kl_divergence(
        p=expert_probabilities,
        q=mixture,
    )

    jsd = (
        0.5 * actor_kl
        + 0.5 * expert_kl
    )

    if normalize_to_unit_interval:
        jsd = (
            jsd
            / math.log(
                2.0
            )
        )

    return jsd


def compute_printed_symmetric_kl_smoothed(
    *,
    actor_probabilities: torch.Tensor,
    expert_probabilities: torch.Tensor,
    action_masks: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Sensitivity implementation corresponding to the
    STRUCTURE printed in Eq. (5):

        0.5 KL(PPO || expert)
        +
        0.5 KL(expert || PPO)

    IMPORTANT:
        This is NOT standard Jensen-Shannon
        divergence.

        With a deterministic one-hot expert, the
        literal expression can be infinite because
        expert probabilities contain zeros.

    Therefore this sensitivity implementation
    explicitly epsilon-smooths BOTH distributions
    over LEGAL actions before computing the printed
    symmetric-KL expression.

    Output:
        [batch, RBG]

    OPEN-REPRODUCTION sensitivity variant only.
    """

    legal = action_masks.to(
        dtype=actor_probabilities.dtype
    )

    actor_smoothed = (
        actor_probabilities
        + epsilon * legal
    )

    expert_smoothed = (
        expert_probabilities
        + epsilon * legal
    )

    actor_smoothed = (
        actor_smoothed
        / actor_smoothed.sum(
            dim=-1,
            keepdim=True,
        )
    )

    expert_smoothed = (
        expert_smoothed
        / expert_smoothed.sum(
            dim=-1,
            keepdim=True,
        )
    )

    actor_to_expert = (
        _kl_divergence(
            p=actor_smoothed,
            q=expert_smoothed,
        )
    )

    expert_to_actor = (
        _kl_divergence(
            p=expert_smoothed,
            q=actor_smoothed,
        )
    )

    return (
        0.5 * actor_to_expert
        + 0.5 * expert_to_actor
    )


def compute_ppo_expert_guidance_loss(
    *,
    actor: OneLDSPPOActor,
    batch: PPOExpertBatch,
    config: PPOExpertGuidanceConfig,
) -> PPOExpertGuidanceLossData:
    """
    Calculate Teacher-2's loss against the CURRENT
    actor.

    No optimizer step occurs here.
    """

    if batch.states.ndim != 2:
        raise ValueError(
            "Expert states must have shape "
            "[batch, state_size]."
        )

    if batch.expert_actions.ndim != 2:
        raise ValueError(
            "Expert actions must have shape "
            "[batch, RBG]."
        )

    if batch.action_masks.ndim != 3:
        raise ValueError(
            "Expert masks must have shape "
            "[batch, RBG, action]."
        )

    batch_size = int(
        batch.states.shape[0]
    )

    if batch_size < 1:
        raise ValueError(
            "Expert batch cannot be empty."
        )

    if batch_size != int(
        batch.expert_actions.shape[0]
    ):
        raise ValueError(
            "Expert batch dimensions disagree."
        )

    if batch_size != int(
        batch.action_masks.shape[0]
    ):
        raise ValueError(
            "Expert mask batch dimension disagrees."
        )

    (
        actor_distribution,
        _,
    ) = actor.build_distribution(
        state=batch.states,
        action_mask=(
            batch.action_masks
        ),
    )

    actor_probabilities = (
        actor_distribution.probs
    )

    expert_probabilities = (
        build_one_hot_expert_distribution(
            expert_actions=(
                batch.expert_actions
            ),
            action_masks=(
                batch.action_masks
            ),
            dtype=(
                actor_probabilities.dtype
            ),
        )
    )

    if (
        config.divergence_mode
        == "true_jsd"
    ):
        divergence_by_rbg = (
            compute_true_jsd(
                actor_probabilities=(
                    actor_probabilities
                ),
                expert_probabilities=(
                    expert_probabilities
                ),
                normalize_to_unit_interval=(
                    config
                    .normalize_true_jsd_to_unit_interval
                ),
            )
        )

    elif (
        config.divergence_mode
        == "printed_symmetric_kl_smoothed"
    ):
        divergence_by_rbg = (
            compute_printed_symmetric_kl_smoothed(
                actor_probabilities=(
                    actor_probabilities
                ),
                expert_probabilities=(
                    expert_probabilities
                ),
                action_masks=(
                    batch.action_masks
                ),
                epsilon=(
                    config
                    .printed_kl_epsilon
                ),
            )
        )

    else:
        raise RuntimeError(
            "Unexpected expert divergence mode."
        )

    if not torch.isfinite(
        divergence_by_rbg
    ).all():
        raise RuntimeError(
            "Expert divergence became non-finite."
        )

    if config.rbg_reduction == "mean":
        loss_per_sample = (
            divergence_by_rbg.mean(
                dim=1
            )
        )

    elif config.rbg_reduction == "sum":
        loss_per_sample = (
            divergence_by_rbg.sum(
                dim=1
            )
        )

    else:
        raise RuntimeError(
            "Unexpected RBG reduction."
        )

    raw_loss = (
        loss_per_sample.mean()
    )

    weighted_loss = (
        config.guidance_weight
        * raw_loss
    )

    return PPOExpertGuidanceLossData(
        actor_probabilities=(
            actor_probabilities
        ),
        expert_probabilities=(
            expert_probabilities
        ),
        divergence_by_rbg=(
            divergence_by_rbg
        ),
        raw_loss=raw_loss,
        weighted_loss=(
            weighted_loss
        ),
    )


def perform_ppo_expert_guidance_update(
    *,
    actor: OneLDSPPOActor,
    actor_optimizer: Optimizer,
    expert_buffer: PPOExpertDemonstrationBuffer,
    config: PPOExpertGuidanceConfig,
    generator: torch.Generator | None = None,
) -> PPOExpertGuidanceUpdateData:
    """
    Perform one actual Teacher-2 actor update.

    Algorithmically:

        sample D_expert
            ->
        current actor distribution
            ->
        expert-guidance divergence
            ->
        zero_grad()
            ->
        backward()
            ->
        actor optimizer step

    The critic is intentionally absent.
    """

    batch = expert_buffer.sample(
        batch_size=(
            config.batch_size
        ),
        generator=generator,
    )

    loss_data = (
        compute_ppo_expert_guidance_loss(
            actor=actor,
            batch=batch,
            config=config,
        )
    )

    if not torch.isfinite(
        loss_data.weighted_loss
    ):
        raise RuntimeError(
            "Expert-guidance loss is non-finite."
        )

    actor_optimizer.zero_grad(
        set_to_none=True
    )

    loss_data.weighted_loss.backward()

    has_gradient = False

    for parameter in actor.parameters():
        if parameter.grad is None:
            continue

        has_gradient = True

        if not torch.isfinite(
            parameter.grad
        ).all():
            raise RuntimeError(
                "Expert guidance produced a "
                "non-finite actor gradient."
            )

    if not has_gradient:
        raise RuntimeError(
            "Expert guidance produced no actor "
            "gradients."
        )

    actor_optimizer.step()

    return PPOExpertGuidanceUpdateData(
        loss_data=loss_data,
        batch_size=(
            config.batch_size
        ),
    )



        