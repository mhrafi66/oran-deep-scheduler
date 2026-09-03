from dataclasses import (
    dataclass,
    is_dataclass,
    replace,
)

import torch

from oran_scheduler.rl.ppo_actor import (
    OneLDSPPOActor,
    compute_joint_action_log_prob,
)
from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
    iter_candidate_permutations,
    permute_ppo_decision_data,
)
from oran_scheduler.rl.ppo_rollout import (
    PPORolloutBatch,
)


@dataclass(frozen=True)
class PPOAugmentedUpdateData:
    """
    PPO optimization data after candidate-order
    augmentation.

    IMPORTANT:
        The real temporal trajectory has already
        been used to calculate GAE before reaching
        this stage.

    Therefore:

        state/action/mask:
            permuted

        old action probability:
            recomputed under the unchanged
            behavior actor

        advantage:
            preserved

        target return:
            preserved

    The permutation changes only the representation
    of the same physical scheduling experience.
    """

    batch: PPORolloutBatch

    advantage: torch.Tensor

    target_return: torch.Tensor

    num_original_transitions: int

    num_optimizer_samples: int


def _validate_augmentation_inputs(
    *,
    batch: PPORolloutBatch,
    advantage: torch.Tensor,
    target_return: torch.Tensor,
) -> int:
    if not is_dataclass(
        batch
    ):
        raise TypeError(
            "PPORolloutBatch must remain a "
            "dataclass for safe augmentation."
        )

    if batch.states.ndim != 2:
        raise ValueError(
            "batch.states must have shape "
            "[transition, state_size]."
        )

    num_transitions = int(
        batch.states.shape[0]
    )

    if num_transitions < 1:
        raise ValueError(
            "Cannot augment an empty PPO batch."
        )

    if batch.actions.ndim != 2:
        raise ValueError(
            "batch.actions must have shape "
            "[transition, RBG]."
        )

    if batch.action_masks.ndim != 3:
        raise ValueError(
            "batch.action_masks must have shape "
            "[transition, RBG, action]."
        )

    if tuple(
        batch.actions.shape[:1]
    ) != (
        num_transitions,
    ):
        raise ValueError(
            "PPO action batch size is inconsistent."
        )

    if int(
        batch.action_masks.shape[0]
    ) != num_transitions:
        raise ValueError(
            "PPO mask batch size is inconsistent."
        )

    if tuple(
        batch.old_log_prob_by_rbg.shape
    ) != tuple(
        batch.actions.shape
    ):
        raise ValueError(
            "old_log_prob_by_rbg must match "
            "the PPO action shape."
        )

    if tuple(
        batch.old_joint_log_prob.shape
    ) != (
        num_transitions,
    ):
        raise ValueError(
            "old_joint_log_prob must have shape "
            "[transition]."
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

    return num_transitions


def build_candidate_augmented_ppo_update(
    *,
    actor: OneLDSPPOActor,
    batch: PPORolloutBatch,
    advantage: torch.Tensor,
    target_return: torch.Tensor,
    config: (
        PPOCandidateAugmentationConfig | None
    ),
    generator: (
        torch.Generator | None
    ) = None,
) -> PPOAugmentedUpdateData:
    """
    Build PPO optimizer samples from a completed
    temporal rollout.

    WITHOUT augmentation:
        return the original PPO optimization data.

    WITH augmentation:
        for every real transition:

            1. optionally retain the original
            2. create N_Pi candidate permutations
            3. remap the selected actions
            4. permute the action mask
            5. recompute old behavior log-probability
               under the STILL-UNCHANGED actor
            6. preserve advantage and return target

    Why recompute log probability?

        In general:

            pi_old(a' | permuted(s))
                !=
            pi_old(a | s)

        until the actor has actually learned
        permutation robustness.

    Why NOT recompute GAE here?

        The shuffled state is another representation
        of the SAME physical transition.

        Reward, return and advantage therefore remain
        training labels for that same experience.

        Re-running GAE independently for shuffled
        rows would also destroy the true temporal
        trajectory structure.
    """

    num_transitions = (
        _validate_augmentation_inputs(
            batch=batch,
            advantage=advantage,
            target_return=target_return,
        )
    )

    if config is None:
        return PPOAugmentedUpdateData(
            batch=batch,
            advantage=advantage,
            target_return=target_return,
            num_original_transitions=(
                num_transitions
            ),
            num_optimizer_samples=(
                num_transitions
            ),
        )

    num_candidates = (
        int(
            batch.action_masks.shape[-1]
        )
        - 1
    )

    if num_candidates <= 0:
        raise ValueError(
            "PPO action space must contain at least "
            "one candidate plus NO ALLOCATION."
        )

    states: list[
        torch.Tensor
    ] = []

    actions: list[
        torch.Tensor
    ] = []

    action_masks: list[
        torch.Tensor
    ] = []

    old_log_prob_by_rbg: list[
        torch.Tensor
    ] = []

    old_joint_log_prob: list[
        torch.Tensor
    ] = []

    advantages: list[
        torch.Tensor
    ] = []

    target_returns: list[
        torch.Tensor
    ] = []

    for transition_index in range(
        num_transitions
    ):
        original_state = (
            batch.states[
                transition_index
            ]
        )

        original_actions = (
            batch.actions[
                transition_index
            ]
        )

        original_mask = (
            batch.action_masks[
                transition_index
            ]
        )

        for permutation in (
            iter_candidate_permutations(
                config=config,
                num_candidates=(
                    num_candidates
                ),
                device=(
                    original_state.device
                ),
                generator=generator,
            )
        ):
            if permutation is None:
                #
                # Original experience.
                #
                # The behavior-policy quantities were
                # already captured at action time.
                #
                augmented_state = (
                    original_state
                )

                augmented_actions = (
                    original_actions
                )

                augmented_mask = (
                    original_mask
                )

                branch_log_prob = (
                    batch
                    .old_log_prob_by_rbg[
                        transition_index
                    ]
                )

                joint_log_prob = (
                    batch
                    .old_joint_log_prob[
                        transition_index
                    ]
                )

            else:
                permuted = (
                    permute_ppo_decision_data(
                        state=(
                            original_state
                        ),
                        actions=(
                            original_actions
                        ),
                        action_mask=(
                            original_mask
                        ),
                        permutation=(
                            permutation
                        ),
                    )
                )

                augmented_state = (
                    permuted.state
                )

                augmented_actions = (
                    permuted.actions
                )

                augmented_mask = (
                    permuted.action_mask
                )

                #
                # CRITICAL PPO DETAIL:
                #
                # The actor has not been updated yet,
                # so it is still exactly pi_old.
                #
                # Evaluate the transformed action
                # under transformed state/mask.
                #
                with torch.no_grad():
                    (
                        evaluated_log_prob,
                        _,
                    ) = actor.evaluate_actions(
                        state=(
                            augmented_state
                            .unsqueeze(
                                0
                            )
                        ),
                        action_mask=(
                            augmented_mask
                            .unsqueeze(
                                0
                            )
                        ),
                        actions=(
                            augmented_actions
                            .unsqueeze(
                                0
                            )
                        ),
                    )

                    branch_log_prob = (
                        evaluated_log_prob[
                            0
                        ]
                    )

                    joint_log_prob = (
                        compute_joint_action_log_prob(
                            evaluated_log_prob
                        )[
                            0
                        ]
                    )

            states.append(
                augmented_state
                .detach()
                .clone()
            )

            actions.append(
                augmented_actions
                .detach()
                .clone()
            )

            action_masks.append(
                augmented_mask
                .detach()
                .clone()
            )

            old_log_prob_by_rbg.append(
                branch_log_prob
                .detach()
                .clone()
            )

            old_joint_log_prob.append(
                joint_log_prob
                .detach()
                .clone()
            )

            #
            # Same physical experience.
            #
            advantages.append(
                advantage[
                    transition_index
                ]
                .detach()
                .clone()
            )

            target_returns.append(
                target_return[
                    transition_index
                ]
                .detach()
                .clone()
            )

    augmented_states = torch.stack(
        states,
        dim=0,
    )

    augmented_actions = torch.stack(
        actions,
        dim=0,
    )

    augmented_masks = torch.stack(
        action_masks,
        dim=0,
    )

    augmented_branch_log_prob = (
        torch.stack(
            old_log_prob_by_rbg,
            dim=0,
        )
    )

    augmented_joint_log_prob = (
        torch.stack(
            old_joint_log_prob,
            dim=0,
        )
    )

    augmented_advantage = torch.stack(
        advantages,
        dim=0,
    )

    augmented_target_return = (
        torch.stack(
            target_returns,
            dim=0,
        )
    )

    #
    # PPORolloutBatch may contain other trajectory
    # diagnostics that PPO optimization does not use.
    #
    # dataclasses.replace lets us preserve those
    # fields while replacing exactly the optimizer-
    # relevant tensors.
    #
    augmented_batch = replace(
        batch,
        states=augmented_states,
        actions=augmented_actions,
        action_masks=augmented_masks,
        old_log_prob_by_rbg=(
            augmented_branch_log_prob
        ),
        old_joint_log_prob=(
            augmented_joint_log_prob
        ),
    )

    return PPOAugmentedUpdateData(
        batch=augmented_batch,
        advantage=augmented_advantage,
        target_return=(
            augmented_target_return
        ),
        num_original_transitions=(
            num_transitions
        ),
        num_optimizer_samples=int(
            augmented_states.shape[0]
        ),
    )


