from dataclasses import dataclass

import torch

from oran_scheduler.rl.ppo_expert_buffer import (
    PPOExpertDemonstrationBuffer,
)
from oran_scheduler.rl.ppo_pf_expert import (
    PPOPFExpertConfig,
    generate_ppo_pf_expert_actions,
)

from oran_scheduler.rl.ppo_candidate_permutation import (
    PPOCandidateAugmentationConfig,
    iter_candidate_permutations,
    permute_ppo_expert_data,
)

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
    apply_user_slot_actions,
    build_empty_cell_allocation,
)
from oran_scheduler.schedulers.one_lds_loop import (
    OneLDSScheduleResult,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    RBGScoreFunction,
)


@dataclass(frozen=True)
class PPOPFExpertLayerLabel:
    """
    Temporary PF-expert supervision for one 1LDS
    user-layer decision.

    This corresponds conceptually to the expert
    information stored in Algorithm 1's temporary
    TTI trajectory before it is copied into
    D_expert.

    state:
        [state_size]

    expert_actions:
        [RBG]

    action_mask:
        [RBG, action]

    best_pf_sum:
        [RBG]
    """

    user_slot_index: int

    state: torch.Tensor

    expert_actions: torch.Tensor

    action_mask: torch.Tensor

    best_pf_sum: torch.Tensor

    num_phy_evaluations: int


@dataclass(frozen=True)
class PPOPFExpertTTILabels:
    """
    All expert labels generated for one complete
    cell TTI.
    """

    layers: tuple[
        PPOPFExpertLayerLabel,
        ...
    ]

    total_num_phy_evaluations: int


def generate_ppo_pf_expert_tti_labels(
    *,
    schedule: OneLDSScheduleResult,
    num_candidates: int,
    past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    score_rbg: RBGScoreFunction,
    config: PPOPFExpertConfig,
) -> PPOPFExpertTTILabels:
    """
    Reconstruct each exact pre-action PPO layer state
    and generate the corresponding PF expert label.

    IMPORTANT:
        The final PPO schedule has already been
        produced when this helper is called.

        However, expert generation is reconstructed
        layer-by-layer using ONLY the PPO actions
        that occurred BEFORE the current layer.

        Therefore future PPO allocations are never
        exposed to the expert.

    Conceptually this is equivalent to Algorithm 1:

        observe s_t,l
            ->
        PPO action
            +
        expert action
            ->
        execute PPO action

    The reconstruction avoids coupling the generic
    1LDS scheduling loop directly to PPO-specific
    expert logic.
    """

    if schedule.actions.ndim != 2:
        raise ValueError(
            "schedule.actions must have shape "
            "[user_slot, RBG]."
        )

    num_user_slots = int(
        schedule.actions.shape[0]
    )

    num_rbgs = int(
        schedule.actions.shape[1]
    )

    if len(
        schedule.decisions
    ) != num_user_slots:
        raise ValueError(
            "Schedule decision count does not match "
            "the number of user slots."
        )

    if (
        schedule
        .allocation
        .num_user_slots
        != num_user_slots
    ):
        raise ValueError(
            "Final allocation user-slot count does "
            "not match schedule actions."
        )

    if (
        schedule
        .allocation
        .num_rbgs
        != num_rbgs
    ):
        raise ValueError(
            "Final allocation RBG count does not "
            "match schedule actions."
        )

    device = schedule.actions.device

    allocation = (
        build_empty_cell_allocation(
            num_user_slots=(
                num_user_slots
            ),
            num_rbgs=num_rbgs,
            device=device,
        )
    )

    layer_labels: list[
        PPOPFExpertLayerLabel
    ] = []

    total_num_phy_evaluations = 0

    for user_slot_index in range(
        num_user_slots
    ):
        decision = (
            schedule
            .decisions[
                user_slot_index
            ]
        )

        #
        # Generate Teacher-2's answer while the
        # reconstructed allocation contains only
        # PREVIOUS PPO decisions.
        #
        expert = (
            generate_ppo_pf_expert_actions(
                allocation=allocation,
                user_slot_index=(
                    user_slot_index
                ),
                num_candidates=(
                    num_candidates
                ),
                past_average_throughput=(
                    past_average_throughput
                ),
                candidate_valid_mask=(
                    candidate_valid_mask
                ),
                score_rbg=score_rbg,
                config=config,
            )
        )

        #
        # Extremely important consistency check:
        #
        # Teacher and PPO must have been looking at
        # the same legal action space.
        #
        if not torch.equal(
            expert.action_mask,
            decision.action_mask,
        ):
            raise RuntimeError(
                "PF expert and PPO decision used "
                "different action masks."
            )

        layer_labels.append(
            PPOPFExpertLayerLabel(
                user_slot_index=(
                    user_slot_index
                ),
                state=(
                    decision
                    .state_data
                    .state
                    .detach()
                    .clone()
                ),
                expert_actions=(
                    expert
                    .expert_actions
                    .detach()
                    .clone()
                ),
                action_mask=(
                    expert
                    .action_mask
                    .detach()
                    .clone()
                ),
                best_pf_sum=(
                    expert
                    .best_pf_sum
                    .detach()
                    .clone()
                ),
                num_phy_evaluations=(
                    expert
                    .num_phy_evaluations
                ),
            )
        )

        total_num_phy_evaluations += (
            expert.num_phy_evaluations
        )

        #
        # Now replay the ACTUAL PPO action.
        #
        # This produces exactly the allocation that
        # existed before the next PPO layer.
        #
        allocation = apply_user_slot_actions(
            allocation=allocation,
            user_slot_index=(
                user_slot_index
            ),
            actions=(
                schedule
                .actions[
                    user_slot_index,
                    :,
                ]
            ),
            num_candidates=(
                num_candidates
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
        )

    #
    # Final reconstruction must reproduce exactly
    # what the real PPO scheduler executed.
    #
    if not torch.equal(
        allocation
        .candidate_by_user_slot,
        schedule
        .allocation
        .candidate_by_user_slot,
    ):
        raise RuntimeError(
            "Layer-by-layer PPO reconstruction did "
            "not reproduce the final schedule."
        )

    return PPOPFExpertTTILabels(
        layers=tuple(
            layer_labels
        ),
        total_num_phy_evaluations=(
            total_num_phy_evaluations
        ),
    )

def commit_ppo_pf_expert_tti_labels(
    *,
    labels: PPOPFExpertTTILabels,
    expert_buffer: PPOExpertDemonstrationBuffer,
    augmentation_config: (
        PPOCandidateAugmentationConfig | None
    ) = None,
    augmentation_generator: (
        torch.Generator | None
    ) = None,
) -> int:
    """
    Copy one TTI's temporary expert supervision into
    the persistent expert demonstration buffer.

    Without augmentation:
        behavior is identical to the original
        implementation:

            one expert demonstration per layer.

    With augmentation:
        each physical layer produces the explicitly
        configured number of candidate-order
        representations.

    Returns:
        Number of demonstrations accepted by the
        expert buffer.

    IMPORTANT:
        Expert augmentation changes only the
        representation.

        It does NOT:
            - rerun the PF expert
            - rerun the PHY
            - change the reward
            - change the actual PPO schedule
    """

    num_added = 0

    for layer in labels.layers:
        #
        # Backward-compatible path.
        #
        if augmentation_config is None:
            accepted = expert_buffer.add(
                state=layer.state,
                expert_actions=(
                    layer.expert_actions
                ),
                action_mask=(
                    layer.action_mask
                ),
            )

            if accepted:
                num_added += 1

            continue

        num_candidates = (
            int(
                layer.action_mask.shape[-1]
            )
            - 1
        )

        for permutation in (
            iter_candidate_permutations(
                config=(
                    augmentation_config
                ),
                num_candidates=(
                    num_candidates
                ),
                device=layer.state.device,
                generator=(
                    augmentation_generator
                ),
            )
        ):
            if permutation is None:
                state = (
                    layer.state
                )

                expert_actions = (
                    layer.expert_actions
                )

                action_mask = (
                    layer.action_mask
                )

            else:
                permuted = (
                    permute_ppo_expert_data(
                        state=layer.state,
                        expert_actions=(
                            layer.expert_actions
                        ),
                        action_mask=(
                            layer.action_mask
                        ),
                        permutation=(
                            permutation
                        ),
                    )
                )

                state = (
                    permuted.state
                )

                expert_actions = (
                    permuted.expert_actions
                )

                action_mask = (
                    permuted.action_mask
                )

            accepted = expert_buffer.add(
                state=state,
                expert_actions=(
                    expert_actions
                ),
                action_mask=(
                    action_mask
                ),
            )

            if accepted:
                num_added += 1

    return num_added




















