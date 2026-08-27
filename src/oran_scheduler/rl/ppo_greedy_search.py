from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    CellAllocation,
    apply_user_slot_actions,
    build_empty_cell_allocation,
    build_next_user_slot_action_mask,
    validate_cell_allocation,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    RBGScoreFunction,
    compute_pf_sum,
)

@dataclass(frozen=True)
class PPOGreedySearchConfig:
    """
    Configuration for the PPO reward's greedy PF comparison.

    pf_denominator_epsilon:
        Numerical protection for the PF denominator.

    pf_comparison_epsilon:
        Optional tolerance when deciding whether an
        alternative PF sum is strictly better.

    The public paper does not specify either numerical
    tolerance. They are reproduction/numerical parameters.
    """

    pf_denominator_epsilon: float = 1.0e-8
    pf_comparison_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.pf_denominator_epsilon <= 0.0:
            raise ValueError(
                "pf_denominator_epsilon must be positive."
            )

        if self.pf_comparison_epsilon < 0.0:
            raise ValueError(
                "pf_comparison_epsilon cannot be negative."
            )


@dataclass(frozen=True)
class PPOLayerGreedySearchData:
    """
    Greedy-search result for one 1LDS user layer.

    better_allocation_exists:
        [RBG], bool.

        True means at least one legal alternative action
        has a larger PF sum than the action selected by
        the deep scheduler.

    chosen_pf_sum:
        [RBG].

        PF sum resulting from the deep scheduler action.

    best_pf_sum:
        [RBG].

        Largest PF sum among the deep scheduler action
        and all legal alternatives.

    num_phy_evaluations:
        Number of candidate-set scoring calls performed.
    """

    better_allocation_exists: torch.Tensor

    chosen_pf_sum: torch.Tensor

    best_pf_sum: torch.Tensor

    num_phy_evaluations: int


@dataclass(frozen=True)
class PPOScheduleGreedySearchData:
    """
    Greedy PF comparison results for a complete
    1LDS schedule.

    allocation:
        Reconstructed final CellAllocation produced
        by replaying the actor actions layer by layer.

    better_allocation_exists:
        Shape [user_layer, RBG].

        True means the greedy PF search found a
        strictly better legal action for that specific
        layer/RBG decision.

    chosen_pf_sum:
        Shape [user_layer, RBG].

        PF sum resulting from the actor's actual action.

    best_pf_sum:
        Shape [user_layer, RBG].

        Best PF sum found among the actor's action and
        all legal alternatives.

    num_phy_evaluations:
        Total number of hypothetical candidate sets
        scored across all layers and RBGs.
    """

    allocation: CellAllocation

    better_allocation_exists: torch.Tensor

    chosen_pf_sum: torch.Tensor

    best_pf_sum: torch.Tensor

    num_phy_evaluations: int


def _build_candidate_set_for_action(
    *,
    previous_candidates: torch.Tensor,
    action: int,
    num_candidates: int,
) -> torch.Tensor:
    """
    Construct the scheduled candidate set after one action.

    Actor convention:

        0 ... num_candidates - 1
            select that candidate

        num_candidates
            NO ALLOCATION

    Previous user-layer allocations are always preserved.
    """

    if action == num_candidates:
        return previous_candidates

    candidate_tensor = torch.tensor(
        [action],
        dtype=torch.long,
        device=previous_candidates.device,
    )

    return torch.cat(
        (
            previous_candidates,
            candidate_tensor,
        ),
        dim=0,
    )


def _validate_layer_search_inputs(
    *,
    allocation: CellAllocation,
    user_slot_index: int,
    chosen_actions: torch.Tensor,
    num_candidates: int,
    past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> None:
    """
    Validate one layer of PPO greedy-search inputs.
    """

    validate_cell_allocation(
        allocation=allocation,
        num_candidates=num_candidates,
    )

    if not (
        0
        <= user_slot_index
        < allocation.num_user_slots
    ):
        raise ValueError(
            "user_slot_index is invalid."
        )

    if tuple(
        chosen_actions.shape
    ) != (
        allocation.num_rbgs,
    ):
        raise ValueError(
            "chosen_actions must have shape [RBG]."
        )

    if torch.is_floating_point(
        chosen_actions
    ):
        raise ValueError(
            "chosen_actions must use an integer dtype."
        )

    if tuple(
        past_average_throughput.shape
    ) != (
        num_candidates,
    ):
        raise ValueError(
            "past_average_throughput must have "
            "shape [candidate]."
        )

    if not torch.is_floating_point(
        past_average_throughput
    ):
        raise ValueError(
            "past_average_throughput must use a "
            "floating-point dtype."
        )

    if not torch.isfinite(
        past_average_throughput
    ).all():
        raise ValueError(
            "past_average_throughput contains "
            "non-finite values."
        )

    if torch.any(
        past_average_throughput < 0.0
    ):
        raise ValueError(
            "Past average throughput cannot be negative."
        )

    if tuple(
        candidate_valid_mask.shape
    ) != (
        num_candidates,
    ):
        raise ValueError(
            "candidate_valid_mask must have "
            "shape [candidate]."
        )

    if candidate_valid_mask.dtype != torch.bool:
        raise ValueError(
            "candidate_valid_mask must have dtype bool."
        )

    device = (
        allocation
        .candidate_by_user_slot
        .device
    )

    if chosen_actions.device != device:
        raise ValueError(
            "chosen_actions must be on the same device "
            "as allocation."
        )

    if past_average_throughput.device != device:
        raise ValueError(
            "past_average_throughput must be on the "
            "same device as allocation."
        )

    if candidate_valid_mask.device != device:
        raise ValueError(
            "candidate_valid_mask must be on the same "
            "device as allocation."
        )

    if torch.any(
        chosen_actions < 0
    ):
        raise ValueError(
            "chosen_actions contains a negative action."
        )

    if torch.any(
        chosen_actions > num_candidates
    ):
        raise ValueError(
            "chosen_actions contains an invalid action."
        )


    remaining_slots = (
        allocation
        .candidate_by_user_slot[
            user_slot_index:,
            :,
        ]
    )

    if torch.any(
        remaining_slots != NO_ALLOCATION
    ):
        raise ValueError(
            "The current and later user slots must be "
            "empty before evaluating this layer."
        )

    legal_action_mask = (
        build_next_user_slot_action_mask(
            allocation=allocation,
            num_candidates=num_candidates,
            user_slot_index=user_slot_index,
            candidate_valid_mask=(
                candidate_valid_mask
            ),
        )
    )

    rbg_indices = torch.arange(
        allocation.num_rbgs,
        dtype=torch.long,
        device=device,
    )

    chosen_is_legal = (
        legal_action_mask[
            rbg_indices,
            chosen_actions,
        ]
    )

    if not torch.all(
        chosen_is_legal
    ):
        raise ValueError(
            "chosen_actions contains an action that is "
            "masked for the current user layer."
        )


def evaluate_ppo_layer_greedy_search(
    *,
    allocation: CellAllocation,
    user_slot_index: int,
    chosen_actions: torch.Tensor,
    num_candidates: int,
    past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    score_rbg: RBGScoreFunction,
    config: PPOGreedySearchConfig,
) -> PPOLayerGreedySearchData:
    """
    Compare one 1LDS user-layer decision against all
    legal single-layer alternatives.

    PAPER-SPECIFIED:
        v_m = -1 if a better allocation choice exists,
        where the alternative has a larger PF metric than
        the deep scheduler's choice.

    PAPER-INFERRED reproduction interpretation:
        - previous user-layer allocations are held fixed;
        - all legal actions for the current layer are tested;
        - NO ALLOCATION is included as a legal alternative;
        - comparison uses total PF sum of the resulting
          scheduled candidate set;
        - unlike the separate PF-Greedy reference scheduler,
          no additional raw-throughput-improvement gate is
          imposed here.
    """

    _validate_layer_search_inputs(
        allocation=allocation,
        user_slot_index=user_slot_index,
        chosen_actions=chosen_actions,
        num_candidates=num_candidates,
        past_average_throughput=(
            past_average_throughput
        ),
        candidate_valid_mask=(
            candidate_valid_mask
        ),
    )

    device = (
        allocation
        .candidate_by_user_slot
        .device
    )

    dtype = (
        past_average_throughput.dtype
    )

    num_rbgs = allocation.num_rbgs

    chosen_pf_sum = torch.zeros(
        num_rbgs,
        dtype=dtype,
        device=device,
    )

    best_pf_sum = torch.zeros(
        num_rbgs,
        dtype=dtype,
        device=device,
    )

    better_allocation_exists = torch.zeros(
        num_rbgs,
        dtype=torch.bool,
        device=device,
    )

    legal_action_mask = (
        build_next_user_slot_action_mask(
            allocation=allocation,
            num_candidates=num_candidates,
            user_slot_index=user_slot_index,
            candidate_valid_mask=(
                candidate_valid_mask
            ),
        )
    )

    num_phy_evaluations = 0


    for rbg_index in range(
        num_rbgs
    ):
        previous_actions = (
            allocation
            .candidate_by_user_slot[
                :user_slot_index,
                rbg_index,
            ]
        )

        previous_candidates = (
            previous_actions[
                previous_actions
                != NO_ALLOCATION
            ]
        )

        chosen_action = int(
            chosen_actions[
                rbg_index
            ].item()
        )

        chosen_candidates = (
            _build_candidate_set_for_action(
                previous_candidates=(
                    previous_candidates
                ),
                action=chosen_action,
                num_candidates=num_candidates,
            )
        )

        chosen_score = score_rbg(
            chosen_candidates,
            rbg_index,
        )

        num_phy_evaluations += 1

        chosen_pf = compute_pf_sum(
            score=chosen_score,
            selected_candidates=(
                chosen_candidates
            ),
            past_average_throughput=(
                past_average_throughput
            ),
            denominator_epsilon=(
                config
                .pf_denominator_epsilon
            ),
        )

        chosen_pf_sum[
            rbg_index
        ] = chosen_pf

        best_pf = chosen_pf


        for alternative_action in range(
            num_candidates + 1
        ):
            if alternative_action == chosen_action:
                continue

            alternative_is_legal = bool(
                legal_action_mask[
                    rbg_index,
                    alternative_action,
                ].item()
            )

            if not alternative_is_legal:
                continue

            alternative_candidates = (
                _build_candidate_set_for_action(
                    previous_candidates=(
                        previous_candidates
                    ),
                    action=alternative_action,
                    num_candidates=(
                        num_candidates
                    ),
                )
            )

            alternative_score = score_rbg(
                alternative_candidates,
                rbg_index,
            )

            num_phy_evaluations += 1

            alternative_pf = compute_pf_sum(
                score=alternative_score,
                selected_candidates=(
                    alternative_candidates
                ),
                past_average_throughput=(
                    past_average_throughput
                ),
                denominator_epsilon=(
                    config
                    .pf_denominator_epsilon
                ),
            )

            if alternative_pf > best_pf:
                best_pf = alternative_pf


        best_pf_sum[
            rbg_index
        ] = best_pf

        required_pf = (
            chosen_pf
            + config.pf_comparison_epsilon
        )

        better_allocation_exists[
            rbg_index
        ] = (
            best_pf > required_pf
        )

    return PPOLayerGreedySearchData(
        better_allocation_exists=(
            better_allocation_exists
        ),
        chosen_pf_sum=chosen_pf_sum,
        best_pf_sum=best_pf_sum,
        num_phy_evaluations=(
            num_phy_evaluations
        ),
    )


def evaluate_ppo_schedule_greedy_search(
    *,
    actions: torch.Tensor,
    num_candidates: int,
    past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    score_rbg: RBGScoreFunction,
    config: PPOGreedySearchConfig,
) -> PPOScheduleGreedySearchData:
    """
    Replay a complete 1LDS action sequence and compute
    the PPO greedy-search result for every layer/RBG.

    Args:
        actions:
            Shape [user_layer, RBG].

            Actor convention:

                0 ... K-1
                    candidate index

                K
                    NO ALLOCATION

        num_candidates:
            Maximum candidate count K.

        past_average_throughput:
            Shape [candidate].

        candidate_valid_mask:
            Shape [candidate].

        score_rbg:
            Callback that physically scores an arbitrary
            scheduled candidate set on one RBG.

    Returns:
        Complete greedy-search information with shape

            [user_layer, RBG]

        for the per-decision quantities.
    """

    if actions.ndim != 2:
        raise ValueError(
            "actions must have shape "
            "[user_layer, RBG]."
        )

    if actions.shape[0] < 1:
        raise ValueError(
            "At least one user layer is required."
        )

    if actions.shape[1] < 1:
        raise ValueError(
            "At least one RBG is required."
        )

    if torch.is_floating_point(
        actions
    ):
        raise ValueError(
            "actions must use an integer dtype."
        )

    num_user_slots = int(
        actions.shape[0]
    )

    num_rbgs = int(
        actions.shape[1]
    )

    device = actions.device

    dtype = (
        past_average_throughput.dtype
    )

    allocation = build_empty_cell_allocation(
        num_user_slots=num_user_slots,
        num_rbgs=num_rbgs,
        device=device,
    )

    better_allocation_exists = torch.zeros(
        (
            num_user_slots,
            num_rbgs,
        ),
        dtype=torch.bool,
        device=device,
    )

    chosen_pf_sum = torch.zeros(
        (
            num_user_slots,
            num_rbgs,
        ),
        dtype=dtype,
        device=device,
    )

    best_pf_sum = torch.zeros(
        (
            num_user_slots,
            num_rbgs,
        ),
        dtype=dtype,
        device=device,
    )

    num_phy_evaluations = 0

    for user_slot_index in range(
        num_user_slots
    ):
        chosen_actions = actions[
            user_slot_index,
            :,
        ]

        layer_result = (
            evaluate_ppo_layer_greedy_search(
                allocation=allocation,
                user_slot_index=(
                    user_slot_index
                ),
                chosen_actions=chosen_actions,
                num_candidates=num_candidates,
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


        better_allocation_exists[
            user_slot_index,
            :,
        ] = (
            layer_result
            .better_allocation_exists
        )

        chosen_pf_sum[
            user_slot_index,
            :,
        ] = (
            layer_result
            .chosen_pf_sum
        )

        best_pf_sum[
            user_slot_index,
            :,
        ] = (
            layer_result
            .best_pf_sum
        )

        num_phy_evaluations += (
            layer_result
            .num_phy_evaluations
        )


        allocation = apply_user_slot_actions(
            allocation=allocation,
            user_slot_index=(
                user_slot_index
            ),
            actions=chosen_actions,
            num_candidates=num_candidates,
            candidate_valid_mask=(
                candidate_valid_mask
            ),
        )



    return PPOScheduleGreedySearchData(
        allocation=allocation,
        better_allocation_exists=(
            better_allocation_exists
        ),
        chosen_pf_sum=chosen_pf_sum,
        best_pf_sum=best_pf_sum,
        num_phy_evaluations=(
            num_phy_evaluations
        ),
    )



