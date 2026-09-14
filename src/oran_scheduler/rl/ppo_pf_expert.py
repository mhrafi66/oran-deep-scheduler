from dataclasses import dataclass
from typing import Literal

import torch

from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    CellAllocation,
    build_next_user_slot_action_mask,
    validate_cell_allocation,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    RBGScoreFunction,
    compute_pf_sum,
)


PFExpertTieBreaking = Literal[
    "lowest_action_index",
    "no_allocation",
]


@dataclass(frozen=True)
class PPOPFExpertConfig:
    """
    PF expert used to generate supervision labels
    for 1LDS-PPO expert guidance.

    PAPER-SPECIFIED:
        The expert is a PF scheduler.

    PAPER-UNSPECIFIED:
        - exact expert search implementation
        - exact numerical PF denominator protection
        - exact tie-breaking behavior

    OPEN-REPRODUCTION interpretation:
        For each RBG at the current 1LDS layer:

            1. hold previous PPO-executed user slots
               fixed;
            2. consider every currently legal action,
               including NO ALLOCATION;
            3. physically evaluate the resulting
               candidate set;
            4. calculate its total PF sum;
            5. choose the action with maximum PF sum.

    Unlike the classical PF-Greedy SDS baseline,
    this expert does not impose an additional
    raw-throughput-improvement gate.
    """

    pf_denominator_epsilon: float = 1.0e-8

    pf_tie_epsilon: float = 0.0

    tie_breaking: PFExpertTieBreaking = (
        "lowest_action_index"
    )

    def __post_init__(self) -> None:
        if self.pf_denominator_epsilon <= 0.0:
            raise ValueError(
                "pf_denominator_epsilon must be "
                "positive."
            )

        if self.pf_tie_epsilon < 0.0:
            raise ValueError(
                "pf_tie_epsilon cannot be negative."
            )

        if self.tie_breaking not in (
            "lowest_action_index",
            "no_allocation",
        ):
            raise ValueError(
                "tie_breaking must be "
                "'lowest_action_index' or "
                "'no_allocation'."
            )


@dataclass(frozen=True)
class PPOPFExpertActionData:
    """
    PF expert label for one 1LDS layer.

    expert_actions:
        [RBG]

        Actor action convention:

            0 ... K-1
                PF-TDS candidate index

            K
                NO ALLOCATION

    best_pf_sum:
        [RBG]

        PF objective associated with the chosen
        expert action.

    action_mask:
        [RBG, K+1]

        Exact legal action mask under which the
        expert label was generated.

    num_phy_evaluations:
        Number of candidate-set score requests made
        while generating this label.
    """

    expert_actions: torch.Tensor

    best_pf_sum: torch.Tensor

    action_mask: torch.Tensor

    num_phy_evaluations: int


def _build_candidate_set_for_action(
    *,
    previous_candidates: torch.Tensor,
    action: int,
    num_candidates: int,
) -> torch.Tensor:
    """
    Construct the scheduled candidate set produced
    by one hypothetical expert action.

    Actor convention:

        action 0 ... K-1
            append that candidate

        action K
            NO ALLOCATION

    Previous PPO-executed layers remain fixed.
    """

    if action == num_candidates:
        return previous_candidates

    candidate_tensor = torch.tensor(
        [
            action,
        ],
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


def _should_replace_on_tie(
    *,
    new_action: int,
    current_action: int,
    num_candidates: int,
    tie_breaking: PFExpertTieBreaking,
) -> bool:
    """
    Resolve PF-score ties deterministically.

    PAPER-UNSPECIFIED reproduction detail.
    """

    if tie_breaking == "lowest_action_index":
        return (
            new_action
            < current_action
        )

    if tie_breaking == "no_allocation":
        no_allocation_action = (
            num_candidates
        )

        return (
            new_action
            == no_allocation_action
            and current_action
            != no_allocation_action
        )

    raise RuntimeError(
        "Unexpected PF expert tie-breaking mode."
    )


def _validate_pf_expert_inputs(
    *,
    allocation: CellAllocation,
    user_slot_index: int,
    num_candidates: int,
    past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> None:
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
            "past_average_throughput cannot be "
            "negative."
        )

    if tuple(
        candidate_valid_mask.shape
    ) != (
        num_candidates,
    ):
        raise ValueError(
            "candidate_valid_mask must have shape "
            "[candidate]."
        )

    if candidate_valid_mask.dtype != torch.bool:
        raise ValueError(
            "candidate_valid_mask must use "
            "torch.bool."
        )

    device = (
        allocation
        .candidate_by_user_slot
        .device
    )

    if (
        past_average_throughput.device
        != device
    ):
        raise ValueError(
            "past_average_throughput is on the "
            "wrong device."
        )

    if candidate_valid_mask.device != device:
        raise ValueError(
            "candidate_valid_mask is on the wrong "
            "device."
        )

    #
    # The current layer and all FUTURE layers must
    # still be empty.
    #
    # Previous layers are the PPO actions already
    # executed in the environment.
    #
    current_and_future = (
        allocation
        .candidate_by_user_slot[
            user_slot_index:,
            :,
        ]
    )

    if torch.any(
        current_and_future
        != NO_ALLOCATION
    ):
        raise ValueError(
            "PF expert expects the current and "
            "future user slots to be unallocated."
        )


def _generate_ppo_pf_expert_actions_batched(
    *,
    allocation: CellAllocation,
    user_slot_index: int,
    num_candidates: int,
    past_average_throughput: torch.Tensor,
    action_mask: torch.Tensor,
    score_many_pf,
    config: PPOPFExpertConfig,
) -> PPOPFExpertActionData:
    """
    GPU-batched PF expert.

    Physics is identical to the scalar expert.

    Difference:

        scalar:
            RBG -> action -> PHY

        batched:
            construct all legal hypotheses
                    ->
            batched PHY
                    ->
            choose expert actions
    """

    device = (
        allocation
        .candidate_by_user_slot
        .device
    )

    dtype = (
        past_average_throughput.dtype
    )

    num_rbgs = (
        allocation.num_rbgs
    )

    #
    # One synchronization for the tiny scheduling
    # control matrices, replacing hundreds of .item()
    # synchronizations.
    #
    previous_allocation_cpu = (
        allocation
        .candidate_by_user_slot[
            :user_slot_index,
            :,
        ]
        .detach()
        .cpu()
    )

    action_mask_cpu = (
        action_mask
        .detach()
        .cpu()
    )

    requests: list[
        tuple[
            tuple[int, ...],
            int,
        ]
    ] = []

    request_rbg: list[int] = []

    request_action: list[int] = []

    for rbg_index in range(
        num_rbgs
    ):
        previous_candidates = tuple(
            int(value)
            for value
            in (
                previous_allocation_cpu[
                    :,
                    rbg_index,
                ]
                .tolist()
            )
            if int(value)
            != NO_ALLOCATION
        )

        for action in range(
            num_candidates + 1
        ):
            if not bool(
                action_mask_cpu[
                    rbg_index,
                    action,
                ]
            ):
                continue

            if action == num_candidates:
                hypothetical_candidates = (
                    previous_candidates
                )

            else:
                hypothetical_candidates = (
                    previous_candidates
                    + (
                        action,
                    )
                )

            requests.append(
                (
                    hypothetical_candidates,
                    rbg_index,
                )
            )

            request_rbg.append(
                rbg_index
            )

            request_action.append(
                action
            )

    with torch.no_grad():
        pf_values = score_many_pf(
            requests,
            past_average_throughput=(
                past_average_throughput
            ),
            denominator_epsilon=(
                config
                .pf_denominator_epsilon
            ),
        )

    if tuple(
        pf_values.shape
    ) != (
        len(
            requests
        ),
    ):
        raise ValueError(
            "Batched PF scorer returned an "
            "unexpected shape."
        )

    if not torch.all(
        torch.isfinite(
            pf_values
        )
    ):
        raise ValueError(
            "Expert PF score is non-finite."
        )

    pf_matrix = torch.full(
        (
            num_rbgs,
            num_candidates + 1,
        ),
        fill_value=float(
            "-inf"
        ),
        dtype=dtype,
        device=device,
    )

    request_rbg_tensor = torch.tensor(
        request_rbg,
        dtype=torch.long,
        device=device,
    )

    request_action_tensor = torch.tensor(
        request_action,
        dtype=torch.long,
        device=device,
    )

    pf_matrix[
        request_rbg_tensor,
        request_action_tensor,
    ] = pf_values

    #
    # Preserve the EXISTING sequential tie semantics
    # exactly.
    #
    # One matrix transfer per layer is acceptable and
    # dramatically cheaper than .item() per action.
    #
    pf_matrix_cpu = (
        pf_matrix
        .detach()
        .cpu()
    )

    expert_action_values: list[int] = []

    best_pf_values: list[float] = []

    for rbg_index in range(
        num_rbgs
    ):
        best_action: int | None = None

        best_pf_value: float | None = None

        for action in range(
            num_candidates + 1
        ):
            if not bool(
                action_mask_cpu[
                    rbg_index,
                    action,
                ]
            ):
                continue

            pf_value = float(
                pf_matrix_cpu[
                    rbg_index,
                    action,
                ]
            )

            if best_action is None:
                best_action = action
                best_pf_value = pf_value
                continue

            assert (
                best_pf_value
                is not None
            )

            improvement = (
                pf_value
                - best_pf_value
            )

            if (
                improvement
                > config.pf_tie_epsilon
            ):
                best_action = action
                best_pf_value = pf_value
                continue

            is_tie = (
                abs(
                    improvement
                )
                <= config.pf_tie_epsilon
            )

            if (
                is_tie
                and _should_replace_on_tie(
                    new_action=action,
                    current_action=(
                        best_action
                    ),
                    num_candidates=(
                        num_candidates
                    ),
                    tie_breaking=(
                        config.tie_breaking
                    ),
                )
            ):
                best_action = action
                best_pf_value = pf_value

        if (
            best_action is None
            or best_pf_value is None
        ):
            raise RuntimeError(
                "PF expert found no legal action "
                "for an RBG."
            )

        expert_action_values.append(
            best_action
        )

        best_pf_values.append(
            best_pf_value
        )

    expert_actions = torch.tensor(
        expert_action_values,
        dtype=torch.long,
        device=device,
    )

    best_pf_sum = torch.tensor(
        best_pf_values,
        dtype=dtype,
        device=device,
    )

    return PPOPFExpertActionData(
        expert_actions=expert_actions,
        best_pf_sum=best_pf_sum,
        action_mask=(
            action_mask
            .detach()
            .clone()
        ),
        num_phy_evaluations=(
            len(
                requests
            )
        ),
    )

def generate_ppo_pf_expert_actions(
    *,
    allocation: CellAllocation,
    user_slot_index: int,
    num_candidates: int,
    past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    score_rbg: RBGScoreFunction,
    config: PPOPFExpertConfig,
) -> PPOPFExpertActionData:
    """
    Generate the PF expert action vector for exactly
    one 1LDS layer.

    IMPORTANT:
        This function does NOT modify `allocation`.

        The expert label is supervision only.

        The PPO actor's action remains the action
        that will actually control the environment.
    """

    _validate_pf_expert_inputs(
        allocation=allocation,
        user_slot_index=user_slot_index,
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

    action_mask = (
        build_next_user_slot_action_mask(
            allocation=allocation,
            num_candidates=num_candidates,
            user_slot_index=user_slot_index,
            candidate_valid_mask=(
                candidate_valid_mask
            ),
        )
    )

    #
    # Optimized physical scorer exposes a batched PF
    # interface. Plain function callbacks used by unit
    # tests/classical scaffolding retain the original
    # scalar implementation below.
    #
    score_many_pf = getattr(
        score_rbg,
        "score_many_pf",
        None,
    )

    if callable(
        score_many_pf
    ):
        return (
            _generate_ppo_pf_expert_actions_batched(
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
                action_mask=(
                    action_mask
                ),
                score_many_pf=(
                    score_many_pf
                ),
                config=config,
            )
        )

    #
    # Actor convention:
    #
    #     num_candidates = NO ALLOCATION.
    #
    expert_actions = torch.full(
        (
            num_rbgs,
        ),
        fill_value=num_candidates,
        dtype=torch.long,
        device=device,
    )

    best_pf_sum = torch.zeros(
        num_rbgs,
        dtype=dtype,
        device=device,
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

        best_action: int | None = None

        best_pf_value: float | None = None

        best_pf_tensor: (
            torch.Tensor | None
        ) = None

        for action in range(
            num_candidates + 1
        ):
            is_legal = bool(
                action_mask[
                    rbg_index,
                    action,
                ].item()
            )

            if not is_legal:
                continue

            hypothetical_candidates = (
                _build_candidate_set_for_action(
                    previous_candidates=(
                        previous_candidates
                    ),
                    action=action,
                    num_candidates=(
                        num_candidates
                    ),
                )
            )

            #
            # Expert-label generation is not part of
            # actor backpropagation.
            #
            with torch.no_grad():
                score = score_rbg(
                    hypothetical_candidates,
                    rbg_index,
                )

                pf_sum = compute_pf_sum(
                    score=score,
                    selected_candidates=(
                        hypothetical_candidates
                    ),
                    past_average_throughput=(
                        past_average_throughput
                    ),
                    denominator_epsilon=(
                        config
                        .pf_denominator_epsilon
                    ),
                )

            num_phy_evaluations += 1

            if pf_sum.ndim != 0:
                raise ValueError(
                    "Expert PF score must be scalar."
                )

            if not bool(
                torch.isfinite(
                    pf_sum
                ).item()
            ):
                raise ValueError(
                    "Expert PF score is non-finite."
                )

            pf_value = float(
                pf_sum.item()
            )

            if best_action is None:
                best_action = action

                best_pf_value = pf_value

                best_pf_tensor = (
                    pf_sum
                    .detach()
                    .clone()
                )

                continue

            assert best_pf_value is not None
            assert best_pf_tensor is not None

            improvement = (
                pf_value
                - best_pf_value
            )

            if (
                improvement
                > config.pf_tie_epsilon
            ):
                best_action = action

                best_pf_value = pf_value

                best_pf_tensor = (
                    pf_sum
                    .detach()
                    .clone()
                )

                continue

            is_tie = (
                abs(
                    improvement
                )
                <= config.pf_tie_epsilon
            )

            if (
                is_tie
                and _should_replace_on_tie(
                    new_action=action,
                    current_action=(
                        best_action
                    ),
                    num_candidates=(
                        num_candidates
                    ),
                    tie_breaking=(
                        config.tie_breaking
                    ),
                )
            ):
                best_action = action

                best_pf_value = pf_value

                best_pf_tensor = (
                    pf_sum
                    .detach()
                    .clone()
                )

        if best_action is None:
            raise RuntimeError(
                "PF expert found no legal action "
                "for an RBG."
            )

        assert best_pf_tensor is not None

        expert_actions[
            rbg_index
        ] = best_action

        best_pf_sum[
            rbg_index
        ] = best_pf_tensor

    return PPOPFExpertActionData(
        expert_actions=(
            expert_actions
            .detach()
            .clone()
        ),
        best_pf_sum=(
            best_pf_sum
            .detach()
            .clone()
        ),
        action_mask=(
            action_mask
            .detach()
            .clone()
        ),
        num_phy_evaluations=(
            num_phy_evaluations
        ),
    )


