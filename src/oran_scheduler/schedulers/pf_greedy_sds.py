from dataclasses import dataclass
from typing import Callable

import torch

from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    CellAllocation,
    selected_candidates_for_rbg,
    validate_cell_allocation,
)


@dataclass(frozen=True)
class PFGreedySDSConfig:
    """
    PF-Greedy spatial-domain scheduler configuration.

    improvement_epsilon_bps:
        Required increase in total RBG throughput before a
        tentative candidate can be considered.

    pf_denominator_epsilon:
        Prevent division by zero in the PF objective.
    """

    improvement_epsilon_bps: float = 0.0

    pf_denominator_epsilon: float = 1.0e-8


@dataclass
class PFGreedyRBGScore:
    """
    Physical score of one hypothetical RBG user set.

    total_rate_bps:
        Scalar total RBG throughput.

    selected_candidate_rate_bps:
        [scheduled_UE]

        Rates in exactly the same order as the candidate
        indices supplied to the scoring function.
    """

    total_rate_bps: torch.Tensor

    selected_candidate_rate_bps: torch.Tensor


RBGScoreFunction = Callable[
    [
        torch.Tensor,
        int,
    ],
    PFGreedyRBGScore,
]


@dataclass
class PFGreedySDSResult:
    """
    Final PF-Greedy spatial allocation.

    allocation:
        Completed [user_slot, RBG] allocation.

    rbg_rate_bps:
        [RBG]

    rbg_pf_sum:
        [RBG]

        PF objective of the final selected UE set.

    num_phy_evaluations:
        Number of hypothetical MU-MIMO sets physically scored.
    """

    allocation: CellAllocation

    rbg_rate_bps: torch.Tensor

    rbg_pf_sum: torch.Tensor

    num_phy_evaluations: int


def validate_pf_greedy_initial_allocation(
    allocation: CellAllocation,
    num_candidates: int,
) -> None:
    """
    PF-Greedy SDS starts from an initial FDS allocation.

    Current pipeline assumption:
        user slot 0 may be populated,
        all later slots start empty.
    """

    validate_cell_allocation(
        allocation=allocation,
        num_candidates=num_candidates,
    )

    if allocation.num_user_slots < 1:
        raise ValueError(
            "At least one user slot is required."
        )

    if allocation.num_user_slots == 1:
        return

    later_slots = (
        allocation
        .candidate_by_user_slot[
            1:,
            :,
        ]
    )

    if torch.any(
        later_slots != NO_ALLOCATION
    ):
        raise ValueError(
            "PF-Greedy SDS expects only user slot 0 "
            "to contain the initial FDS allocation."
        )

def compute_pf_sum(
    score: PFGreedyRBGScore,
    selected_candidates: torch.Tensor,
    past_average_throughput: torch.Tensor,
    denominator_epsilon: float,
) -> torch.Tensor:
    """
    Compute the PF sum of one tentative spatial allocation.

        sum_u R_u / Rbar_u
    """

    if score.total_rate_bps.ndim != 0:
        raise ValueError(
            "total_rate_bps must be a scalar."
        )

    if tuple(
        score.selected_candidate_rate_bps.shape
    ) != (
        selected_candidates.numel(),
    ):
        raise ValueError(
            "selected_candidate_rate_bps must contain "
            "one rate per selected candidate."
        )

    denominators = (
        past_average_throughput[
            selected_candidates
        ]
    )

    denominators = torch.clamp(
        denominators,
        min=denominator_epsilon,
    )

    pf_values = (
        score.selected_candidate_rate_bps
        / denominators
    )

    return pf_values.sum()


def run_pf_greedy_sds(
    initial_allocation: CellAllocation,
    num_candidates: int,
    past_average_throughput: torch.Tensor,
    score_rbg: RBGScoreFunction,
    config: PFGreedySDSConfig,
) -> PFGreedySDSResult:
    """
    Run PF-Greedy spatial-domain scheduling.

    For every RBG and additional user slot:

        1. Consider every candidate not already scheduled.
        2. Recompute physical MU-MIMO rates for each tentative set.
        3. Reject sets that do not improve total RBG throughput.
        4. Among remaining sets, choose the one with maximum
           PF sum.
        5. Repeat for the next spatial user slot.

    Candidate ordering only resolves exact ties.
    """


    validate_pf_greedy_initial_allocation(
        allocation=initial_allocation,
        num_candidates=num_candidates,
    )

    if tuple(
        past_average_throughput.shape
    ) != (
        num_candidates,
    ):
        raise ValueError(
            "past_average_throughput must have shape "
            "[candidate]."
        )

    if torch.any(
        past_average_throughput < 0
    ):
        raise ValueError(
            "Past average throughput cannot be negative."
        )

    if config.improvement_epsilon_bps < 0.0:
        raise ValueError(
            "improvement_epsilon_bps cannot be negative."
        )

    if config.pf_denominator_epsilon <= 0.0:
        raise ValueError(
            "pf_denominator_epsilon must be positive."
        )


    actions = (
        initial_allocation
        .candidate_by_user_slot
        .clone()
    )

    allocation = CellAllocation(
        candidate_by_user_slot=actions
    )

    device = actions.device


    rbg_rate_bps = torch.zeros(
        allocation.num_rbgs,
        dtype=torch.float32,
        device=device,
    )

    rbg_pf_sum = torch.zeros(
        allocation.num_rbgs,
        dtype=torch.float32,
        device=device,
    )

    num_phy_evaluations = 0

    for rbg_index in range(
        allocation.num_rbgs
    ):

        current_candidates = (
            selected_candidates_for_rbg(
                allocation=allocation,
                rbg_index=rbg_index,
            )
        )

        if current_candidates.numel() == 0:
            continue


        current_score = score_rbg(
            current_candidates,
            rbg_index,
        )

        num_phy_evaluations += 1

        current_rate = torch.as_tensor(
            current_score.total_rate_bps,
            dtype=torch.float32,
            device=device,
        )

        if current_rate.ndim != 0:
            raise ValueError(
                "score_rbg total rate must be scalar."
            )

        current_pf_sum = compute_pf_sum(
            score=current_score,
            selected_candidates=(
                current_candidates
            ),
            past_average_throughput=(
                past_average_throughput
            ),
            denominator_epsilon=(
                config
                .pf_denominator_epsilon
            ),
        )

        rbg_rate_bps[
            rbg_index
        ] = current_rate

        rbg_pf_sum[
            rbg_index
        ] = current_pf_sum


        for user_slot_index in range(
            1,
            allocation.num_user_slots,
        ):

            best_candidate_index = None

            best_candidates = None

            best_score = None

            best_pf_sum = None

            for candidate_index in range(
                num_candidates
            ):

                already_scheduled = torch.any(
                    current_candidates
                    == candidate_index
                )

                if bool(
                    already_scheduled.item()
                ):
                    continue


                candidate_tensor = torch.tensor(
                    [candidate_index],
                    dtype=torch.long,
                    device=device,
                )

                tentative_candidates = torch.cat(
                    (
                        current_candidates,
                        candidate_tensor,
                    ),
                    dim=0,
                )


                tentative_score = score_rbg(
                    tentative_candidates,
                    rbg_index,
                )

                num_phy_evaluations += 1

                tentative_rate = torch.as_tensor(
                    tentative_score.total_rate_bps,
                    dtype=torch.float32,
                    device=device,
                )

                if tentative_rate.ndim != 0:
                    raise ValueError(
                        "score_rbg total rate must be scalar."
                    )


                required_rate = (
                    current_rate
                    + config.improvement_epsilon_bps
                )

                if tentative_rate <= required_rate:
                    continue

                tentative_pf_sum = (
                    compute_pf_sum(
                        score=tentative_score,
                        selected_candidates=(
                            tentative_candidates
                        ),
                        past_average_throughput=(
                            past_average_throughput
                        ),
                        denominator_epsilon=(
                            config
                            .pf_denominator_epsilon
                        ),
                    )
                )


                if (
                    best_pf_sum is None
                    or tentative_pf_sum
                    > best_pf_sum
                ):
                    best_candidate_index = (
                        candidate_index
                    )

                    best_candidates = (
                        tentative_candidates
                    )

                    best_score = (
                        tentative_score
                    )

                    best_pf_sum = (
                        tentative_pf_sum
                    )


            if best_candidate_index is None:
                break


            allocation.candidate_by_user_slot[
                user_slot_index,
                rbg_index,
            ] = best_candidate_index

            current_candidates = (
                best_candidates
            )

            current_score = (
                best_score
            )

            current_rate = torch.as_tensor(
                current_score.total_rate_bps,
                dtype=torch.float32,
                device=device,
            )

            current_pf_sum = (
                best_pf_sum
            )

            rbg_rate_bps[
                rbg_index
            ] = current_rate

            rbg_pf_sum[
                rbg_index
            ] = current_pf_sum


    return PFGreedySDSResult(
        allocation=allocation,
        rbg_rate_bps=rbg_rate_bps,
        rbg_pf_sum=rbg_pf_sum,
        num_phy_evaluations=(
            num_phy_evaluations
        ),
    )


