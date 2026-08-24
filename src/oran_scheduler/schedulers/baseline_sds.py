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
class BaselineSDSConfig:
    """
    Configuration for the paper-style first-improvement SDS.

    improvement_epsilon_bps:
        Required throughput improvement before accepting
        an additional UE.

        Default 0 means strictly higher throughput.

        A positive value can later be used only as a
        numerical-stability/reproduction parameter.
    """

    improvement_epsilon_bps: float = 0.0


@dataclass
class BaselineSDSResult:
    """
    Final spatial allocation and its RBG throughputs.

    allocation:
        Completed user-slot x RBG allocation.

    rbg_rate_bps:
        [RBG]

    num_phy_evaluations:
        Number of hypothetical RBG user sets scored.
    """

    allocation: CellAllocation

    rbg_rate_bps: torch.Tensor

    num_phy_evaluations: int


RBGScoreFunction = Callable[
    [
        torch.Tensor,
        int,
    ],
    torch.Tensor,
]

def validate_baseline_sds_initial_allocation(
    allocation: CellAllocation,
    num_candidates: int,
) -> None:
    """
    Baseline SDS starts from the initial FDS allocation.

    For the current pipeline we expect:
        - slot 0 may contain one initial UE per RBG
        - every later user slot must initially be empty

    The exact algorithm producing slot 0 is publicly
    unspecified by the paper and belongs to the FDS stage.
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
            "Baseline SDS expects only user slot 0 "
            "to contain the initial FDS allocation."
        )


def run_baseline_sds(
    initial_allocation: CellAllocation,
    num_candidates: int,
    score_rbg: RBGScoreFunction,
    config: BaselineSDSConfig,
) -> BaselineSDSResult:
    """
    Run first-improvement spatial-domain scheduling.

    Candidate ordering is assumed to be the PF-TDS ordering:

        candidate 0
        candidate 1
        ...
        candidate K-1

    For each RBG and each additional user slot:

        1. Try candidates in PF order.
        2. Skip UEs already scheduled on that RBG.
        3. Physically re-evaluate the tentative user set.
        4. Accept the FIRST candidate that improves total
           RBG throughput.
        5. If no candidate improves throughput, stop adding
           users to that RBG.
    """


    validate_baseline_sds_initial_allocation(
        allocation=initial_allocation,
        num_candidates=num_candidates,
    )

    if config.improvement_epsilon_bps < 0.0:
        raise ValueError(
            "improvement_epsilon_bps cannot be negative."
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


        current_rate = score_rbg(
            current_candidates,
            rbg_index,
        )

        num_phy_evaluations += 1

        current_rate = torch.as_tensor(
            current_rate,
            dtype=torch.float32,
            device=device,
        )

        if current_rate.ndim != 0:
            raise ValueError(
                "score_rbg must return a scalar."
            )

        rbg_rate_bps[
            rbg_index
        ] = current_rate


        for user_slot_index in range(
            1,
            allocation.num_user_slots,
        ):

            accepted_candidate = False


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


                tentative_rate = score_rbg(
                    tentative_candidates,
                    rbg_index,
                )

                num_phy_evaluations += 1

                tentative_rate = torch.as_tensor(
                    tentative_rate,
                    dtype=torch.float32,
                    device=device,
                )

                if tentative_rate.ndim != 0:
                    raise ValueError(
                        "score_rbg must return a scalar."
                    )


                required_rate = (
                    current_rate
                    + config.improvement_epsilon_bps
                )

                if tentative_rate > required_rate:


                    allocation.candidate_by_user_slot[
                        user_slot_index,
                        rbg_index,
                    ] = candidate_index

                    current_candidates = (
                        tentative_candidates
                    )

                    current_rate = (
                        tentative_rate
                    )

                    rbg_rate_bps[
                        rbg_index
                    ] = current_rate

                    accepted_candidate = True

                    break


            if not accepted_candidate:
                break

    return BaselineSDSResult(
        allocation=allocation,
        rbg_rate_bps=rbg_rate_bps,
        num_phy_evaluations=(
            num_phy_evaluations
        ),
    )




