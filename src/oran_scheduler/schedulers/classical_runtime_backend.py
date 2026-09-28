from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from oran_scheduler.rl.ppo_greedy_search import (
    PPOGreedySearchConfig,
)
from oran_scheduler.rl.ppo_physical_score import (
    CachedPPOPhysicalRBGScorer,
    PPOPhysicalScoreInputs,
)
from oran_scheduler.rl.ppo_physical_tti import (
    PPOPhysicalTTIOutcome,
    evaluate_ppo_physical_tti,
)
from oran_scheduler.schedulers.allocation import (
    NO_ALLOCATION,
    CellAllocation,
    build_initial_allocation_from_fds,
)
from oran_scheduler.schedulers.baseline_sds import (
    BaselineSDSConfig,
    run_baseline_sds,
)
from oran_scheduler.schedulers.pf_greedy_sds import (
    PFGreedySDSConfig,
    run_pf_greedy_sds,
)
from oran_scheduler.schedulers.su_mimo_fds import (
    SUMIMOFDSConfig,
    run_su_mimo_fds,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    PreparedOneLDSCellTTI,
)


ClassicalSchedulerMode = Literal[
    "baseline",
    "pf_greedy",
]


@dataclass(frozen=True)
class ClassicalRuntimeScheduleResult:
    """
    Classical scheduler result using the exact same
    PF-TDS candidate representation and physical PHY
    infrastructure used by the PPO pipeline.

    Important reproduction note
    ---------------------------
    The public paper does not uniquely specify the
    proprietary initial SU-MIMO FDS implementation.

    For this comparison harness we therefore construct
    first-layer FDS from PHYSICALLY evaluated single-UE
    RBG rates followed by the repository's PF-FDS rule.

    This is an OPEN-REPRODUCTION comparison control.
    """

    mode: ClassicalSchedulerMode

    initial_allocation: CellAllocation

    allocation: CellAllocation

    single_user_rate_bps: torch.Tensor

    physical_outcome: PPOPhysicalTTIOutcome

    scheduler_num_phy_requests: int

    scheduler_num_unique_phy_evaluations: int


def allocation_to_ppo_actions(
    *,
    allocation: CellAllocation,
    num_candidates: int,
) -> torch.Tensor:
    """
    Convert CellAllocation representation:

        NO_ALLOCATION = -1

    into the PPO action convention:

        NO_ALLOCATION = num_candidates.

    Output:
        [user_slot, RBG]
    """

    if num_candidates <= 0:
        raise ValueError(
            "num_candidates must be positive."
        )

    actions = (
        allocation
        .candidate_by_user_slot
        .clone()
    )

    invalid_negative = (
        actions < NO_ALLOCATION
    )

    if torch.any(
        invalid_negative
    ):
        raise ValueError(
            "Allocation contains an invalid "
            "negative candidate index."
        )

    selected = (
        actions
        != NO_ALLOCATION
    )

    if (
        torch.any(
            actions[
                selected
            ]
            >= num_candidates
        )
    ):
        raise ValueError(
            "Allocation contains a candidate index "
            "outside the candidate set."
        )

    return torch.where(
        selected,
        actions,
        torch.full_like(
            actions,
            fill_value=num_candidates,
        ),
    )


def build_single_user_rate_matrix(
    *,
    physical_scorer: CachedPPOPhysicalRBGScorer,
    candidate_valid_mask: torch.Tensor,
    num_rbgs: int,
) -> torch.Tensor:
    """
    Physically evaluate every single-candidate/RBG pair.

    Output:
        [candidate, RBG]

    Invalid candidate slots remain zero.
    """

    if candidate_valid_mask.ndim != 1:
        raise ValueError(
            "candidate_valid_mask must have "
            "shape [candidate]."
        )

    if candidate_valid_mask.dtype != torch.bool:
        raise ValueError(
            "candidate_valid_mask must use bool."
        )

    if num_rbgs <= 0:
        raise ValueError(
            "num_rbgs must be positive."
        )

    num_candidates = int(
        candidate_valid_mask.numel()
    )

    device = (
        candidate_valid_mask.device
    )

    rate = torch.zeros(
        (
            num_candidates,
            num_rbgs,
        ),
        dtype=torch.float32,
        device=device,
    )

    for candidate_index in range(
        num_candidates
    ):

        if not bool(
            candidate_valid_mask[
                candidate_index
            ].item()
        ):
            continue

        candidate = torch.tensor(
            [
                candidate_index,
            ],
            dtype=torch.long,
            device=device,
        )

        for rbg_index in range(
            num_rbgs
        ):

            score = physical_scorer(
                candidate,
                rbg_index,
            )

            total_rate = torch.as_tensor(
                score.total_rate_bps,
                dtype=torch.float32,
                device=device,
            )

            if total_rate.ndim != 0:
                raise RuntimeError(
                    "Physical scorer returned "
                    "non-scalar total rate."
                )

            rate[
                candidate_index,
                rbg_index,
            ] = total_rate

    return rate


def build_physical_pf_initial_allocation(
    *,
    single_user_rate_bps: torch.Tensor,
    candidate_past_average_throughput: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    num_user_slots: int,
    fds_config: SUMIMOFDSConfig | None = None,
) -> CellAllocation:
    """
    Build the open-reproduction first-layer FDS using
    physically evaluated single-UE rates.

    Inputs:
        single_user_rate_bps:
            [candidate, RBG]

        candidate_past_average_throughput:
            [candidate]

        candidate_valid_mask:
            [candidate]
    """

    if single_user_rate_bps.ndim != 2:
        raise ValueError(
            "single_user_rate_bps must have "
            "shape [candidate, RBG]."
        )

    num_candidates = int(
        single_user_rate_bps.shape[
            0
        ]
    )

    if tuple(
        candidate_past_average_throughput.shape
    ) != (
        num_candidates,
    ):
        raise ValueError(
            "Candidate throughput history has "
            "the wrong shape."
        )

    if tuple(
        candidate_valid_mask.shape
    ) != (
        num_candidates,
    ):
        raise ValueError(
            "Candidate valid mask has "
            "the wrong shape."
        )

    if num_user_slots <= 0:
        raise ValueError(
            "num_user_slots must be positive."
        )

    fds_result = run_su_mimo_fds(
        candidate_rate=(
            single_user_rate_bps[
                None,
                None,
                :,
                :,
            ]
        ),

        candidate_past_average_throughput=(
            candidate_past_average_throughput[
                None,
                None,
                :,
            ]
        ),

        candidate_valid_mask=(
            candidate_valid_mask[
                None,
                None,
                :,
            ]
        ),

        config=(
            SUMIMOFDSConfig()
            if fds_config is None
            else fds_config
        ),
    )

    return build_initial_allocation_from_fds(
        selected_candidate_by_rbg=(
            fds_result
            .selected_candidate_indices[
                0,
                0,
                :,
            ]
        ),

        selected_rbg_valid_mask=(
            fds_result
            .selected_valid_mask[
                0,
                0,
                :,
            ]
        ),

        num_user_slots=num_user_slots,
    )


def run_classical_runtime_scheduler(
    *,
    mode: ClassicalSchedulerMode,
    prepared: PreparedOneLDSCellTTI,
    physical_inputs: PPOPhysicalScoreInputs,
    num_user_slots: int,
    baseline_config: BaselineSDSConfig | None = None,
    pf_greedy_config: PFGreedySDSConfig | None = None,
    fds_config: SUMIMOFDSConfig | None = None,
) -> ClassicalRuntimeScheduleResult:
    """
    Run Baseline SDS or PF-Greedy SDS against the same
    PF-TDS candidate set and common PHY used by PPO.

    Pipeline:

        PF-TDS candidate set
            ->
        physical single-user RBG evaluation
            ->
        open-reproduction PF FDS
            ->
        Baseline / PF-Greedy SDS
            ->
        common final allocation PHY evaluation

    This function does NOT:
        * update traffic queues;
        * update PF history;
        * advance a TTI;
        * perform PPO learning.
    """

    if mode not in (
        "baseline",
        "pf_greedy",
    ):
        raise ValueError(
            f"Unsupported classical scheduler: {mode}."
        )

    if num_user_slots <= 0:
        raise ValueError(
            "num_user_slots must be positive."
        )

    candidate_valid_mask = (
        prepared
        .candidate_valid_mask
    )

    candidate_history = (
        prepared
        .decision_inputs
        .past_average_throughput
    )

    num_candidates = int(
        candidate_valid_mask
        .numel()
    )

    num_rbgs = int(
        prepared
        .decision_inputs
        .subband_cqi
        .shape[
            1
        ]
    )

    if tuple(
        candidate_history.shape
    ) != (
        num_candidates,
    ):
        raise ValueError(
            "Prepared candidate history has "
            "unexpected shape."
        )

    if (
        not torch.equal(
            physical_inputs
            .candidate_global_ue_indices,
            prepared
            .candidate_global_ue_indices,
        )
    ):
        raise ValueError(
            "Physical inputs do not match "
            "the PF-TDS candidate ordering."
        )

    physical_scorer = (
        CachedPPOPhysicalRBGScorer(
            physical_inputs
        )
    )

    single_user_rate = (
        build_single_user_rate_matrix(
            physical_scorer=(
                physical_scorer
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
            num_rbgs=num_rbgs,
        )
    )

    initial_allocation = (
        build_physical_pf_initial_allocation(
            single_user_rate_bps=(
                single_user_rate
            ),

            candidate_past_average_throughput=(
                candidate_history
            ),

            candidate_valid_mask=(
                candidate_valid_mask
            ),

            num_user_slots=(
                num_user_slots
            ),

            fds_config=fds_config,
        )
    )

    if mode == "baseline":

        def baseline_score_rbg(
            selected_candidates: torch.Tensor,
            rbg_index: int,
        ) -> torch.Tensor:

            return (
                physical_scorer(
                    selected_candidates,
                    rbg_index,
                )
                .total_rate_bps
            )

        scheduler_result = (
            run_baseline_sds(
                initial_allocation=(
                    initial_allocation
                ),

                num_candidates=(
                    num_candidates
                ),

                score_rbg=(
                    baseline_score_rbg
                ),

                config=(
                    BaselineSDSConfig()
                    if baseline_config is None
                    else baseline_config
                ),

                candidate_valid_mask=(
                    candidate_valid_mask
                ),
            )
        )

    else:

        scheduler_result = (
            run_pf_greedy_sds(
                initial_allocation=(
                    initial_allocation
                ),

                num_candidates=(
                    num_candidates
                ),

                past_average_throughput=(
                    candidate_history
                ),

                score_rbg=(
                    physical_scorer
                ),

                config=(
                    PFGreedySDSConfig()
                    if pf_greedy_config is None
                    else pf_greedy_config
                ),

                candidate_valid_mask=(
                    candidate_valid_mask
                ),
            )
        )

    final_allocation = (
        scheduler_result
        .allocation
    )

    actions = allocation_to_ppo_actions(
        allocation=(
            final_allocation
        ),
        num_candidates=(
            num_candidates
        ),
    )

    #
    # Reuse the same final physical evaluator as PPO.
    #
    # The counterfactual PPO reward judge is disabled;
    # classical evaluation only needs the realized PHY.
    #
    physical_outcome = (
        evaluate_ppo_physical_tti(
            allocation=(
                final_allocation
            ),

            actions=actions,

            past_average_throughput=(
                candidate_history
            ),

            candidate_valid_mask=(
                candidate_valid_mask
            ),

            physical_inputs=(
                physical_inputs
            ),

            greedy_config=(
                PPOGreedySearchConfig()
            ),

            physical_scorer=None,

            run_counterfactual_greedy=False,
        )
    )

    return ClassicalRuntimeScheduleResult(
        mode=mode,

        initial_allocation=(
            initial_allocation
        ),

        allocation=(
            final_allocation
        ),

        single_user_rate_bps=(
            single_user_rate
        ),

        physical_outcome=(
            physical_outcome
        ),

        scheduler_num_phy_requests=(
            physical_scorer
            .num_score_requests
        ),

        scheduler_num_unique_phy_evaluations=(
            physical_scorer
            .num_unique_phy_evaluations
        ),
    )
