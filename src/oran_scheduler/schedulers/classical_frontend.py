from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.allocation import (
    CellAllocation,
    build_initial_allocation_from_fds,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    PFTimeDomainResult,
    run_pf_tds,
)
from oran_scheduler.schedulers.su_mimo_fds import (
    SUMIMOFDSConfig,
    SUMIMOFDSResult,
    run_su_mimo_fds,
)
from oran_scheduler.simulator.serving_layout import (
    gather_candidate_rbg_values,
    gather_candidate_ue_values,
)

@dataclass
class ClassicalFrontendData:
    """
    Output of the classical TDS -> initial-FDS front end.

    td_instantaneous_rate:
        [batch, cell, padded_UE]

    candidate_global_ue_indices:
        [batch, cell, candidate]

        Invalid padded candidate slots contain -1.

    candidate_rbg_rate:
        [batch, cell, candidate, RBG]

    candidate_past_average_throughput:
        [batch, cell, candidate]
    """

    td_instantaneous_rate: torch.Tensor

    tds_result: PFTimeDomainResult

    candidate_global_ue_indices: torch.Tensor

    candidate_rbg_rate: torch.Tensor

    candidate_past_average_throughput: torch.Tensor

    fds_result: SUMIMOFDSResult

def run_classical_frontend(
    ue_rbg_rate: torch.Tensor,
    global_ue_indices: torch.Tensor,
    valid_ue_mask: torch.Tensor,
    past_average_throughput: torch.Tensor,
    tds_config: PFTimeDomainConfig,
    fds_config: SUMIMOFDSConfig,
) -> ClassicalFrontendData:
    """
    Run:

        per-UE physical rates
            ->
        PF TDS
            ->
        candidate gathering
            ->
        open-reproduction PF FDS.

    ue_rbg_rate:
        [batch, cell, padded_UE, RBG]

    global_ue_indices:
        [batch, cell, padded_UE]

    valid_ue_mask:
        [batch, cell, padded_UE]

    past_average_throughput:
        [batch, cell, padded_UE]

    Important:
        The PF-based FDS is an open-reproduction surrogate.
        The paper does not publicly specify the exact initial
        SU-MIMO FDS algorithm.
    """

    if ue_rbg_rate.ndim != 4:
        raise ValueError(
            "ue_rbg_rate must have shape "
            "[batch, cell, UE, RBG]."
        )

    expected_ue_shape = (
        ue_rbg_rate.shape[
            :3
        ]
    )

    if tuple(
        global_ue_indices.shape
    ) != tuple(
        expected_ue_shape
    ):
        raise ValueError(
            "global_ue_indices dimensions "
            "do not match ue_rbg_rate."
        )

    if tuple(
        valid_ue_mask.shape
    ) != tuple(
        expected_ue_shape
    ):
        raise ValueError(
            "valid_ue_mask dimensions "
            "do not match ue_rbg_rate."
        )

    if tuple(
        past_average_throughput.shape
    ) != tuple(
        expected_ue_shape
    ):
        raise ValueError(
            "past_average_throughput dimensions "
            "do not match ue_rbg_rate."
        )

    if torch.any(
        ue_rbg_rate < 0
    ):
        raise ValueError(
            "UE achievable rates cannot be negative."
        )

    if torch.any(
        past_average_throughput < 0
    ):
        raise ValueError(
            "Past-average throughput cannot be negative."
        )

    valid_ue_mask = (
        valid_ue_mask.to(
            dtype=torch.bool
        )
    )

    td_instantaneous_rate = (
        ue_rbg_rate.sum(
            dim=-1
        )
    )

    tds_result = run_pf_tds(
        instantaneous_rate=(
            td_instantaneous_rate
        ),
        past_average_throughput=(
            past_average_throughput
        ),
        config=tds_config,
        valid_ue_mask=(
            valid_ue_mask
        ),
    )

    candidate_rbg_rate = (
        gather_candidate_rbg_values(
            ue_rbg_values=(
                ue_rbg_rate
            ),
            candidate_indices=(
                tds_result
                .candidate_indices
            ),
            candidate_valid_mask=(
                tds_result
                .candidate_valid_mask
            ),
        )
    )

    candidate_history = (
        gather_candidate_ue_values(
            ue_values=(
                past_average_throughput
            ),
            candidate_indices=(
                tds_result
                .candidate_indices
            ),
            candidate_valid_mask=(
                tds_result
                .candidate_valid_mask
            ),
        )
    )


    candidate_global_ue_indices = torch.gather(
        global_ue_indices,
        dim=2,
        index=(
            tds_result
            .candidate_indices
        ),
    )

    candidate_global_ue_indices = torch.where(
        tds_result.candidate_valid_mask,
        candidate_global_ue_indices,
        torch.full_like(
            candidate_global_ue_indices,
            fill_value=-1,
        ),
    )

    fds_result = run_su_mimo_fds(
        candidate_rate=(
            candidate_rbg_rate
        ),
        candidate_past_average_throughput=(
            candidate_history
        ),
        candidate_valid_mask=(
            tds_result
            .candidate_valid_mask
        ),
        config=fds_config,
    )

    return ClassicalFrontendData(
        td_instantaneous_rate=(
            td_instantaneous_rate
        ),
        tds_result=tds_result,
        candidate_global_ue_indices=(
            candidate_global_ue_indices
        ),
        candidate_rbg_rate=(
            candidate_rbg_rate
        ),
        candidate_past_average_throughput=(
            candidate_history
        ),
        fds_result=fds_result,
    )


def build_classical_initial_cell_allocation(
    frontend_data: ClassicalFrontendData,
    batch_index: int,
    cell_index: int,
    num_user_slots: int,
) -> CellAllocation:
    """
    Convert one cell's FDS output into the allocation
    consumed by Baseline/PF-Greedy SDS.
    """

    return build_initial_allocation_from_fds(
        selected_candidate_by_rbg=(
            frontend_data
            .fds_result
            .selected_candidate_indices[
                batch_index,
                cell_index,
                :,
            ]
        ),
        selected_rbg_valid_mask=(
            frontend_data
            .fds_result
            .selected_valid_mask[
                batch_index,
                cell_index,
                :,
            ]
        ),
        num_user_slots=num_user_slots,
    )

