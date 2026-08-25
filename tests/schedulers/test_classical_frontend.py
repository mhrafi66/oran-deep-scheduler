import torch

from oran_scheduler.schedulers.classical_frontend import (
    build_classical_initial_cell_allocation,
    run_classical_frontend,
)
from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)
from oran_scheduler.schedulers.su_mimo_fds import (
    SUMIMOFDSConfig,
)


def test_pf_tds_to_fds_pipeline():
    device = "cuda:0"

    ue_rbg_rate = torch.tensor(
        [
            [
                [
                    [10.0, 1.0],
                    [4.0, 8.0],
                    [6.0, 6.0],
                ]
            ]
        ],
        device=device,
    )

    past_average = torch.tensor(
        [
            [
                [
                    2.0,
                    1.0,
                    2.0,
                ]
            ]
        ],
        device=device,
    )

    global_ue_indices = torch.tensor(
        [
            [
                [
                    100,
                    200,
                    300,
                ]
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    valid_ue_mask = torch.tensor(
        [
            [
                [
                    True,
                    True,
                    True,
                ]
            ]
        ],
        dtype=torch.bool,
        device=device,
    )

    result = run_classical_frontend(
        ue_rbg_rate=ue_rbg_rate,
        global_ue_indices=(
            global_ue_indices
        ),
        valid_ue_mask=valid_ue_mask,
        past_average_throughput=(
            past_average
        ),
        tds_config=PFTimeDomainConfig(
            num_candidates=4,
        ),
        fds_config=SUMIMOFDSConfig(),
    )

    expected_candidate_global_ues = torch.tensor(
        [
            [
                [
                    200,
                    300,
                    100,
                    -1,
                ]
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result.candidate_global_ue_indices,
        expected_candidate_global_ues,
    )

    expected_valid_mask = torch.tensor(
        [
            [
                [
                    True,
                    True,
                    True,
                    False,
                ]
            ]
        ],
        dtype=torch.bool,
        device=device,
    )

    torch.testing.assert_close(
        result
        .tds_result
        .candidate_valid_mask,
        expected_valid_mask,
    )

    expected_fds_candidates = torch.tensor(
        [
            [
                [
                    2,
                    0,
                ]
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result
        .fds_result
        .selected_candidate_indices,
        expected_fds_candidates,
    )

    allocation = (
        build_classical_initial_cell_allocation(
            frontend_data=result,
            batch_index=0,
            cell_index=0,
            num_user_slots=4,
        )
    )

    expected_allocation = torch.tensor(
        [
            [2, 0],
            [-1, -1],
            [-1, -1],
            [-1, -1],
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        allocation.candidate_by_user_slot,
        expected_allocation,
    )

