import torch

from oran_scheduler.schedulers.classical_traffic_eval import (
    gather_serving_values_to_candidates,
    scatter_candidate_values_to_serving,
)


def test_scatter_candidate_values_to_serving() -> None:

    output = (
        scatter_candidate_values_to_serving(
            candidate_values=(
                torch.tensor(
                    [
                        10.0,
                        20.0,
                        0.0,
                    ]
                )
            ),

            candidate_serving_indices=(
                torch.tensor(
                    [
                        2,
                        0,
                        0,
                    ],
                    dtype=torch.long,
                )
            ),

            candidate_valid_mask=(
                torch.tensor(
                    [
                        True,
                        True,
                        False,
                    ],
                    dtype=torch.bool,
                )
            ),

            num_serving_ues=4,
        )
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                20.0,
                0.0,
                10.0,
                0.0,
            ]
        ),
    )


def test_gather_serving_values_to_candidates() -> None:

    output = (
        gather_serving_values_to_candidates(
            serving_values=(
                torch.tensor(
                    [
                        20.0,
                        30.0,
                        10.0,
                        40.0,
                    ]
                )
            ),

            candidate_serving_indices=(
                torch.tensor(
                    [
                        2,
                        0,
                        1,
                    ],
                    dtype=torch.long,
                )
            ),

            candidate_valid_mask=(
                torch.tensor(
                    [
                        True,
                        True,
                        False,
                    ],
                    dtype=torch.bool,
                )
            ),
        )
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                10.0,
                20.0,
                0.0,
            ]
        ),
    )
