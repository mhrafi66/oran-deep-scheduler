from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.pf_tds import (
    update_past_average_throughput,
)


@dataclass(frozen=True)
class CellThroughputHistoryUpdate:
    """
    Result of advancing one cell's throughput history
    by one TTI.

    delivered_rate_bps:
        Realized throughput mapped back to the padded
        serving-UE layout.

        Shape:
            [serving_ue]

    updated_average_throughput_bps:
        Exponentially smoothed throughput history to
        use in the next TTI.

        Shape:
            [serving_ue]
    """

    delivered_rate_bps: torch.Tensor

    updated_average_throughput_bps: torch.Tensor

def map_candidate_rates_to_serving_ues(
    *,
    candidate_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    candidate_delivered_rate_bps: torch.Tensor,
    num_serving_ue_slots: int,
) -> torch.Tensor:
    """
    Map candidate-space delivered rates back into
    one cell's padded serving-UE layout.

    Args:
        candidate_indices:
            [candidate]

            Each value is an index into the cell's
            padded serving-UE dimension.

        candidate_valid_mask:
            [candidate]

            False for padded/invalid candidate slots.

        candidate_delivered_rate_bps:
            [candidate]

            Realized throughput for each TDS candidate
            across the current TTI.

        num_serving_ue_slots:
            Size of the cell's padded serving-UE
            dimension.

    Returns:
        [serving_ue]

        UEs without realized throughput receive zero.
    """

    if num_serving_ue_slots <= 0:
        raise ValueError(
            "num_serving_ue_slots must be positive."
        )

    if candidate_indices.ndim != 1:
        raise ValueError(
            "candidate_indices must have shape "
            "[candidate]."
        )

    expected_candidate_shape = (
        candidate_indices.shape[0],
    )

    if tuple(
        candidate_valid_mask.shape
    ) != expected_candidate_shape:
        raise ValueError(
            "candidate_valid_mask must have the same "
            "candidate dimension as candidate_indices."
        )

    if tuple(
        candidate_delivered_rate_bps.shape
    ) != expected_candidate_shape:
        raise ValueError(
            "candidate_delivered_rate_bps must have "
            "shape [candidate]."
        )


    if (
        candidate_valid_mask.device
        != candidate_indices.device
    ):
        raise ValueError(
            "candidate_valid_mask and candidate_indices "
            "must be on the same device."
        )

    if (
        candidate_delivered_rate_bps.device
        != candidate_indices.device
    ):
        raise ValueError(
            "candidate_delivered_rate_bps and "
            "candidate_indices must be on the same device."
        )

    if torch.any(
        candidate_delivered_rate_bps < 0.0
    ):
        raise ValueError(
            "Delivered throughput cannot be negative."
        )

    valid_candidate_indices = (
        candidate_indices[
            candidate_valid_mask
        ]
    )

    valid_candidate_rates = (
        candidate_delivered_rate_bps[
            candidate_valid_mask
        ]
    )

    if valid_candidate_indices.numel() > 0:
        if torch.any(
            valid_candidate_indices < 0
        ):
            raise ValueError(
                "Valid candidate indices cannot be negative."
            )

        if torch.any(
            valid_candidate_indices
            >= num_serving_ue_slots
        ):
            raise ValueError(
                "A valid candidate index exceeds the "
                "serving-UE dimension."
            )

        if torch.unique(
            valid_candidate_indices
        ).numel() != (
            valid_candidate_indices.numel()
        ):
            raise ValueError(
                "Valid candidate indices must be unique."
            )

    delivered_rate_bps = torch.zeros(
        (
            num_serving_ue_slots,
        ),
        dtype=(
            candidate_delivered_rate_bps.dtype
        ),
        device=candidate_indices.device,
    )

    if valid_candidate_indices.numel() > 0:
        delivered_rate_bps[
            valid_candidate_indices
        ] = valid_candidate_rates

    return delivered_rate_bps


def update_cell_throughput_history(
    *,
    previous_average_throughput_bps: torch.Tensor,
    serving_ue_valid_mask: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    candidate_delivered_rate_bps: torch.Tensor,
    forgetting_factor: float,
) -> CellThroughputHistoryUpdate:
    """
    Advance one cell's PF throughput history by one TTI.

    PAPER-SPECIFIED equation:

        R_new
            =
        (1 - epsilon) * delivered_rate
        +
        epsilon * R_old

    OPEN-REPRODUCTION PARAMETER:

        The public paper defines epsilon but does not
        provide the exact numerical value used in the
        simulator.

    UEs that receive no data in this TTI have
    delivered_rate = 0 and therefore their history
    decays according to epsilon.
    """

    if previous_average_throughput_bps.ndim != 1:
        raise ValueError(
            "previous_average_throughput_bps must "
            "have shape [serving_ue]."
        )

    if tuple(
        serving_ue_valid_mask.shape
    ) != tuple(
        previous_average_throughput_bps.shape
    ):
        raise ValueError(
            "serving_ue_valid_mask must match the "
            "serving-UE throughput-history shape."
        )

    if (
        serving_ue_valid_mask.device
        != previous_average_throughput_bps.device
    ):
        raise ValueError(
            "serving_ue_valid_mask and throughput "
            "history must be on the same device."
        )

    if torch.any(
        previous_average_throughput_bps < 0.0
    ):
        raise ValueError(
            "Previous average throughput cannot "
            "be negative."
        )

    num_serving_ue_slots = (
        previous_average_throughput_bps.shape[0]
    )

    delivered_rate_bps = (
        map_candidate_rates_to_serving_ues(
            candidate_indices=candidate_indices,
            candidate_valid_mask=(
                candidate_valid_mask
            ),
            candidate_delivered_rate_bps=(
                candidate_delivered_rate_bps
            ),
            num_serving_ue_slots=(
                num_serving_ue_slots
            ),
        )
    )

    updated_average = (
        update_past_average_throughput(
            previous_average=(
                previous_average_throughput_bps
            ),
            delivered_rate=(
                delivered_rate_bps
            ),
            forgetting_factor=(
                forgetting_factor
            ),
        )
    )

    # Padding is not a real UE and must never acquire
    # scheduler history.
    updated_average = torch.where(
        serving_ue_valid_mask,
        updated_average,
        torch.zeros_like(
            updated_average
        ),
    )

    delivered_rate_bps = torch.where(
        serving_ue_valid_mask,
        delivered_rate_bps,
        torch.zeros_like(
            delivered_rate_bps
        ),
    )

    return CellThroughputHistoryUpdate(
        delivered_rate_bps=(
            delivered_rate_bps
        ),
        updated_average_throughput_bps=(
            updated_average
        ),
    )


