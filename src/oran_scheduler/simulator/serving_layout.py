from dataclasses import dataclass

import torch

from oran_scheduler.simulator.cell_association import (
    CellAssociationData,
)
from oran_scheduler.simulator.rbg import (
    RBGChannelData,
)

@dataclass
class ServingCellData:
    """
    Scheduler-facing UE pools after explicit cell association.

    power:
        [batch, cell, padded_UE, RBG]

        RBG power from each associated UE's serving BS.

    global_ue_indices:
        [batch, cell, padded_UE]

        Global UE index for every real UE slot.
        Padding slots contain -1.

    valid_ue_mask:
        [batch, cell, padded_UE]

        True for a real associated UE.
        False for padding.

    num_ues_per_cell:
        [batch, cell]

        Number of real UEs associated with each cell.
    """

    power: torch.Tensor

    global_ue_indices: torch.Tensor

    valid_ue_mask: torch.Tensor

    num_ues_per_cell: torch.Tensor

def validate_serving_layout_inputs(
    rbg_data: RBGChannelData,
    association: CellAssociationData,
) -> None:
    """
    Validate compatibility between the global RBG tensor
    and the cell-association result.
    """

    if rbg_data.power.ndim != 4:
        raise ValueError(
            "Expected RBG power shape "
            "[batch, global_UE, BS, RBG], "
            f"got {tuple(rbg_data.power.shape)}."
        )

    if association.serving_bs.ndim != 2:
        raise ValueError(
            "Expected serving_bs shape [batch, global_UE], "
            f"got {tuple(association.serving_bs.shape)}."
        )

    batch_size, num_ues, num_bs, _ = (
        rbg_data.power.shape
    )

    if tuple(association.serving_bs.shape) != (
        batch_size,
        num_ues,
    ):
        raise ValueError(
            "serving_bs dimensions do not match "
            "the RBG tensor."
        )

    if tuple(association.association_mask.shape) != (
        batch_size,
        num_ues,
        num_bs,
    ):
        raise ValueError(
            "association_mask dimensions do not match "
            "the RBG tensor."
        )

    if tuple(association.num_ues_per_cell.shape) != (
        batch_size,
        num_bs,
    ):
        raise ValueError(
            "num_ues_per_cell dimensions do not match "
            "the RBG tensor."
        )

    if association.serving_bs.device != rbg_data.power.device:
        raise ValueError(
            "Cell association and RBG tensors must be "
            "on the same device."
        )

def build_serving_cell_data(
    rbg_data: RBGChannelData,
    association: CellAssociationData,
) -> ServingCellData:
    """
    Build padded per-cell scheduler tensors.

    Input:

        RBG power:
            [batch, global_UE, BS, RBG]

        serving_bs:
            [batch, global_UE]

    Output:

        serving power:
            [batch, cell, padded_UE, RBG]

        global UE indices:
            [batch, cell, padded_UE]

        valid UE mask:
            [batch, cell, padded_UE]
    """

    validate_serving_layout_inputs(
        rbg_data=rbg_data,
        association=association,
    )

    (
        batch_size,
        num_ues,
        num_cells,
        num_rbgs,
    ) = rbg_data.power.shape

    max_ues_per_cell = int(
        association.num_ues_per_cell.max().item()
    )

    serving_power = torch.zeros(
        batch_size,
        num_cells,
        max_ues_per_cell,
        num_rbgs,
        dtype=rbg_data.power.dtype,
        device=rbg_data.power.device,
    )

    global_ue_indices = torch.full(
        (
            batch_size,
            num_cells,
            max_ues_per_cell,
        ),
        fill_value=-1,
        dtype=torch.long,
        device=rbg_data.power.device,
    )

    valid_ue_mask = torch.zeros(
        batch_size,
        num_cells,
        max_ues_per_cell,
        dtype=torch.bool,
        device=rbg_data.power.device,
    )

    for batch_index in range(batch_size):

        for cell_index in range(num_cells):

            cell_ue_indices = torch.nonzero(
                association.serving_bs[
                    batch_index
                ] == cell_index,
                as_tuple=False,
            ).squeeze(-1)

            num_cell_ues = cell_ue_indices.numel()

            if num_cell_ues == 0:
                continue

            global_ue_indices[
                batch_index,
                cell_index,
                :num_cell_ues,
            ] = cell_ue_indices

            valid_ue_mask[
                batch_index,
                cell_index,
                :num_cell_ues,
            ] = True

            serving_power[
                batch_index,
                cell_index,
                :num_cell_ues,
                :,
            ] = rbg_data.power[
                batch_index,
                cell_ue_indices,
                cell_index,
                :,
            ]

    return ServingCellData(
        power=serving_power,
        global_ue_indices=global_ue_indices,
        valid_ue_mask=valid_ue_mask,
        num_ues_per_cell=(
            association.num_ues_per_cell.clone()
        ),
    )

def gather_candidate_rbg_values(
    ue_rbg_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Gather arbitrary per-UE/per-RBG quantities for PF candidates.

    Args:
        ue_rbg_values:
            [batch, cell, padded_UE, RBG]

        candidate_indices:
            [batch, cell, candidate]

        candidate_valid_mask:
            [batch, cell, candidate]

    Returns:
        Candidate values with shape:

            [batch, cell, candidate, RBG]

    The input may represent channel power, SINR, achievable rate,
    or any other per-UE/per-RBG quantity.
    """

    if ue_rbg_values.ndim != 4:
        raise ValueError(
            "ue_rbg_values must have shape "
            "[batch, cell, UE, RBG]."
        )

    if candidate_indices.ndim != 3:
        raise ValueError(
            "candidate_indices must have shape "
            "[batch, cell, candidate]."
        )

    if candidate_valid_mask.shape != candidate_indices.shape:
        raise ValueError(
            "candidate_valid_mask must have the same "
            "shape as candidate_indices."
        )

    if (
        tuple(ue_rbg_values.shape[:2])
        != tuple(candidate_indices.shape[:2])
    ):
        raise ValueError(
            "Batch and cell dimensions must match."
        )

    num_rbgs = ue_rbg_values.shape[-1]

    gather_indices = (
        candidate_indices
        .unsqueeze(-1)
        .expand(
            -1,
            -1,
            -1,
            num_rbgs,
        )
    )

    candidate_values = torch.gather(
        ue_rbg_values,
        dim=2,
        index=gather_indices,
    )

    return torch.where(
        candidate_valid_mask.unsqueeze(-1),
        candidate_values,
        torch.zeros_like(candidate_values),
    )

def gather_candidate_global_ue_indices(
    serving_data: ServingCellData,
    candidate_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Translate local padded candidate slots into global UE IDs.

    Invalid candidate slots are returned as -1.
    """

    candidate_global_indices = torch.gather(
        serving_data.global_ue_indices,
        dim=2,
        index=candidate_indices,
    )

    candidate_global_indices = torch.where(
        candidate_valid_mask,
        candidate_global_indices,
        torch.full_like(
            candidate_global_indices,
            fill_value=-1,
        ),
    )

    return candidate_global_indices

def gather_candidate_ue_values(
    ue_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Gather per-UE scalar values for candidate slots.

    ue_values:
        [batch, cell, padded_UE]

    Returns:
        [batch, cell, candidate]
    """

    if ue_values.ndim != 3:
        raise ValueError(
            "ue_values must have shape "
            "[batch, cell, padded_UE]."
        )

    candidate_values = torch.gather(
        ue_values,
        dim=2,
        index=candidate_indices,
    )

    return torch.where(
        candidate_valid_mask,
        candidate_values,
        torch.zeros_like(candidate_values),
    )

def gather_serving_ue_rbg_values(
    global_ue_rbg_values: torch.Tensor,
    serving_data: ServingCellData,
) -> torch.Tensor:
    """
    Convert a global per-UE/per-RBG tensor into the padded
    serving-cell scheduler layout.

    Args:
        global_ue_rbg_values:
            [batch, global_UE, RBG]

        serving_data:
            Existing serving-cell layout containing the mapping
            from padded local UE slots to global UE indices.

    Returns:
        [batch, cell, padded_UE, RBG]

    Invalid padded UE slots are filled with zero.

    The input may represent achievable rate, SINR, CQI-derived
    values, or any other per-UE/per-RBG quantity.
    """
    if global_ue_rbg_values.ndim != 3:
        raise ValueError(
            "global_ue_rbg_values must have shape "
            "[batch, global_UE, RBG]."
        )

    if serving_data.global_ue_indices.ndim != 3:
        raise ValueError(
            "serving_data.global_ue_indices must have shape "
            "[batch, cell, padded_UE]."
        )

    if (
        global_ue_rbg_values.shape[0]
        != serving_data.global_ue_indices.shape[0]
    ):
        raise ValueError(
            "Batch dimensions must match."
        )

    if (
        global_ue_rbg_values.device
        != serving_data.global_ue_indices.device
    ):
        raise ValueError(
            "global_ue_rbg_values and serving_data "
            "must be on the same device."
        )

    safe_global_indices = torch.where(
        serving_data.valid_ue_mask,
        serving_data.global_ue_indices,
        torch.zeros_like(
            serving_data.global_ue_indices
        ),
    )

    valid_global_indices = (
        serving_data.global_ue_indices[
            serving_data.valid_ue_mask
        ]
    )

    if valid_global_indices.numel() > 0:
        if torch.any(valid_global_indices < 0):
            raise ValueError(
                "Valid serving slots contain negative "
                "global UE indices."
            )

        if torch.any(
            valid_global_indices
            >= global_ue_rbg_values.shape[1]
        ):
            raise ValueError(
                "Serving layout contains a global UE index "
                "outside global_ue_rbg_values."
            )

    num_rbgs = (
        global_ue_rbg_values.shape[-1]
    )

    gather_indices = (
        safe_global_indices
        .unsqueeze(-1)
        .expand(
            -1,
            -1,
            -1,
            num_rbgs,
        )
    )

    num_cells = (
        serving_data.global_ue_indices.shape[1]
    )

    expanded_global_values = (
        global_ue_rbg_values
        .unsqueeze(1)
        .expand(
            -1,
            num_cells,
            -1,
            -1,
        )
    )

    serving_values = torch.gather(
        expanded_global_values,
        dim=2,
        index=gather_indices,
    )

    return torch.where(
        serving_data.valid_ue_mask.unsqueeze(-1),
        serving_values,
        torch.zeros_like(
            serving_values
        ),
    )


