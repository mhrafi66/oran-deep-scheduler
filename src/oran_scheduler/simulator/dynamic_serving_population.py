from __future__ import annotations

import torch

from oran_scheduler.simulator.global_ue_state import (
    GlobalUESchedulerStateRegistry,
)


class RegistrySelectedCellMembershipProvider:
    """
    Translate the current GLOBAL association registry
    into the UE population of each selected scheduler
    stream.

    Persistent identity lives in:

        GlobalUESchedulerStateRegistry

    This adapter answers:

        "Which global UEs belong to selected
         scheduler stream s right now?"

    The selected physical cell IDs themselves remain
    fixed. The UE membership of those cells may
    change over time through handover.

    OPEN-REPRODUCTION:
        Infrastructure for dynamic-association
        experiments. Not specified by the Nokia
        scheduler paper.
    """

    def __init__(
        self,
        *,
        registry: GlobalUESchedulerStateRegistry,
        selected_cell_indices: tuple[
            int,
            ...,
        ],
    ) -> None:

        if not selected_cell_indices:
            raise ValueError(
                "selected_cell_indices cannot "
                "be empty."
            )

        if len(
            set(
                selected_cell_indices
            )
        ) != len(
            selected_cell_indices
        ):
            raise ValueError(
                "selected cell indices must "
                "be unique."
            )

        if any(
            cell_index < 0
            for cell_index
            in selected_cell_indices
        ):
            raise ValueError(
                "selected cell indices cannot "
                "be negative."
            )

        self.registry = registry

        self.selected_cell_indices = (
            tuple(
                int(cell_index)
                for cell_index
                in selected_cell_indices
            )
        )


    @property
    def num_streams(
        self,
    ) -> int:

        return len(
            self.selected_cell_indices
        )


    def __call__(
        self,
        tti_index: int,
        stream_index: int,
    ) -> torch.Tensor:

        if tti_index < 0:
            raise ValueError(
                "tti_index cannot be negative."
            )

        if not (
            0
            <= stream_index
            < self.num_streams
        ):
            raise ValueError(
                "stream_index outside selected "
                "cell range."
            )

        cell_index = (
            self
            .selected_cell_indices[
                stream_index
            ]
        )

        return (
            self
            .registry
            .global_ues_for_cell(
                cell_index
            )
        )
