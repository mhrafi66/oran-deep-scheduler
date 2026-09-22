from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)

from oran_scheduler.simulator.global_ue_state import (
    GlobalUESchedulerStateRegistry,
)

from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIStateManager,
)

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
    TrafficBufferManager,
)


@dataclass(frozen=True)
class CellLocalSchedulerState:
    """
    Cell-local scheduler/traffic state materialized
    from persistent GLOBAL UE state.

    global_ue_indices:
        Persistent identities represented by this
        cell-local state.

    state_manager:
        PF / scheduler history in this cell's local
        serving order.

    traffic_manager:
        Aggregate traffic queues in the same local
        serving order.

    IMPORTANT:
        This object is a temporary local view.

        Persistent ownership belongs to
        GlobalUESchedulerStateRegistry.
    """

    cell_index: int

    global_ue_indices: torch.Tensor

    state_manager: OneLDSCellTTIStateManager

    traffic_manager: TrafficBufferManager


def materialize_cell_local_scheduler_state(
    *,
    registry: GlobalUESchedulerStateRegistry,
    cell_index: int,
    tds_config: PFTimeDomainConfig,
    throughput_forgetting_factor: float,
    ftp3_config: FTP3TrafficConfig,
    full_buffer_state_bits: float,
    traffic_seed: int | None = None,
) -> CellLocalSchedulerState:
    """
    Build local scheduler + aggregate traffic state
    from global UE-owned state.

    Intended use around association changes:

        global state
            ->
        materialize current cell membership
            ->
        execute scheduler TTIs
            ->
        synchronize updated state back globally

    This preserves:
        - global UE identity,
        - PF throughput history,
        - FTP backlog,
        - Full-Buffer traffic class.

    Packet-level QoS FIFO state is NOT yet handled
    here. That is the next integration layer.
    """

    if cell_index < 0:
        raise ValueError(
            "cell_index cannot be negative."
        )

    if (
        not math.isfinite(
            throughput_forgetting_factor
        )
        or not (
            0.0
            <= throughput_forgetting_factor
            <= 1.0
        )
    ):
        raise ValueError(
            "throughput_forgetting_factor must "
            "lie in [0, 1]."
        )

    if (
        not math.isfinite(
            full_buffer_state_bits
        )
        or full_buffer_state_bits <= 0.0
    ):
        raise ValueError(
            "full_buffer_state_bits must be "
            "positive and finite."
        )

    view = registry.cell_state(
        cell_index
    )

    num_ues = int(
        view.global_ue_indices.numel()
    )

    if num_ues == 0:
        raise ValueError(
            "Cannot materialize an empty cell yet. "
            "Empty-cell runtime support will be "
            "added during end-to-end integration."
        )

    #
    # The current TrafficBufferManager represents
    # Full Buffer with one scalar scheduler-visible
    # proxy.
    #
    # Therefore all global FB entries must agree
    # with the requested proxy.
    #
    if torch.any(
        view.full_buffer_mask
    ):

        fb_values = (
            view.buffer_bits[
                view.full_buffer_mask
            ]
        )

        expected = torch.full_like(
            fb_values,
            fill_value=(
                full_buffer_state_bits
            ),
        )

        if not torch.allclose(
            fb_values,
            expected,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                "Global Full-Buffer proxy state "
                "does not match "
                "full_buffer_state_bits."
            )

    valid_mask = torch.ones(
        num_ues,
        dtype=torch.bool,
        device=(
            registry.device
        ),
    )

    state_manager = (
        OneLDSCellTTIStateManager(
            initial_average_throughput_bps=(
                view
                .average_throughput_bps
            ),

            serving_global_ue_indices=(
                view.global_ue_indices
            ),

            serving_ue_valid_mask=(
                valid_mask
            ),

            tds_config=tds_config,

            throughput_forgetting_factor=(
                throughput_forgetting_factor
            ),
        )
    )

    #
    # TrafficBufferManager ignores finite backlog
    # semantics for Full-Buffer UEs and substitutes
    # full_buffer_state_bits.
    #
    # Set the constructor's finite queue input to
    # zero for FB positions and preserve real FTP
    # queues exactly.
    #
    initial_ftp_buffer_bits = torch.where(
        view.full_buffer_mask,

        torch.zeros_like(
            view.buffer_bits
        ),

        view.buffer_bits,
    )

    traffic_manager = TrafficBufferManager(
        full_buffer_mask=(
            view.full_buffer_mask
        ),

        ftp3_config=ftp3_config,

        full_buffer_state_bits=(
            full_buffer_state_bits
        ),

        initial_ftp_buffer_bits=(
            initial_ftp_buffer_bits
        ),

        seed=traffic_seed,
    )

    return CellLocalSchedulerState(
        cell_index=cell_index,

        global_ue_indices=(
            view
            .global_ue_indices
            .detach()
            .clone()
        ),

        state_manager=(
            state_manager
        ),

        traffic_manager=(
            traffic_manager
        ),
    )


def synchronize_cell_local_scheduler_state(
    *,
    local_state: CellLocalSchedulerState,
    registry: GlobalUESchedulerStateRegistry,
) -> None:
    """
    Scatter cell-local PF and aggregate queue state
    back to persistent GLOBAL UE identity.

    Call only between TTIs.

    This does NOT modify association.
    Association is owned by the handover controller
    / registry serving-BS update.
    """

    if (
        local_state
        .state_manager
        .has_active_tti
    ):
        raise RuntimeError(
            "Cannot synchronize PF state while "
            "scheduler TTI is active."
        )

    if (
        local_state
        .traffic_manager
        .has_active_tti
    ):
        raise RuntimeError(
            "Cannot synchronize traffic state while "
            "traffic TTI is active."
        )

    registry.update_pf_history(
        global_ue_indices=(
            local_state
            .global_ue_indices
        ),

        average_throughput_bps=(
            local_state
            .state_manager
            .current_average_throughput_bps
        ),
    )

    registry.update_buffer_bits(
        global_ue_indices=(
            local_state
            .global_ue_indices
        ),

        buffer_bits=(
            local_state
            .traffic_manager
            .current_buffer_bits
        ),
    )
