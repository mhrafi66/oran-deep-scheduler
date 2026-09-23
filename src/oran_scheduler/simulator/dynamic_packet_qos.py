from __future__ import annotations

from dataclasses import replace

import torch

from oran_scheduler.simulator.dynamic_handover_coordinator import (
    DynamicHandoverCoordinator,
    DynamicHandoverTransition,
)

from oran_scheduler.simulator.handover_state_bridge import (
    CellLocalSchedulerState,
)

from oran_scheduler.simulator.packet_qos_handover import (
    GlobalUEPacketQoSRegistry,
)

from oran_scheduler.simulator.packet_qos_runtime import (
    PacketQoSTrafficBufferManager,
)


class PacketQoSDynamicHandoverCoordinator:
    """
    Packet-QoS-aware wrapper around the already
    validated DynamicHandoverCoordinator.

    Persistent ownership is split deliberately:

        DynamicHandoverCoordinator
            serving BS
            PF history
            aggregate queue
            traffic class
            global traffic arrival process

        GlobalUEPacketQoSRegistry
            packet arrival TTI
            remaining packet bits
            deadline-missed state

    Cell-local PacketQoSTrafficBufferManager objects
    are temporary execution views.

    During handover:

        local packet FIFO
            ->
        global packet registry
            ->
        association update
            ->
        new cell-local packet manager
            ->
        restore FIFO by GLOBAL UE identity

    OPEN-REPRODUCTION:
        This is simulator infrastructure for dynamic
        association experiments and is not specified
        by the Nokia scheduler paper.
    """

    def __init__(
        self,
        *,
        base_coordinator: DynamicHandoverCoordinator,

        packet_registry: (
            GlobalUEPacketQoSRegistry
        ),

        packet_qos_deadline_ttis: (
            int | None
        ) = None,
    ) -> None:

        if (
            packet_qos_deadline_ttis
            is not None
            and packet_qos_deadline_ttis <= 0
        ):
            raise ValueError(
                "packet_qos_deadline_ttis must "
                "be positive when provided."
            )

        self.base_coordinator = (
            base_coordinator
        )

        self.packet_registry = (
            packet_registry
        )

        self.packet_qos_deadline_ttis = (
            packet_qos_deadline_ttis
        )


    @property
    def registry(
        self,
    ):
        return (
            self
            .base_coordinator
            .registry
        )


    @property
    def num_streams(
        self,
    ) -> int:

        return (
            self
            .base_coordinator
            .num_streams
        )


    @property
    def selected_cell_indices(
        self,
    ):

        return (
            self
            .base_coordinator
            .selected_cell_indices
        )


    @property
    def membership_provider(
        self,
    ):

        return (
            self
            .base_coordinator
            .membership_provider
        )


    @property
    def traffic_arrival_process(
        self,
    ):

        return (
            self
            .base_coordinator
            .traffic_arrival_process
        )


    def _packetize_local_state(
        self,
        *,
        stream_index: int,
        local_state: CellLocalSchedulerState,
    ) -> CellLocalSchedulerState:
        """
        Replace one ordinary aggregate traffic
        manager with a packet-aware local manager,
        then restore the UE FIFOs from global
        identity-owned state.

        Aggregate queue bits and packet FIFO bits
        MUST agree.

        We refuse to silently invent packet ages for
        an unexplained pre-existing aggregate queue.
        """

        view = (
            self
            .registry
            .gather(
                local_state
                .global_ue_indices
            )
        )

        initial_ftp_buffer_bits = (
            torch.where(
                view.full_buffer_mask,

                torch.zeros_like(
                    view.buffer_bits
                ),

                view.buffer_bits,
            )
        )

        manager = (
            PacketQoSTrafficBufferManager(
                full_buffer_mask=(
                    view
                    .full_buffer_mask
                ),

                ftp3_config=(
                    self
                    .base_coordinator
                    .ftp3_config
                ),

                full_buffer_state_bits=(
                    self
                    .base_coordinator
                    .full_buffer_state_bits
                ),

                initial_ftp_buffer_bits=(
                    initial_ftp_buffer_bits
                ),

                seed=(
                    self
                    .base_coordinator
                    .local_traffic_seed_base
                    + stream_index
                ),

                packet_qos_enabled=True,

                packet_qos_cell_index=(
                    local_state
                    .cell_index
                ),

                packet_qos_deadline_ttis=(
                    self
                    .packet_qos_deadline_ttis
                ),

                #
                # Dynamic cell-local managers are
                # rebuilt after reassociation.
                #
                # Do not use their per-instance CSV
                # cumulative counters as global
                # experiment metrics.
                #
                packet_qos_csv_path=None,
            )
        )

        self.packet_registry.restore_to_local(
            global_ue_indices=(
                local_state
                .global_ue_indices
            ),

            manager=manager,
        )

        restored_bits = (
            manager
            .current_buffer_bits
        )

        if not torch.allclose(
            restored_bits,
            view.buffer_bits,
            rtol=0.0,
            atol=1.0e-4,
        ):
            raise RuntimeError(
                "Aggregate global queue and "
                "packet-level global FIFO disagree. "
                "Dynamic packet QoS cannot invent "
                "packet arrival history for an "
                "existing aggregate backlog."
            )

        return CellLocalSchedulerState(
            cell_index=(
                local_state.cell_index
            ),

            global_ue_indices=(
                local_state
                .global_ue_indices
                .detach()
                .clone()
            ),

            state_manager=(
                local_state.state_manager
            ),

            traffic_manager=manager,
        )


    def _packetize_states(
        self,
        states: tuple[
            CellLocalSchedulerState,
            ...,
        ],
    ) -> tuple[
        CellLocalSchedulerState,
        ...,
    ]:

        return tuple(
            self._packetize_local_state(
                stream_index=(
                    stream_index
                ),

                local_state=(
                    local_state
                ),
            )

            for (
                stream_index,
                local_state,
            ) in enumerate(
                states
            )
        )


    def materialize_selected_cells(
        self,
    ) -> tuple[
        CellLocalSchedulerState,
        ...,
    ]:

        ordinary_states = (
            self
            .base_coordinator
            .materialize_selected_cells()
        )

        return self._packetize_states(
            ordinary_states
        )


    def synchronize_selected_cells(
        self,
        local_states: tuple[
            CellLocalSchedulerState,
            ...,
        ],
    ) -> None:
        """
        Commit BOTH packet-level and aggregate state
        before any serving-cell change.
        """

        if len(
            local_states
        ) != self.num_streams:
            raise ValueError(
                "local state count does not "
                "match selected stream count."
            )

        for local_state in local_states:

            manager = (
                local_state
                .traffic_manager
            )

            if not isinstance(
                manager,
                PacketQoSTrafficBufferManager,
            ):
                raise TypeError(
                    "Packet-QoS dynamic coordinator "
                    "requires "
                    "PacketQoSTrafficBufferManager."
                )

            self.packet_registry.synchronize_from_local(
                global_ue_indices=(
                    local_state
                    .global_ue_indices
                ),

                manager=manager,
            )

        #
        # Aggregate queue and PF history are then
        # synchronized through the already-tested
        # base coordinator.
        #
        self.base_coordinator.synchronize_selected_cells(
            local_states
        )


    def packet_arrivals_for_stream(
        self,
        *,
        tti_index: int,
        stream_index: int,

        local_states: tuple[
            CellLocalSchedulerState,
            ...,
        ],
    ) -> torch.Tensor:

        return (
            self
            .base_coordinator
            .packet_arrivals_for_stream(
                tti_index=tti_index,

                stream_index=stream_index,

                local_states=(
                    local_states
                ),
            )
        )


    def advance_handover(
        self,
        *,
        tti_index: int,

        link_power: torch.Tensor,

        local_states: tuple[
            CellLocalSchedulerState,
            ...,
        ],
    ) -> DynamicHandoverTransition:
        """
        Preserve packet FIFOs across one association
        transition.

        The base coordinator already performs:

            aggregate synchronization
            handover decision
            serving-BS update
            local rematerialization

        We add:

            packet export BEFORE handover
            packet restore AFTER handover
        """

        #
        # Critical ordering:
        #
        # Save packet state while source-cell local
        # identities still describe the OLD
        # association.
        #
        self.synchronize_selected_cells(
            local_states
        )

        #
        # The base coordinator performs its own
        # aggregate synchronization once more.
        #
        # That duplicate aggregate scatter is
        # intentionally harmless and lets us retain
        # the already-tested base transition logic.
        #
        transition = (
            self
            .base_coordinator
            .advance_handover(
                tti_index=tti_index,

                link_power=link_power,

                local_states=(
                    local_states
                ),
            )
        )

        packet_states = (
            self._packetize_states(
                transition.local_states
            )
        )

        return replace(
            transition,

            local_states=(
                packet_states
            ),
        )
