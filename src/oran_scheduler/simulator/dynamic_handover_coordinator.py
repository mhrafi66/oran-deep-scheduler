from __future__ import annotations

from dataclasses import dataclass

import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
)

from oran_scheduler.simulator.dynamic_serving_population import (
    RegistrySelectedCellMembershipProvider,
)

from oran_scheduler.simulator.global_traffic_arrivals import (
    GlobalUETrafficArrivalProcess,
)

from oran_scheduler.simulator.global_ue_state import (
    GlobalUESchedulerStateRegistry,
)

from oran_scheduler.simulator.handover import (
    HandoverController,
    HandoverStepResult,
)

from oran_scheduler.simulator.handover_state_bridge import (
    CellLocalSchedulerState,
    materialize_cell_local_scheduler_state,
    synchronize_cell_local_scheduler_state,
)

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
)


@dataclass(frozen=True)
class DynamicHandoverTransition:
    """
    Result of one global association decision.

    local_states:
        freshly materialized selected-cell scheduler
        state AFTER applying the handover decision.

    moved_global_ue_indices:
        persistent UE identities whose serving BS
        changed in this step.

    affected_cell_indices:
        source + destination cells participating in
        at least one handover.
    """

    tti_index: int

    handover_result: HandoverStepResult

    moved_global_ue_indices: torch.Tensor

    affected_cell_indices: tuple[
        int,
        ...,
    ]

    local_states: tuple[
        CellLocalSchedulerState,
        ...,
    ]


class DynamicHandoverCoordinator:
    """
    Coordinate persistent global UE state with
    dynamically rebuilt cell-local scheduler state.

    Per handover decision:

        existing local state
            ->
        synchronize PF/aggregate queue globally
            ->
        evaluate handover
            ->
        update global serving BS
            ->
        rebuild selected-cell local state

    IMPORTANT
    ---------
    The ordering of rows in `link_power` must match
    the ordering in:

        registry.snapshot().global_ue_indices

    The coordinator does NOT itself execute PPO or
    classical scheduling.

    It prepares scientifically consistent dynamic
    scheduler state for the execution layer.

    Packet-level FIFO migration is owned separately
    by GlobalUEPacketQoSRegistry and will be connected
    in the runtime integration step.

    OPEN-REPRODUCTION:
        This is dynamic-association simulator
        infrastructure. It is not claimed to match a
        proprietary Nokia handover implementation.
    """

    def __init__(
        self,
        *,
        registry: GlobalUESchedulerStateRegistry,

        handover_controller: HandoverController,

        selected_cell_indices: tuple[
            int,
            ...,
        ],

        tds_config: PFTimeDomainConfig,

        throughput_forgetting_factor: float,

        ftp3_config: FTP3TrafficConfig,

        full_buffer_state_bits: float,

        traffic_arrival_process: (
            GlobalUETrafficArrivalProcess
        ),

        local_traffic_seed_base: int = 10000,
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
                "selected_cell_indices must "
                "be unique."
            )

        self.registry = registry

        self.handover_controller = (
            handover_controller
        )

        self.selected_cell_indices = (
            tuple(
                int(cell)
                for cell
                in selected_cell_indices
            )
        )

        self.tds_config = tds_config

        self.throughput_forgetting_factor = float(
            throughput_forgetting_factor
        )

        self.ftp3_config = ftp3_config

        self.full_buffer_state_bits = float(
            full_buffer_state_bits
        )

        self.traffic_arrival_process = (
            traffic_arrival_process
        )

        self.local_traffic_seed_base = int(
            local_traffic_seed_base
        )

        self.membership_provider = (
            RegistrySelectedCellMembershipProvider(
                registry=registry,

                selected_cell_indices=(
                    self.selected_cell_indices
                ),
            )
        )

        self._validate_global_alignment()


    @property
    def num_streams(
        self,
    ) -> int:

        return len(
            self.selected_cell_indices
        )


    def _validate_global_alignment(
        self,
    ) -> None:
        """
        Handover-controller row ordering and registry
        row ordering must refer to the same UE
        population.
        """

        registry_snapshot = (
            self.registry.snapshot()
        )

        controller_serving = (
            self
            .handover_controller
            .serving_bs
        )

        if tuple(
            controller_serving.shape
        ) != tuple(
            registry_snapshot
            .serving_bs
            .shape
        ):
            raise ValueError(
                "Handover controller and global "
                "registry have different UE counts."
            )

        if (
            controller_serving.device
            != registry_snapshot
            .serving_bs
            .device
        ):
            raise ValueError(
                "Handover controller and global "
                "registry are on different devices."
            )

        if not torch.equal(
            controller_serving,
            registry_snapshot.serving_bs,
        ):
            raise ValueError(
                "Handover controller serving-BS "
                "state does not match global "
                "registry."
            )


    def _validate_local_states(
        self,
        local_states: tuple[
            CellLocalSchedulerState,
            ...,
        ],
    ) -> None:

        if len(
            local_states
        ) != self.num_streams:
            raise ValueError(
                "local state count does not match "
                "selected stream count."
            )

        for (
            stream_index,
            local_state,
        ) in enumerate(
            local_states
        ):

            expected_cell = (
                self
                .selected_cell_indices[
                    stream_index
                ]
            )

            if (
                local_state.cell_index
                != expected_cell
            ):
                raise ValueError(
                    "Cell-local state is in the "
                    "wrong stream position."
                )

            expected_ids = (
                self.registry
                .global_ues_for_cell(
                    expected_cell
                )
            )

            if not torch.equal(
                local_state
                .global_ue_indices,
                expected_ids,
            ):
                raise ValueError(
                    "Cell-local UE membership is "
                    "stale relative to the global "
                    "registry."
                )


    def materialize_selected_cells(
        self,
    ) -> tuple[
        CellLocalSchedulerState,
        ...,
    ]:
        """
        Build scheduler/aggregate-traffic state for
        every selected physical cell using current
        global association.
        """

        states = []

        for (
            stream_index,
            cell_index,
        ) in enumerate(
            self.selected_cell_indices
        ):

            states.append(
                materialize_cell_local_scheduler_state(
                    registry=self.registry,

                    cell_index=cell_index,

                    tds_config=self.tds_config,

                    throughput_forgetting_factor=(
                        self
                        .throughput_forgetting_factor
                    ),

                    ftp3_config=self.ftp3_config,

                    full_buffer_state_bits=(
                        self
                        .full_buffer_state_bits
                    ),

                    traffic_seed=(
                        self
                        .local_traffic_seed_base
                        + stream_index
                    ),
                )
            )

        return tuple(
            states
        )


    def synchronize_selected_cells(
        self,
        local_states: tuple[
            CellLocalSchedulerState,
            ...,
        ],
    ) -> None:
        """
        Commit cell-local PF and aggregate queues to
        global UE identity before association changes.
        """

        self._validate_local_states(
            local_states
        )

        for local_state in local_states:

            synchronize_cell_local_scheduler_state(
                local_state=local_state,
                registry=self.registry,
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
        """
        Return this TTI's deterministic GLOBAL
        arrival realization gathered into CURRENT
        local serving-UE order.

        Therefore handover does not restart or
        resample UE traffic.
        """

        self._validate_local_states(
            local_states
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

        local_state = (
            local_states[
                stream_index
            ]
        )

        return (
            self
            .traffic_arrival_process
            .arrivals_for(
                tti_index=tti_index,

                global_ue_indices=(
                    local_state
                    .global_ue_indices
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
        Atomically:

            1. save current selected-cell PF/queues,
            2. perform handover decision,
            3. update global association,
            4. rematerialize selected-cell state.

        This should run BETWEEN scheduler TTIs.
        """

        self._validate_global_alignment()

        self.synchronize_selected_cells(
            local_states
        )

        before = (
            self.registry.snapshot()
        )

        handover_result = (
            self
            .handover_controller
            .step(
                tti_index=tti_index,

                link_power=link_power,
            )
        )

        #
        # The controller is the decision authority.
        # The registry then follows the completed
        # handover mask.
        #
        self.registry.apply_serving_bs_update(
            new_serving_bs=(
                handover_result
                .serving_bs
            ),

            handover_mask=(
                handover_result
                .handover_mask
            ),
        )

        after = (
            self.registry.snapshot()
        )

        if not torch.equal(
            after.serving_bs,
            handover_result.serving_bs,
        ):
            raise RuntimeError(
                "Global registry did not converge "
                "to handover-controller serving "
                "state."
            )

        moved_global_ue_indices = (
            before
            .global_ue_indices[
                handover_result
                .handover_mask
            ]
            .detach()
            .clone()
        )

        affected_cells: set[int] = set()

        moved_rows = torch.nonzero(
            handover_result.handover_mask,
            as_tuple=False,
        ).squeeze(-1)

        for row in (
            moved_rows
            .detach()
            .cpu()
            .tolist()
        ):

            affected_cells.add(
                int(
                    handover_result
                    .previous_serving_bs[
                        row
                    ]
                    .item()
                )
            )

            affected_cells.add(
                int(
                    handover_result
                    .serving_bs[
                        row
                    ]
                    .item()
                )
            )

        refreshed_local_states = (
            self.materialize_selected_cells()
        )

        return DynamicHandoverTransition(
            tti_index=tti_index,

            handover_result=(
                handover_result
            ),

            moved_global_ue_indices=(
                moved_global_ue_indices
            ),

            affected_cell_indices=(
                tuple(
                    sorted(
                        affected_cells
                    )
                )
            ),

            local_states=(
                refreshed_local_states
            ),
        )
