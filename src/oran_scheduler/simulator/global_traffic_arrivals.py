from __future__ import annotations

from dataclasses import dataclass

import torch

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
)


@dataclass(frozen=True)
class GlobalTrafficArrivalSnapshot:
    """
    One TTI of packet arrivals indexed by persistent
    GLOBAL UE identity.

    packet_arrivals:
        int64 [UE]

    Full-Buffer UEs always have zero packet arrivals
    because their infinite backlog is represented
    separately.
    """

    tti_index: int

    global_ue_indices: torch.Tensor

    packet_arrivals: torch.Tensor


class GlobalUETrafficArrivalProcess:
    """
    Association-independent FTP3 arrival generator.

    WHY THIS EXISTS
    ---------------
    TrafficBufferManager currently owns a random
    generator in CELL-LOCAL state.

    Reconstructing a cell-local manager after
    handover would therefore restart/change the
    packet-arrival process.

    Instead, this object generates arrivals for the
    complete persistent GLOBAL UE population.

    Cell-local managers receive deterministic slices
    of this global TTI realization.

    Therefore:

        UE traffic realization
            does NOT change
        when
            UE serving cell changes.

    The global UE ordering is fixed at construction.

    OPEN-REPRODUCTION:
        This is simulator infrastructure for dynamic
        association experiments. It is not specified
        by the Nokia scheduler paper.
    """

    def __init__(
        self,
        *,
        global_ue_indices: torch.Tensor,
        full_buffer_mask: torch.Tensor,
        ftp3_config: FTP3TrafficConfig,
        seed: int,
    ) -> None:

        if global_ue_indices.ndim != 1:
            raise ValueError(
                "global_ue_indices must have "
                "shape [UE]."
            )

        if (
            global_ue_indices.dtype
            == torch.bool
            or torch.is_floating_point(
                global_ue_indices
            )
        ):
            raise ValueError(
                "global_ue_indices must use "
                "an integer dtype."
            )

        if tuple(
            full_buffer_mask.shape
        ) != tuple(
            global_ue_indices.shape
        ):
            raise ValueError(
                "full_buffer_mask must match "
                "global UE shape."
            )

        if (
            full_buffer_mask.dtype
            != torch.bool
        ):
            raise ValueError(
                "full_buffer_mask must use "
                "torch.bool."
            )

        if torch.any(
            global_ue_indices < 0
        ):
            raise ValueError(
                "global UE IDs cannot be negative."
            )

        if (
            torch.unique(
                global_ue_indices
            ).numel()
            != global_ue_indices.numel()
        ):
            raise ValueError(
                "global UE IDs must be unique."
            )

        self.device = (
            global_ue_indices.device
        )

        self.ftp3_config = (
            ftp3_config
        )

        self._global_ue_indices = (
            global_ue_indices
            .detach()
            .clone()
            .to(
                dtype=torch.long,
            )
        )

        self._full_buffer_mask = (
            full_buffer_mask
            .detach()
            .clone()
        )

        self._row_by_global_ue = {
            int(global_ue): row
            for row, global_ue
            in enumerate(
                self
                ._global_ue_indices
                .detach()
                .cpu()
                .tolist()
            )
        }

        #
        # Deliberately use CPU RNG.
        #
        # The generated integer arrival trace is then
        # moved to the simulation device.
        #
        # This prevents GPU/device changes from
        # changing the traffic realization.
        #
        self._generator = (
            torch.Generator(
                device="cpu"
            )
        )

        self._generator.manual_seed(
            int(seed)
        )

        self._last_tti_index: (
            int | None
        ) = None

        self._current_snapshot: (
            GlobalTrafficArrivalSnapshot | None
        ) = None


    @property
    def num_ues(
        self,
    ) -> int:

        return int(
            self
            ._global_ue_indices
            .numel()
        )


    def _sample_new_tti(
        self,
        *,
        tti_index: int,
    ) -> GlobalTrafficArrivalSnapshot:

        expected = torch.full(
            (
                self.num_ues,
            ),

            fill_value=(
                self
                .ftp3_config
                .expected_packets_per_tti
            ),

            dtype=torch.float32,

            device="cpu",
        )

        full_buffer_cpu = (
            self
            ._full_buffer_mask
            .detach()
            .cpu()
        )

        #
        # Full-Buffer traffic has no finite FTP
        # arrival process.
        #
        expected[
            full_buffer_cpu
        ] = 0.0

        sampled = torch.poisson(
            expected,
            generator=self._generator,
        ).to(
            dtype=torch.long
        )

        sampled[
            full_buffer_cpu
        ] = 0

        snapshot = (
            GlobalTrafficArrivalSnapshot(
                tti_index=tti_index,

                global_ue_indices=(
                    self
                    ._global_ue_indices
                    .detach()
                    .clone()
                ),

                packet_arrivals=(
                    sampled.to(
                        device=self.device
                    )
                ),
            )
        )

        self._last_tti_index = (
            int(
                tti_index
            )
        )

        self._current_snapshot = (
            snapshot
        )

        return snapshot


    def snapshot_for_tti(
        self,
        tti_index: int,
    ) -> GlobalTrafficArrivalSnapshot:
        """
        Return one global arrival realization.

        Multiple requests for the SAME TTI are
        idempotent. This matters because multiple
        cells request their local slice during the
        same synchronized multicell TTI.

        New TTIs must advance consecutively so that
        accidental skipped traffic intervals are not
        silently ignored.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index cannot be negative."
            )

        if (
            self._last_tti_index
            is None
        ):
            return self._sample_new_tti(
                tti_index=tti_index
            )

        if (
            tti_index
            == self._last_tti_index
        ):
            assert (
                self._current_snapshot
                is not None
            )

            return (
                self
                ._current_snapshot
            )

        if (
            tti_index
            != self._last_tti_index + 1
        ):
            raise ValueError(
                "Traffic TTIs must advance "
                "consecutively."
            )

        return self._sample_new_tti(
            tti_index=tti_index
        )


    def arrivals_for(
        self,
        *,
        tti_index: int,
        global_ue_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather one cell's arrivals by GLOBAL UE
        identity.

        Local slot ordering is irrelevant.
        """

        if global_ue_indices.ndim != 1:
            raise ValueError(
                "global_ue_indices must have "
                "shape [UE]."
            )

        if (
            global_ue_indices.dtype
            == torch.bool
            or torch.is_floating_point(
                global_ue_indices
            )
        ):
            raise ValueError(
                "global_ue_indices must use "
                "an integer dtype."
            )

        snapshot = self.snapshot_for_tti(
            tti_index
        )

        rows: list[int] = []

        for global_ue in (
            global_ue_indices
            .detach()
            .cpu()
            .tolist()
        ):

            key = int(
                global_ue
            )

            if (
                key
                not in self._row_by_global_ue
            ):
                raise KeyError(
                    f"Unknown global UE {key}."
                )

            rows.append(
                self
                ._row_by_global_ue[
                    key
                ]
            )

        row_tensor = torch.tensor(
            rows,
            dtype=torch.long,
            device=self.device,
        )

        return (
            snapshot
            .packet_arrivals
            .index_select(
                0,
                row_tensor,
            )
            .detach()
            .clone()
        )
