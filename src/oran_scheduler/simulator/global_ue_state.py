from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GlobalUEStateView:
    """
    State for an ordered subset of persistent global
    UE identities.

    All tensors have shape:

        [UE]

    The ordering is exactly the ordering requested by
    the caller.
    """

    global_ue_indices: torch.Tensor

    serving_bs: torch.Tensor

    average_throughput_bps: torch.Tensor

    buffer_bits: torch.Tensor

    full_buffer_mask: torch.Tensor


@dataclass(frozen=True)
class GlobalUEStateSnapshot:
    """
    Immutable snapshot of the complete global UE
    state registry.
    """

    global_ue_indices: torch.Tensor

    serving_bs: torch.Tensor

    average_throughput_bps: torch.Tensor

    buffer_bits: torch.Tensor

    full_buffer_mask: torch.Tensor


class GlobalUESchedulerStateRegistry:
    """
    Persistent scheduler/traffic state keyed by
    GLOBAL UE identity.

    WHY THIS EXISTS
    ---------------
    Existing cell state managers store state in local
    serving-slot order.

    That is valid while association is fixed:

        cell A:
            slot 0 -> UE 17
            slot 1 -> UE 42

    but breaks once a UE changes cell.

    This registry makes global UE identity the owner
    of persistent state.

    A handover changes:

        serving_bs

    but does NOT inherently reset:

        PF throughput history
        aggregate FTP backlog
        traffic class

    Packet-level FIFO state is deliberately handled
    separately in a later integration stage.

    OPEN-REPRODUCTION:
        This is simulator infrastructure for mobility
        experiments. It is not specified by the Nokia
        scheduler paper.
    """

    def __init__(
        self,
        *,
        global_ue_indices: torch.Tensor,
        serving_bs: torch.Tensor,
        average_throughput_bps: torch.Tensor,
        buffer_bits: torch.Tensor,
        full_buffer_mask: torch.Tensor,
    ) -> None:

        if global_ue_indices.ndim != 1:
            raise ValueError(
                "global_ue_indices must have "
                "shape [UE]."
            )

        num_ues = int(
            global_ue_indices.numel()
        )

        expected_shape = (
            num_ues,
        )

        tensors = {
            "serving_bs": serving_bs,
            "average_throughput_bps": (
                average_throughput_bps
            ),
            "buffer_bits": buffer_bits,
            "full_buffer_mask": (
                full_buffer_mask
            ),
        }

        for name, tensor in tensors.items():

            if tuple(
                tensor.shape
            ) != expected_shape:
                raise ValueError(
                    f"{name} must have "
                    f"shape {expected_shape}."
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

        if (
            serving_bs.dtype
            == torch.bool
            or torch.is_floating_point(
                serving_bs
            )
        ):
            raise ValueError(
                "serving_bs must use an "
                "integer dtype."
            )

        if (
            full_buffer_mask.dtype
            != torch.bool
        ):
            raise ValueError(
                "full_buffer_mask must use "
                "torch.bool."
            )

        if not torch.is_floating_point(
            average_throughput_bps
        ):
            raise ValueError(
                "average_throughput_bps must "
                "use floating point."
            )

        if not torch.is_floating_point(
            buffer_bits
        ):
            raise ValueError(
                "buffer_bits must use "
                "floating point."
            )

        device = (
            global_ue_indices.device
        )

        for name, tensor in tensors.items():

            if tensor.device != device:
                raise ValueError(
                    f"{name} is on the wrong "
                    "device."
                )

        if torch.any(
            global_ue_indices < 0
        ):
            raise ValueError(
                "global UE IDs cannot be "
                "negative."
            )

        if (
            torch.unique(
                global_ue_indices
            ).numel()
            != num_ues
        ):
            raise ValueError(
                "global UE IDs must be unique."
            )

        if torch.any(
            serving_bs < 0
        ):
            raise ValueError(
                "serving BS cannot be negative."
            )

        if not torch.isfinite(
            average_throughput_bps
        ).all():
            raise ValueError(
                "PF history contains non-finite "
                "values."
            )

        if torch.any(
            average_throughput_bps < 0.0
        ):
            raise ValueError(
                "PF history cannot be negative."
            )

        if not torch.isfinite(
            buffer_bits
        ).all():
            raise ValueError(
                "buffer state contains non-finite "
                "values."
            )

        if torch.any(
            buffer_bits < 0.0
        ):
            raise ValueError(
                "buffer state cannot be negative."
            )

        self.device = device

        self._global_ue_indices = (
            global_ue_indices
            .detach()
            .clone()
            .to(
                dtype=torch.long,
            )
        )

        self._serving_bs = (
            serving_bs
            .detach()
            .clone()
            .to(
                dtype=torch.long,
            )
        )

        self._average_throughput_bps = (
            average_throughput_bps
            .detach()
            .clone()
        )

        self._buffer_bits = (
            buffer_bits
            .detach()
            .clone()
        )

        self._full_buffer_mask = (
            full_buffer_mask
            .detach()
            .clone()
        )

        #
        # Full-buffer scheduler-visible buffer values
        # are state proxies, not real finite queues.
        #
        # Preserve their initial proxy values even if
        # later aggregate queue synchronization tries
        # to modify them.
        #
        self._full_buffer_proxy_bits = (
            self._buffer_bits
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


    @property
    def num_ues(
        self,
    ) -> int:

        return int(
            self
            ._global_ue_indices
            .numel()
        )


    def snapshot(
        self,
    ) -> GlobalUEStateSnapshot:

        return GlobalUEStateSnapshot(
            global_ue_indices=(
                self
                ._global_ue_indices
                .detach()
                .clone()
            ),

            serving_bs=(
                self
                ._serving_bs
                .detach()
                .clone()
            ),

            average_throughput_bps=(
                self
                ._average_throughput_bps
                .detach()
                .clone()
            ),

            buffer_bits=(
                self
                ._buffer_bits
                .detach()
                .clone()
            ),

            full_buffer_mask=(
                self
                ._full_buffer_mask
                .detach()
                .clone()
            ),
        )


    def _rows_for(
        self,
        global_ue_indices: torch.Tensor,
    ) -> torch.Tensor:

        if global_ue_indices.ndim != 1:
            raise ValueError(
                "requested global UE IDs must "
                "have shape [UE]."
            )

        if (
            global_ue_indices.dtype
            == torch.bool
            or torch.is_floating_point(
                global_ue_indices
            )
        ):
            raise ValueError(
                "requested global UE IDs must "
                "use an integer dtype."
            )

        ids = (
            global_ue_indices
            .detach()
            .cpu()
            .tolist()
        )

        rows: list[int] = []

        for global_ue in ids:

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

        return torch.tensor(
            rows,
            dtype=torch.long,
            device=self.device,
        )


    def gather(
        self,
        global_ue_indices: torch.Tensor,
    ) -> GlobalUEStateView:
        """
        Gather state in caller-requested global-UE
        order.
        """

        rows = self._rows_for(
            global_ue_indices
        )

        return GlobalUEStateView(
            global_ue_indices=(
                global_ue_indices
                .detach()
                .clone()
                .to(
                    device=self.device,
                    dtype=torch.long,
                )
            ),

            serving_bs=(
                self
                ._serving_bs
                .index_select(
                    0,
                    rows,
                )
                .detach()
                .clone()
            ),

            average_throughput_bps=(
                self
                ._average_throughput_bps
                .index_select(
                    0,
                    rows,
                )
                .detach()
                .clone()
            ),

            buffer_bits=(
                self
                ._buffer_bits
                .index_select(
                    0,
                    rows,
                )
                .detach()
                .clone()
            ),

            full_buffer_mask=(
                self
                ._full_buffer_mask
                .index_select(
                    0,
                    rows,
                )
                .detach()
                .clone()
            ),
        )


    def global_ues_for_cell(
        self,
        cell_index: int,
    ) -> torch.Tensor:
        """
        Return global UE IDs currently attached to
        one cell.

        UE IDs retain registry order for deterministic
        reproducibility.
        """

        if cell_index < 0:
            raise ValueError(
                "cell_index cannot be negative."
            )

        mask = (
            self._serving_bs
            == int(
                cell_index
            )
        )

        return (
            self
            ._global_ue_indices[
                mask
            ]
            .detach()
            .clone()
        )


    def cell_state(
        self,
        cell_index: int,
    ) -> GlobalUEStateView:

        return self.gather(
            self.global_ues_for_cell(
                cell_index
            )
        )


    def update_pf_history(
        self,
        *,
        global_ue_indices: torch.Tensor,
        average_throughput_bps: torch.Tensor,
    ) -> None:
        """
        Scatter cell-local PF state back into the
        global identity registry.
        """

        rows = self._rows_for(
            global_ue_indices
        )

        if tuple(
            average_throughput_bps.shape
        ) != tuple(
            global_ue_indices.shape
        ):
            raise ValueError(
                "PF update must match requested "
                "UE shape."
            )

        if (
            average_throughput_bps.device
            != self.device
        ):
            raise ValueError(
                "PF update is on the wrong device."
            )

        if not torch.is_floating_point(
            average_throughput_bps
        ):
            raise ValueError(
                "PF update must use floating point."
            )

        if (
            not torch.isfinite(
                average_throughput_bps
            ).all()
            or torch.any(
                average_throughput_bps
                < 0.0
            )
        ):
            raise ValueError(
                "PF update must be finite and "
                "non-negative."
            )

        self._average_throughput_bps[
            rows
        ] = (
            average_throughput_bps
            .detach()
            .to(
                dtype=(
                    self
                    ._average_throughput_bps
                    .dtype
                )
            )
        )


    def update_buffer_bits(
        self,
        *,
        global_ue_indices: torch.Tensor,
        buffer_bits: torch.Tensor,
    ) -> None:
        """
        Scatter aggregate queue state by global UE.

        Full-buffer UEs retain their scheduler-visible
        full-buffer proxy rather than being treated as
        finite queues.
        """

        rows = self._rows_for(
            global_ue_indices
        )

        if tuple(
            buffer_bits.shape
        ) != tuple(
            global_ue_indices.shape
        ):
            raise ValueError(
                "buffer update must match requested "
                "UE shape."
            )

        if (
            buffer_bits.device
            != self.device
        ):
            raise ValueError(
                "buffer update is on the wrong "
                "device."
            )

        if not torch.is_floating_point(
            buffer_bits
        ):
            raise ValueError(
                "buffer update must use floating "
                "point."
            )

        if (
            not torch.isfinite(
                buffer_bits
            ).all()
            or torch.any(
                buffer_bits < 0.0
            )
        ):
            raise ValueError(
                "buffer update must be finite and "
                "non-negative."
            )

        local_full_buffer = (
            self
            ._full_buffer_mask[
                rows
            ]
        )

        current = (
            self
            ._buffer_bits[
                rows
            ]
        )

        candidate = (
            buffer_bits
            .detach()
            .to(
                dtype=(
                    self
                    ._buffer_bits
                    .dtype
                )
            )
        )

        self._buffer_bits[
            rows
        ] = torch.where(
            local_full_buffer,
            current,
            candidate,
        )


    def apply_serving_bs_update(
        self,
        *,
        new_serving_bs: torch.Tensor,
        handover_mask: torch.Tensor,
    ) -> None:
        """
        Apply association changes WITHOUT resetting
        any UE-owned scheduler/traffic state.

        Intended input:
            HandoverStepResult.serving_bs
            HandoverStepResult.handover_mask
        """

        expected_shape = (
            self.num_ues,
        )

        if tuple(
            new_serving_bs.shape
        ) != expected_shape:
            raise ValueError(
                "new_serving_bs shape mismatch."
            )

        if tuple(
            handover_mask.shape
        ) != expected_shape:
            raise ValueError(
                "handover_mask shape mismatch."
            )

        if (
            new_serving_bs.device
            != self.device
            or handover_mask.device
            != self.device
        ):
            raise ValueError(
                "handover update is on the wrong "
                "device."
            )

        if (
            new_serving_bs.dtype
            == torch.bool
            or torch.is_floating_point(
                new_serving_bs
            )
        ):
            raise ValueError(
                "new_serving_bs must use an "
                "integer dtype."
            )

        if (
            handover_mask.dtype
            != torch.bool
        ):
            raise ValueError(
                "handover_mask must use "
                "torch.bool."
            )

        if torch.any(
            new_serving_bs < 0
        ):
            raise ValueError(
                "new serving BS cannot be "
                "negative."
            )

        self._serving_bs = torch.where(
            handover_mask,
            new_serving_bs.to(
                dtype=torch.long,
            ),
            self._serving_bs,
        )
