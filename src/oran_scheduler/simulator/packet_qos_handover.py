from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import torch

from oran_scheduler.simulator.packet_qos_runtime import (
    PacketQoSTrafficBufferManager,
    _TrackedPacket,
)


@dataclass(frozen=True)
class PacketQoSPacketState:
    """
    Persistent state of one queued FTP packet.

    arrival_tti:
        Original arrival TTI. This must survive
        handover so delay/HoL metrics remain valid.

    remaining_bits:
        Unserved packet payload.

    deadline_missed:
        Whether this packet has already crossed its
        configured deadline.

        Preserving this prevents counting the same
        deadline violation again after handover.
    """

    arrival_tti: int

    remaining_bits: float

    deadline_missed: bool = False


    def __post_init__(
        self,
    ) -> None:

        if self.arrival_tti < 0:
            raise ValueError(
                "arrival_tti cannot be negative."
            )

        if (
            not math.isfinite(
                self.remaining_bits
            )
            or self.remaining_bits <= 0.0
        ):
            raise ValueError(
                "remaining_bits must be finite "
                "and positive."
            )


@dataclass(frozen=True)
class PacketQoSUEState:
    """
    FIFO packet state owned by one persistent
    GLOBAL UE identity.
    """

    packets: tuple[
        PacketQoSPacketState,
        ...,
    ] = tuple()


    @property
    def queue_bits(
        self,
    ) -> float:

        return float(
            sum(
                packet.remaining_bits
                for packet in self.packets
            )
        )


def _validate_local_ue_index(
    *,
    manager: PacketQoSTrafficBufferManager,
    ue_index: int,
) -> None:

    if ue_index < 0:
        raise ValueError(
            "ue_index cannot be negative."
        )

    num_ues = int(
        manager
        .current_buffer_bits
        .numel()
    )

    if ue_index >= num_ues:
        raise ValueError(
            "ue_index is outside manager "
            "population."
        )


def export_packet_qos_ue_state(
    *,
    manager: PacketQoSTrafficBufferManager,
    ue_index: int,
) -> PacketQoSUEState:
    """
    Export one local UE's shadow FIFO.

    Export is legal only BETWEEN TTIs.

    TEMPORARY INTEGRATION NOTE:
        PacketQoSTrafficBufferManager does not yet
        expose a public per-UE FIFO state API.

        This handover bridge intentionally accesses
        its internal packet queue until the complete
        dynamic-association path is stabilized.
    """

    _validate_local_ue_index(
        manager=manager,
        ue_index=ue_index,
    )

    if manager.has_active_tti:
        raise RuntimeError(
            "Cannot export packet QoS state "
            "during an active TTI."
        )

    queue = (
        manager
        ._packet_queues[
            ue_index
        ]
    )

    is_full_buffer = bool(
        manager
        ._qos_full_buffer_mask[
            ue_index
        ].item()
    )

    if (
        is_full_buffer
        and len(queue) != 0
    ):
        raise RuntimeError(
            "Full-Buffer UE unexpectedly owns "
            "packet FIFO state."
        )

    packets = tuple(
        PacketQoSPacketState(
            arrival_tti=int(
                packet.arrival_tti
            ),

            remaining_bits=float(
                packet.remaining_bits
            ),

            deadline_missed=bool(
                packet.deadline_missed
            ),
        )
        for packet in queue
    )

    return PacketQoSUEState(
        packets=packets
    )


def restore_packet_qos_ue_state(
    *,
    manager: PacketQoSTrafficBufferManager,
    ue_index: int,
    state: PacketQoSUEState,
) -> None:
    """
    Restore one UE FIFO into one cell-local manager.

    The manager's aggregate FTP queue is synchronized
    to the restored packet sum so aggregate and packet
    views remain scientifically consistent.
    """

    _validate_local_ue_index(
        manager=manager,
        ue_index=ue_index,
    )

    if manager.has_active_tti:
        raise RuntimeError(
            "Cannot restore packet QoS state "
            "during an active TTI."
        )

    is_full_buffer = bool(
        manager
        ._qos_full_buffer_mask[
            ue_index
        ].item()
    )

    if (
        is_full_buffer
        and state.packets
    ):
        raise ValueError(
            "Full-Buffer UE cannot restore "
            "finite packet FIFO state."
        )

    rebuilt = deque(
        _TrackedPacket(
            arrival_tti=(
                packet.arrival_tti
            ),

            remaining_bits=(
                packet.remaining_bits
            ),

            deadline_missed=(
                packet.deadline_missed
            ),
        )
        for packet in state.packets
    )

    manager._packet_queues[
        ue_index
    ] = rebuilt

    if is_full_buffer:
        #
        # Keep TrafficBufferManager's normal
        # Full-Buffer scheduler proxy.
        #
        return

    manager._buffer_bits[
        ue_index
    ] = torch.as_tensor(
        state.queue_bits,
        dtype=(
            manager
            ._buffer_bits
            .dtype
        ),
        device=(
            manager
            ._buffer_bits
            .device
        ),
    )


class GlobalUEPacketQoSRegistry:
    """
    Packet-level FIFO state keyed by persistent
    GLOBAL UE identity.

    This complements GlobalUESchedulerStateRegistry:

        GlobalUESchedulerStateRegistry
            PF history
            aggregate buffer
            traffic class
            serving BS

        GlobalUEPacketQoSRegistry
            packet arrival times
            packet remaining bits
            deadline-missed flags

    A cell-local PacketQoSTrafficBufferManager is
    therefore a temporary execution view rather than
    the long-term owner of packet identity state.
    """

    def __init__(
        self,
        *,
        global_ue_indices: torch.Tensor,
        full_buffer_mask: torch.Tensor,
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

        ids = (
            global_ue_indices
            .detach()
            .cpu()
            .tolist()
        )

        if len(
            set(
                int(value)
                for value in ids
            )
        ) != len(ids):
            raise ValueError(
                "global UE IDs must be unique."
            )

        if any(
            int(value) < 0
            for value in ids
        ):
            raise ValueError(
                "global UE IDs cannot be negative."
            )

        fb = (
            full_buffer_mask
            .detach()
            .cpu()
            .tolist()
        )

        self._full_buffer_by_ue = {
            int(global_ue): bool(
                is_full_buffer
            )
            for (
                global_ue,
                is_full_buffer,
            ) in zip(
                ids,
                fb,
                strict=True,
            )
        }

        self._state_by_ue = {
            int(global_ue): (
                PacketQoSUEState()
            )
            for global_ue in ids
        }


    def state_for(
        self,
        global_ue_index: int,
    ) -> PacketQoSUEState:

        key = int(
            global_ue_index
        )

        if key not in self._state_by_ue:
            raise KeyError(
                f"Unknown global UE {key}."
            )

        return self._state_by_ue[
            key
        ]


    def set_state(
        self,
        *,
        global_ue_index: int,
        state: PacketQoSUEState,
    ) -> None:
        """
        Initialize or replace one persistent UE's
        packet FIFO state.

        This is primarily useful when constructing
        scientifically controlled initial conditions.

        Full-Buffer UEs may not own finite packet
        FIFO state.
        """

        key = int(
            global_ue_index
        )

        if key not in self._state_by_ue:
            raise KeyError(
                f"Unknown global UE {key}."
            )

        if (
            self._full_buffer_by_ue[
                key
            ]
            and state.packets
        ):
            raise ValueError(
                "Full-Buffer UE cannot own "
                "finite packet FIFO state."
            )

        self._state_by_ue[
            key
        ] = state


    def synchronize_from_local(
        self,
        *,
        global_ue_indices: torch.Tensor,
        manager: PacketQoSTrafficBufferManager,
    ) -> None:
        """
        Export every local UE queue into global
        identity-owned storage.
        """

        if global_ue_indices.ndim != 1:
            raise ValueError(
                "global_ue_indices must have "
                "shape [UE]."
            )

        if int(
            global_ue_indices.numel()
        ) != int(
            manager
            .current_buffer_bits
            .numel()
        ):
            raise ValueError(
                "Local UE identity count does not "
                "match packet manager population."
            )

        local_fb = (
            manager
            .full_buffer_mask
            .detach()
            .cpu()
            .tolist()
        )

        ids = (
            global_ue_indices
            .detach()
            .cpu()
            .tolist()
        )

        for (
            local_index,
            global_ue,
        ) in enumerate(
            ids
        ):

            key = int(
                global_ue
            )

            if key not in self._state_by_ue:
                raise KeyError(
                    f"Unknown global UE {key}."
                )

            expected_fb = (
                self
                ._full_buffer_by_ue[
                    key
                ]
            )

            if bool(
                local_fb[
                    local_index
                ]
            ) != expected_fb:
                raise ValueError(
                    "Traffic class mismatch for "
                    f"global UE {key}."
                )

            self._state_by_ue[
                key
            ] = (
                export_packet_qos_ue_state(
                    manager=manager,
                    ue_index=local_index,
                )
            )


    def restore_to_local(
        self,
        *,
        global_ue_indices: torch.Tensor,
        manager: PacketQoSTrafficBufferManager,
    ) -> None:
        """
        Restore global packet FIFO state into current
        cell-local UE ordering.

        This is identity based, not slot based.
        """

        if global_ue_indices.ndim != 1:
            raise ValueError(
                "global_ue_indices must have "
                "shape [UE]."
            )

        if int(
            global_ue_indices.numel()
        ) != int(
            manager
            .current_buffer_bits
            .numel()
        ):
            raise ValueError(
                "Local UE identity count does not "
                "match packet manager population."
            )

        local_fb = (
            manager
            .full_buffer_mask
            .detach()
            .cpu()
            .tolist()
        )

        ids = (
            global_ue_indices
            .detach()
            .cpu()
            .tolist()
        )

        for (
            local_index,
            global_ue,
        ) in enumerate(
            ids
        ):

            key = int(
                global_ue
            )

            if key not in self._state_by_ue:
                raise KeyError(
                    f"Unknown global UE {key}."
                )

            expected_fb = (
                self
                ._full_buffer_by_ue[
                    key
                ]
            )

            if bool(
                local_fb[
                    local_index
                ]
            ) != expected_fb:
                raise ValueError(
                    "Traffic class mismatch for "
                    f"global UE {key}."
                )

            restore_packet_qos_ue_state(
                manager=manager,
                ue_index=local_index,
                state=(
                    self
                    ._state_by_ue[
                        key
                    ]
                ),
            )
