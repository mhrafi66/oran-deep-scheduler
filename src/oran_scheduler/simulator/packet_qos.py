from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math


@dataclass
class Packet:
    """
    One queued packet.

    arrival_tti:
        TTI at which the packet entered the queue.

    size_bits:
        Original packet size.

    remaining_bits:
        Bits still awaiting service.

    deadline_tti:
        Optional inclusive service deadline.

        A packet still unfinished when simulation
        advances past deadline_tti is considered a
        deadline miss.
    """

    arrival_tti: int
    size_bits: float
    remaining_bits: float
    deadline_tti: int | None = None
    traffic_class: str = "best_effort"

    def __post_init__(
        self,
    ) -> None:

        if self.arrival_tti < 0:
            raise ValueError(
                "arrival_tti must be non-negative."
            )

        if (
            not math.isfinite(self.size_bits)
            or self.size_bits <= 0.0
        ):
            raise ValueError(
                "size_bits must be finite and positive."
            )

        if (
            not math.isfinite(self.remaining_bits)
            or self.remaining_bits <= 0.0
            or self.remaining_bits > self.size_bits
        ):
            raise ValueError(
                "remaining_bits must lie in "
                "(0, size_bits]."
            )

        if (
            self.deadline_tti is not None
            and self.deadline_tti
            < self.arrival_tti
        ):
            raise ValueError(
                "deadline_tti cannot precede arrival."
            )


@dataclass(frozen=True)
class PacketServiceResult:
    delivered_bits: float

    completed_packets: int

    completed_delay_ttis: tuple[int, ...]

    deadline_misses: int

    dropped_packets: int

    dropped_bits: float

    queue_bits_after: float

    queue_packets_after: int

    hol_delay_ttis_after: int | None


class PacketFIFOQueue:
    """
    FIFO packet queue for future QoS-aware evaluation.

    This is intentionally separate from the current
    aggregate-bit TrafficBufferManager so existing
    reproduction results remain unchanged.

    It gives us the primitives needed for:

        packet latency,
        HoL delay,
        deadline misses,
        finite-buffer drops,
        service-class metadata.
    """

    def __init__(
        self,
        *,
        max_queue_bits: float | None = None,
        drop_expired_packets: bool = True,
    ) -> None:

        if max_queue_bits is not None:
            if (
                not math.isfinite(max_queue_bits)
                or max_queue_bits <= 0.0
            ):
                raise ValueError(
                    "max_queue_bits must be finite "
                    "and positive."
                )

        self.max_queue_bits = max_queue_bits

        self.drop_expired_packets = (
            drop_expired_packets
        )

        self._packets: deque[Packet] = deque()

        self.total_dropped_packets = 0

        self.total_dropped_bits = 0.0

        self.total_deadline_misses = 0


    @property
    def queue_bits(
        self,
    ) -> float:

        return float(
            sum(
                packet.remaining_bits
                for packet in self._packets
            )
        )


    @property
    def num_packets(
        self,
    ) -> int:

        return len(self._packets)


    def hol_delay_ttis(
        self,
        *,
        tti_index: int,
    ) -> int | None:

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if not self._packets:
            return None

        return (
            tti_index
            - self._packets[0].arrival_tti
        )


    def enqueue(
        self,
        *,
        arrival_tti: int,
        size_bits: float,
        deadline_tti: int | None = None,
        traffic_class: str = "best_effort",
    ) -> bool:
        """
        Return True if admitted, False if dropped
        because the finite queue would overflow.
        """

        packet = Packet(
            arrival_tti=arrival_tti,
            size_bits=size_bits,
            remaining_bits=size_bits,
            deadline_tti=deadline_tti,
            traffic_class=traffic_class,
        )

        if (
            self.max_queue_bits is not None
            and (
                self.queue_bits
                + size_bits
                > self.max_queue_bits
            )
        ):
            self.total_dropped_packets += 1

            self.total_dropped_bits += (
                size_bits
            )

            return False

        self._packets.append(packet)

        return True


    def _drop_expired(
        self,
        *,
        tti_index: int,
    ) -> tuple[int, float]:

        if not self.drop_expired_packets:
            return 0, 0.0

        kept: deque[Packet] = deque()

        missed = 0
        dropped_bits = 0.0

        while self._packets:

            packet = self._packets.popleft()

            expired = (
                packet.deadline_tti is not None
                and tti_index
                > packet.deadline_tti
            )

            if expired:
                missed += 1

                dropped_bits += (
                    packet.remaining_bits
                )

            else:
                kept.append(packet)

        self._packets = kept

        self.total_deadline_misses += missed

        self.total_dropped_packets += missed

        self.total_dropped_bits += dropped_bits

        return missed, dropped_bits


    def serve(
        self,
        *,
        tti_index: int,
        capacity_bits: float,
    ) -> PacketServiceResult:

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if (
            not math.isfinite(capacity_bits)
            or capacity_bits < 0.0
        ):
            raise ValueError(
                "capacity_bits must be finite "
                "and non-negative."
            )

        (
            deadline_misses,
            expired_bits,
        ) = self._drop_expired(
            tti_index=tti_index
        )

        remaining_capacity = (
            capacity_bits
        )

        delivered_bits = 0.0

        completed_delays: list[int] = []

        completed_packets = 0

        while (
            remaining_capacity > 0.0
            and self._packets
        ):

            packet = self._packets[0]

            amount = min(
                remaining_capacity,
                packet.remaining_bits,
            )

            packet.remaining_bits -= amount

            remaining_capacity -= amount

            delivered_bits += amount

            if packet.remaining_bits <= 1.0e-9:

                completed_packets += 1

                completed_delays.append(
                    tti_index
                    - packet.arrival_tti
                )

                self._packets.popleft()

        return PacketServiceResult(
            delivered_bits=delivered_bits,

            completed_packets=(
                completed_packets
            ),

            completed_delay_ttis=tuple(
                completed_delays
            ),

            deadline_misses=(
                deadline_misses
            ),

            dropped_packets=(
                deadline_misses
            ),

            dropped_bits=(
                expired_bits
            ),

            queue_bits_after=(
                self.queue_bits
            ),

            queue_packets_after=(
                self.num_packets
            ),

            hol_delay_ttis_after=(
                self.hol_delay_ttis(
                    tti_index=tti_index
                )
            ),
        )
