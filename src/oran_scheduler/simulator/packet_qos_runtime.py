from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import csv
from pathlib import Path

import torch

from oran_scheduler.simulator.traffic import (
    FTP3TrafficConfig,
    TrafficBufferManager,
    TrafficServiceResult,
    TrafficTTIStart,
)


@dataclass
class _TrackedPacket:
    arrival_tti: int
    remaining_bits: float
    deadline_missed: bool = False


@dataclass(frozen=True)
class PacketQoSTTISummary:
    """
    Packet-level QoS diagnostics for one cell/TTI.

    IMPORTANT:
        This tracker does NOT change the actual
        TrafficBufferManager queue.

        It is a shadow FIFO reconstruction driven by
        the exact packet-arrival trace and exact
        delivered bits.

        Therefore existing scheduler behavior remains
        unchanged.
    """

    tti_index: int
    cell_index: int

    completed_packets: int

    new_deadline_misses: int

    cumulative_completed_packets: int

    cumulative_deadline_misses: int

    mean_completion_delay_ttis: float

    p50_completion_delay_ttis: float

    p95_completion_delay_ttis: float

    p99_completion_delay_ttis: float

    max_completion_delay_ttis: float

    queued_packets: int

    shadow_queue_bits: float

    mean_hol_delay_ttis: float

    p95_hol_delay_ttis: float

    max_hol_delay_ttis: float


def _percentile(
    values: list[float],
    q: float,
) -> float:

    if not values:
        return 0.0

    tensor = torch.tensor(
        values,
        dtype=torch.float64,
    )

    return float(
        torch.quantile(
            tensor,
            q,
        ).item()
    )


class PacketQoSTrafficBufferManager(
    TrafficBufferManager
):
    """
    TrafficBufferManager with a shadow packet FIFO.

    Existing traffic semantics remain IDENTICAL.

    Added diagnostics:
        - packet completion delay,
        - HoL delay,
        - deadline violations,
        - number of queued packets.

    Same-TTI completion convention:
        delay = 1 TTI.

    Example:

        packet arrives at start of TTI 10
        completed during TTI 10

        completion delay = 1 TTI.

    Deadline semantics:
        packet_qos_deadline_ttis = D

        miss occurs once elapsed delay exceeds D.
    """

    def __init__(
        self,
        *,
        full_buffer_mask: torch.Tensor,
        ftp3_config: FTP3TrafficConfig,
        full_buffer_state_bits: float,
        initial_ftp_buffer_bits: (
            torch.Tensor | None
        ) = None,
        seed: int | None = None,

        packet_qos_enabled: bool = False,

        packet_qos_csv_path: (
            Path | str | None
        ) = None,

        packet_qos_cell_index: int = 0,

        packet_qos_deadline_ttis: (
            int | None
        ) = None,
    ) -> None:

        super().__init__(
            full_buffer_mask=full_buffer_mask,
            ftp3_config=ftp3_config,
            full_buffer_state_bits=(
                full_buffer_state_bits
            ),
            initial_ftp_buffer_bits=(
                initial_ftp_buffer_bits
            ),
            seed=seed,
        )

        if packet_qos_cell_index < 0:
            raise ValueError(
                "packet_qos_cell_index must be "
                "non-negative."
            )

        if (
            packet_qos_deadline_ttis
            is not None
            and packet_qos_deadline_ttis <= 0
        ):
            raise ValueError(
                "packet_qos_deadline_ttis must be "
                "positive when provided."
            )

        self.packet_qos_enabled = (
            bool(packet_qos_enabled)
        )

        self.packet_qos_cell_index = (
            int(packet_qos_cell_index)
        )

        self.packet_qos_deadline_ttis = (
            packet_qos_deadline_ttis
        )

        self.packet_qos_csv_path = (
            None
            if packet_qos_csv_path is None
            else Path(
                packet_qos_csv_path
            )
        )

        self._packet_queues = [
            deque()
            for _ in range(
                int(full_buffer_mask.numel())
            )
        ]

        self._qos_full_buffer_mask = (
            full_buffer_mask
            .detach()
            .cpu()
            .bool()
        )

        self._cumulative_completed = 0

        self._cumulative_deadline_misses = 0

        self._last_qos_summary: (
            PacketQoSTTISummary | None
        ) = None

        if (
            self.packet_qos_enabled
            and self.packet_qos_csv_path
            is not None
        ):
            self._initialize_csv()


    @property
    def last_packet_qos_summary(
        self,
    ) -> PacketQoSTTISummary | None:

        return self._last_qos_summary


    def _initialize_csv(
        self,
    ) -> None:

        assert (
            self.packet_qos_csv_path
            is not None
        )

        path = self.packet_qos_csv_path

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if path.exists():
            return

        with path.open(
            "w",
            newline="",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "tti",
                    "cell",
                    "completed_packets",
                    "new_deadline_misses",
                    "cumulative_completed_packets",
                    "cumulative_deadline_misses",
                    "mean_completion_delay_ttis",
                    "p50_completion_delay_ttis",
                    "p95_completion_delay_ttis",
                    "p99_completion_delay_ttis",
                    "max_completion_delay_ttis",
                    "queued_packets",
                    "shadow_queue_bits",
                    "mean_hol_delay_ttis",
                    "p95_hol_delay_ttis",
                    "max_hol_delay_ttis",
                ],
            )

            writer.writeheader()


    def _append_csv(
        self,
        summary: PacketQoSTTISummary,
    ) -> None:

        if self.packet_qos_csv_path is None:
            return

        with self.packet_qos_csv_path.open(
            "a",
            newline="",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "tti",
                    "cell",
                    "completed_packets",
                    "new_deadline_misses",
                    "cumulative_completed_packets",
                    "cumulative_deadline_misses",
                    "mean_completion_delay_ttis",
                    "p50_completion_delay_ttis",
                    "p95_completion_delay_ttis",
                    "p99_completion_delay_ttis",
                    "max_completion_delay_ttis",
                    "queued_packets",
                    "shadow_queue_bits",
                    "mean_hol_delay_ttis",
                    "p95_hol_delay_ttis",
                    "max_hol_delay_ttis",
                ],
            )

            writer.writerow(
                {
                    "tti": summary.tti_index,
                    "cell": summary.cell_index,
                    "completed_packets": (
                        summary.completed_packets
                    ),
                    "new_deadline_misses": (
                        summary.new_deadline_misses
                    ),
                    "cumulative_completed_packets": (
                        summary
                        .cumulative_completed_packets
                    ),
                    "cumulative_deadline_misses": (
                        summary
                        .cumulative_deadline_misses
                    ),
                    "mean_completion_delay_ttis": (
                        summary
                        .mean_completion_delay_ttis
                    ),
                    "p50_completion_delay_ttis": (
                        summary
                        .p50_completion_delay_ttis
                    ),
                    "p95_completion_delay_ttis": (
                        summary
                        .p95_completion_delay_ttis
                    ),
                    "p99_completion_delay_ttis": (
                        summary
                        .p99_completion_delay_ttis
                    ),
                    "max_completion_delay_ttis": (
                        summary
                        .max_completion_delay_ttis
                    ),
                    "queued_packets": (
                        summary.queued_packets
                    ),
                    "shadow_queue_bits": (
                        summary.shadow_queue_bits
                    ),
                    "mean_hol_delay_ttis": (
                        summary.mean_hol_delay_ttis
                    ),
                    "p95_hol_delay_ttis": (
                        summary.p95_hol_delay_ttis
                    ),
                    "max_hol_delay_ttis": (
                        summary.max_hol_delay_ttis
                    ),
                }
            )


    def begin_tti(
        self,
        *,
        tti_index: int,
        packet_arrivals: (
            torch.Tensor | None
        ) = None,
    ) -> TrafficTTIStart:

        result = super().begin_tti(
            tti_index=tti_index,
            packet_arrivals=packet_arrivals,
        )

        if not self.packet_qos_enabled:
            return result

        arrivals = (
            result.packet_arrivals
            .detach()
            .cpu()
        )

        packet_size_bits = float(
            self
            .ftp3_config
            .packet_size_bits
        )

        for ue_index in range(
            int(arrivals.numel())
        ):

            if bool(
                self._qos_full_buffer_mask[
                    ue_index
                ].item()
            ):
                continue

            count = int(
                arrivals[
                    ue_index
                ].item()
            )

            for _ in range(
                count
            ):
                self._packet_queues[
                    ue_index
                ].append(
                    _TrackedPacket(
                        arrival_tti=(
                            tti_index
                        ),
                        remaining_bits=(
                            packet_size_bits
                        ),
                    )
                )

        return result


    def apply_service(
        self,
        *,
        offered_service_capacity_bps: (
            torch.Tensor
        ),
    ) -> TrafficServiceResult:

        result = super().apply_service(
            offered_service_capacity_bps=(
                offered_service_capacity_bps
            )
        )

        if not self.packet_qos_enabled:
            return result

        tti_index = int(
            result.tti_index
        )

        delivered = (
            result.delivered_bits
            .detach()
            .cpu()
        )

        completed_delays: list[
            float
        ] = []

        new_deadline_misses = 0

        for ue_index, queue in enumerate(
            self._packet_queues
        ):

            if bool(
                self._qos_full_buffer_mask[
                    ue_index
                ].item()
            ):
                continue

            #
            # Mark newly violated deadlines BEFORE
            # applying this TTI's service.
            #
            if (
                self.packet_qos_deadline_ttis
                is not None
            ):
                for packet in queue:

                    elapsed_ttis = (
                        tti_index
                        - packet.arrival_tti
                        + 1
                    )

                    if (
                        elapsed_ttis
                        > self
                        .packet_qos_deadline_ttis
                        and not packet.deadline_missed
                    ):
                        packet.deadline_missed = True

                        new_deadline_misses += 1

            remaining_service = float(
                delivered[
                    ue_index
                ].item()
            )

            while (
                remaining_service > 1.0e-9
                and queue
            ):

                packet = queue[0]

                used = min(
                    remaining_service,
                    packet.remaining_bits,
                )

                packet.remaining_bits -= used

                remaining_service -= used

                if (
                    packet.remaining_bits
                    <= 1.0e-6
                ):
                    delay = (
                        tti_index
                        - packet.arrival_tti
                        + 1
                    )

                    completed_delays.append(
                        float(delay)
                    )

                    queue.popleft()

                    self._cumulative_completed += 1

        self._cumulative_deadline_misses += (
            new_deadline_misses
        )

        shadow_bits_by_ue = []

        hol_delays = []

        queued_packets = 0

        for ue_index, queue in enumerate(
            self._packet_queues
        ):

            if bool(
                self._qos_full_buffer_mask[
                    ue_index
                ].item()
            ):
                shadow_bits_by_ue.append(
                    0.0
                )
                continue

            queue_bits = float(
                sum(
                    packet.remaining_bits
                    for packet in queue
                )
            )

            shadow_bits_by_ue.append(
                queue_bits
            )

            queued_packets += len(
                queue
            )

            if queue:
                hol_delays.append(
                    float(
                        tti_index
                        - queue[0].arrival_tti
                        + 1
                    )
                )

        #
        # Scientific consistency check:
        #
        # Shadow packet FIFO must reconstruct the
        # actual aggregate FTP queue.
        #
        shadow_tensor = torch.tensor(
            shadow_bits_by_ue,
            dtype=torch.float64,
        )

        real_after = (
            result
            .buffer_after_service_bits
            .detach()
            .cpu()
            .to(
                dtype=torch.float64
            )
        )

        ftp_mask = (
            ~self._qos_full_buffer_mask
        )

        if not torch.allclose(
            shadow_tensor[
                ftp_mask
            ],
            real_after[
                ftp_mask
            ],
            rtol=1.0e-5,
            atol=1.0e-2,
        ):
            max_error = float(
                torch.max(
                    torch.abs(
                        shadow_tensor[
                            ftp_mask
                        ]
                        - real_after[
                            ftp_mask
                        ]
                    )
                ).item()
            )

            raise RuntimeError(
                "Packet QoS shadow FIFO diverged "
                "from aggregate FTP queue. "
                f"Max error={max_error} bits."
            )

        mean_delay = (
            float(
                sum(completed_delays)
                / len(completed_delays)
            )
            if completed_delays
            else 0.0
        )

        mean_hol = (
            float(
                sum(hol_delays)
                / len(hol_delays)
            )
            if hol_delays
            else 0.0
        )

        summary = PacketQoSTTISummary(
            tti_index=tti_index,

            cell_index=(
                self.packet_qos_cell_index
            ),

            completed_packets=len(
                completed_delays
            ),

            new_deadline_misses=(
                new_deadline_misses
            ),

            cumulative_completed_packets=(
                self._cumulative_completed
            ),

            cumulative_deadline_misses=(
                self
                ._cumulative_deadline_misses
            ),

            mean_completion_delay_ttis=(
                mean_delay
            ),

            p50_completion_delay_ttis=(
                _percentile(
                    completed_delays,
                    0.50,
                )
            ),

            p95_completion_delay_ttis=(
                _percentile(
                    completed_delays,
                    0.95,
                )
            ),

            p99_completion_delay_ttis=(
                _percentile(
                    completed_delays,
                    0.99,
                )
            ),

            max_completion_delay_ttis=(
                max(
                    completed_delays,
                    default=0.0,
                )
            ),

            queued_packets=(
                queued_packets
            ),

            shadow_queue_bits=float(
                shadow_tensor.sum().item()
            ),

            mean_hol_delay_ttis=(
                mean_hol
            ),

            p95_hol_delay_ttis=(
                _percentile(
                    hol_delays,
                    0.95,
                )
            ),

            max_hol_delay_ttis=max(
                hol_delays,
                default=0.0,
            ),
        )

        self._last_qos_summary = (
            summary
        )

        self._append_csv(
            summary
        )

        return result
