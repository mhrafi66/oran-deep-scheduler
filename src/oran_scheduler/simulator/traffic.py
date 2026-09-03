from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class FTP3TrafficConfig:
    """
    FTP Model 3 traffic parameters.

    PAPER-SPECIFIED:
        packet/file size and arrival rate depend on
        training vs evaluation configuration.

    3GPP FTP3:
        arrivals for a UE follow a Poisson process.

    OPEN-REPRODUCTION INPUT:
        tti_duration_s is supplied explicitly rather
        than silently assuming a particular simulator
        TTI implementation.
    """

    packet_size_bytes: int

    packet_arrival_rate_per_s: float

    tti_duration_s: float

    def __post_init__(self) -> None:
        if self.packet_size_bytes <= 0:
            raise ValueError(
                "packet_size_bytes must be positive."
            )

        if (
            not math.isfinite(
                self.packet_arrival_rate_per_s
            )
            or self.packet_arrival_rate_per_s < 0.0
        ):
            raise ValueError(
                "packet_arrival_rate_per_s must be "
                "non-negative and finite."
            )

        if (
            not math.isfinite(
                self.tti_duration_s
            )
            or self.tti_duration_s <= 0.0
        ):
            raise ValueError(
                "tti_duration_s must be positive "
                "and finite."
            )

    @property
    def packet_size_bits(
        self,
    ) -> float:
        return float(
            self.packet_size_bytes * 8
        )

    @property
    def expected_packets_per_tti(
        self,
    ) -> float:
        return (
            self.packet_arrival_rate_per_s
            * self.tti_duration_s
        )
    

def build_training_ftp3_config(
    *,
    tti_duration_s: float,
) -> FTP3TrafficConfig:
    """
    PAPER-SPECIFIED training FTP3 parameters:

        1.5 kB
        500 packets/s

    Decimal SI interpretation:
        1.5 kB -> 1500 bytes
    """

    return FTP3TrafficConfig(
        packet_size_bytes=1500,
        packet_arrival_rate_per_s=500.0,
        tti_duration_s=tti_duration_s,
    )


def build_evaluation_ftp3_config(
    *,
    tti_duration_s: float,
) -> FTP3TrafficConfig:
    """
    PAPER-SPECIFIED evaluation FTP3 parameters:

        0.5 MB
        20 packets/s

    Decimal SI interpretation:
        0.5 MB -> 500,000 bytes
    """

    return FTP3TrafficConfig(
        packet_size_bytes=500_000,
        packet_arrival_rate_per_s=20.0,
        tti_duration_s=tti_duration_s,
    )


@dataclass(frozen=True)
class TrafficTTIStart:
    """
    Traffic state after arrivals have been generated
    at the start of a TTI.

    All tensors have shape:

        [UE]

    buffer_for_scheduler_bits is the DL-buffer state
    that the scheduler should observe for this TTI.
    """

    tti_index: int

    packet_arrivals: torch.Tensor

    arrived_bits: torch.Tensor

    buffer_before_arrivals_bits: torch.Tensor

    buffer_for_scheduler_bits: torch.Tensor


@dataclass(frozen=True)
class TrafficServiceResult:
    """
    Result after applying PHY service capacity to
    the UE traffic queues.

    All tensors have shape:

        [UE]
    """

    tti_index: int

    offered_service_capacity_bps: torch.Tensor

    service_capacity_bits: torch.Tensor

    delivered_bits: torch.Tensor

    delivered_rate_bps: torch.Tensor

    unused_service_capacity_bits: torch.Tensor

    had_data_to_receive: torch.Tensor

    buffer_before_service_bits: torch.Tensor

    buffer_after_service_bits: torch.Tensor


class TrafficBufferManager:
    """
    Maintain per-UE downlink traffic queues across
    TTIs.

    FTP3 UEs:
        buffer_bits is a real finite backlog.

    Full-Buffer UEs:
        service is never queue-limited.

        Their exposed scheduler buffer is held at
        full_buffer_state_bits, which is a finite
        state representation of an effectively
        infinite backlog.

    IMPORTANT:
        full_buffer_state_bits is NOT a physical
        Full-Buffer queue size. It is the value
        presented to the scheduler's DL-buffer
        feature.
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
    ) -> None:
        if full_buffer_mask.ndim != 1:
            raise ValueError(
                "full_buffer_mask must have shape "
                "[UE]."
            )

        if full_buffer_mask.dtype != torch.bool:
            raise ValueError(
                "full_buffer_mask must use "
                "torch.bool."
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

        self.ftp3_config = ftp3_config

        self.full_buffer_state_bits = (
            float(
                full_buffer_state_bits
            )
        )

        self._full_buffer_mask = (
            full_buffer_mask
            .detach()
            .clone()
        )

        self._device = (
            full_buffer_mask.device
        )

        self._num_ues = int(
            full_buffer_mask.shape[0]
        )

        if initial_ftp_buffer_bits is None:
            initial_ftp_buffer_bits = torch.zeros(
                self._num_ues,
                dtype=torch.float32,
                device=self._device,
            )

        if tuple(
            initial_ftp_buffer_bits.shape
        ) != (
            self._num_ues,
        ):
            raise ValueError(
                "initial_ftp_buffer_bits must have "
                "shape [UE]."
            )

        if (
            initial_ftp_buffer_bits.device
            != self._device
        ):
            raise ValueError(
                "Initial FTP buffer and traffic mask "
                "must be on the same device."
            )

        if not torch.is_floating_point(
            initial_ftp_buffer_bits
        ):
            raise ValueError(
                "initial_ftp_buffer_bits must use a "
                "floating-point dtype."
            )

        if not torch.isfinite(
            initial_ftp_buffer_bits
        ).all():
            raise ValueError(
                "Initial FTP buffer contains "
                "non-finite values."
            )

        if torch.any(
            initial_ftp_buffer_bits < 0.0
        ):
            raise ValueError(
                "Initial FTP buffer cannot be "
                "negative."
            )

        self._buffer_bits = (
            initial_ftp_buffer_bits
            .detach()
            .clone()
        )

        #
        # Full Buffer queues are conceptually
        # infinite, but their scheduler-visible
        # buffer feature needs a finite value.
        #
        self._buffer_bits = torch.where(
            self._full_buffer_mask,
            torch.full_like(
                self._buffer_bits,
                self.full_buffer_state_bits,
            ),
            self._buffer_bits,
        )

        self._generator: (
            torch.Generator | None
        ) = None

        if seed is not None:
            self._generator = torch.Generator(
                device=self._device
            )

            self._generator.manual_seed(
                seed
            )

        self._active_tti_index: (
            int | None
        ) = None

        self._last_completed_tti_index: (
            int | None
        ) = None


    @property
    def current_buffer_bits(
        self,
    ) -> torch.Tensor:
        return (
            self._buffer_bits
            .detach()
            .clone()
        )


    @property
    def full_buffer_mask(
        self,
    ) -> torch.Tensor:
        return (
            self._full_buffer_mask
            .detach()
            .clone()
        )


    @property
    def has_active_tti(
        self,
    ) -> bool:
        return (
            self._active_tti_index
            is not None
        )


    def sample_packet_arrivals(
        self,
    ) -> torch.Tensor:
        """
        Sample one FTP3 arrival count per UE.

        FTP3:
            N ~ Poisson(lambda * TTI)

        Full Buffer:
            zero packet arrivals are recorded because
            FB traffic is represented separately as
            infinite backlog.

        Output:
            int64 [UE]
        """

        expected_arrivals = torch.full(
            (
                self._num_ues,
            ),
            (
                self
                .ftp3_config
                .expected_packets_per_tti
            ),
            dtype=torch.float32,
            device=self._device,
        )

        expected_arrivals = torch.where(
            self._full_buffer_mask,
            torch.zeros_like(
                expected_arrivals
            ),
            expected_arrivals,
        )

        sampled = torch.poisson(
            expected_arrivals,
            generator=self._generator,
        )

        return sampled.to(
            dtype=torch.long
        )


    def begin_tti(
        self,
        *,
        tti_index: int,
        packet_arrivals: (
            torch.Tensor | None
        ) = None,
    ) -> TrafficTTIStart:
        """
        Advance traffic generation to the beginning
        of one TTI.

        Normal simulation:
            packet_arrivals=None
            -> sample FTP3 Poisson arrivals.

        Deterministic testing/replay:
            packet_arrivals=[...]
            -> use the supplied arrival trace.

        Arrivals occur before the scheduler observes
        the DL buffer.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if self._active_tti_index is not None:
            raise RuntimeError(
                "The previous traffic TTI must be "
                "served before another TTI begins."
            )

        if (
            self._last_completed_tti_index
            is not None
        ):
            expected_tti = (
                self._last_completed_tti_index
                + 1
            )

            if tti_index != expected_tti:
                raise ValueError(
                    "Traffic TTIs must advance in "
                    "contiguous order."
                )

        if packet_arrivals is None:
            packet_arrivals = (
                self.sample_packet_arrivals()
            )

        if tuple(
            packet_arrivals.shape
        ) != (
            self._num_ues,
        ):
            raise ValueError(
                "packet_arrivals must have shape "
                "[UE]."
            )

        if (
            packet_arrivals.device
            != self._device
        ):
            raise ValueError(
                "packet_arrivals is on the wrong "
                "device."
            )

        if (
            torch.is_floating_point(
                packet_arrivals
            )
            or packet_arrivals.dtype
            == torch.bool
        ):
            raise ValueError(
                "packet_arrivals must use an "
                "integer dtype."
            )

        if torch.any(
            packet_arrivals < 0
        ):
            raise ValueError(
                "packet_arrivals cannot be negative."
            )

        if torch.any(
            packet_arrivals[
                self._full_buffer_mask
            ] != 0
        ):
            raise ValueError(
                "Full-Buffer UEs must not receive "
                "explicit FTP3 packet arrivals."
            )

        buffer_before = (
            self._buffer_bits
            .detach()
            .clone()
        )

        arrived_bits = (
            packet_arrivals.to(
                dtype=self._buffer_bits.dtype
            )
            * (
                self
                .ftp3_config
                .packet_size_bits
            )
        )

        #
        # FB rows remain at the finite state proxy.
        # FTP rows receive actual queued data.
        #
        ftp_buffer_after_arrivals = (
            self._buffer_bits
            + arrived_bits
        )

        self._buffer_bits = torch.where(
            self._full_buffer_mask,
            torch.full_like(
                self._buffer_bits,
                self.full_buffer_state_bits,
            ),
            ftp_buffer_after_arrivals,
        )

        self._active_tti_index = (
            tti_index
        )

        return TrafficTTIStart(
            tti_index=tti_index,
            packet_arrivals=(
                packet_arrivals
                .detach()
                .clone()
            ),
            arrived_bits=(
                arrived_bits
                .detach()
                .clone()
            ),
            buffer_before_arrivals_bits=(
                buffer_before
            ),
            buffer_for_scheduler_bits=(
                self._buffer_bits
                .detach()
                .clone()
            ),
        )


    def apply_service(
        self,
        *,
        offered_service_capacity_bps: (
            torch.Tensor
        ),
    ) -> TrafficServiceResult:
        """
        Convert PHY service capacity into actually
        delivered traffic.

        Full Buffer:
            delivered = PHY capacity.

        FTP3:
            delivered bits =
                min(
                    PHY capacity during TTI,
                    queued bits,
                )

        This distinction is critical:
            PHY capacity != delivered throughput
            whenever an FTP queue contains
            insufficient data.
        """

        if self._active_tti_index is None:
            raise RuntimeError(
                "begin_tti() must be called before "
                "applying service."
            )

        if tuple(
            offered_service_capacity_bps.shape
        ) != (
            self._num_ues,
        ):
            raise ValueError(
                "offered_service_capacity_bps must "
                "have shape [UE]."
            )

        if (
            offered_service_capacity_bps.device
            != self._device
        ):
            raise ValueError(
                "Service capacity is on the wrong "
                "device."
            )

        if not torch.is_floating_point(
            offered_service_capacity_bps
        ):
            raise ValueError(
                "Service capacity must use a "
                "floating-point dtype."
            )

        if not torch.isfinite(
            offered_service_capacity_bps
        ).all():
            raise ValueError(
                "Service capacity contains "
                "non-finite values."
            )

        if torch.any(
            offered_service_capacity_bps
            < 0.0
        ):
            raise ValueError(
                "Service capacity cannot be "
                "negative."
            )

        buffer_before_service = (
            self._buffer_bits
            .detach()
            .clone()
        )

        service_capacity_bits = (
            offered_service_capacity_bps
            * self.ftp3_config.tti_duration_s
        )

        ftp_delivered_bits = torch.minimum(
            service_capacity_bits,
            buffer_before_service,
        )

        delivered_bits = torch.where(
            self._full_buffer_mask,
            service_capacity_bits,
            ftp_delivered_bits,
        )

        delivered_rate_bps = (
            delivered_bits
            / self.ftp3_config.tti_duration_s
        )

        unused_service_capacity_bits = (
            service_capacity_bits
            - delivered_bits
        )

        had_data_to_receive = torch.where(
            self._full_buffer_mask,
            torch.ones(
                self._num_ues,
                dtype=torch.bool,
                device=self._device,
            ),
            buffer_before_service > 0.0,
        )

        ftp_buffer_after_service = (
            buffer_before_service
            - delivered_bits
        )

        #
        # Guard against tiny negative floating-point
        # residue.
        #
        ftp_buffer_after_service = (
            torch.clamp(
                ftp_buffer_after_service,
                min=0.0,
            )
        )

        self._buffer_bits = torch.where(
            self._full_buffer_mask,
            torch.full_like(
                self._buffer_bits,
                self.full_buffer_state_bits,
            ),
            ftp_buffer_after_service,
        )

        tti_index = self._active_tti_index

        self._last_completed_tti_index = (
            tti_index
        )

        self._active_tti_index = None

        return TrafficServiceResult(
            tti_index=tti_index,
            offered_service_capacity_bps=(
                offered_service_capacity_bps
                .detach()
                .clone()
            ),
            service_capacity_bits=(
                service_capacity_bits
                .detach()
                .clone()
            ),
            delivered_bits=(
                delivered_bits
                .detach()
                .clone()
            ),
            delivered_rate_bps=(
                delivered_rate_bps
                .detach()
                .clone()
            ),
            unused_service_capacity_bits=(
                unused_service_capacity_bits
                .detach()
                .clone()
            ),
            had_data_to_receive=(
                had_data_to_receive
                .detach()
                .clone()
            ),
            buffer_before_service_bits=(
                buffer_before_service
            ),
            buffer_after_service_bits=(
                self._buffer_bits
                .detach()
                .clone()
            ),
        )







