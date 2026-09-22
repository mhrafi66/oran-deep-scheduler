from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from sionna.phy.channel import (
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)

from oran_scheduler.simulator.channel import (
    ChannelConfig,
    attach_topology_to_channel,
    create_channel_arrays,
    create_uma_channel_model,
    set_channel_seed,
)
from oran_scheduler.simulator.topology import (
    TopologyData,
)


@dataclass(frozen=True)
class TemporalChannelWindow:
    """
    One velocity-driven Sionna channel realization
    sampled at scheduler-TTI spacing.

    h_freq shape:

        [
            batch,
            UE,
            RX antenna,
            BS,
            TX antenna,
            TTI sample,
            subcarrier,
        ]

    IMPORTANT:
        Time samples inside ONE window belong to one
        common Sionna realization and are evolved
        using the UE velocities supplied in the
        topology.

        This is materially different from the
        existing implementation where every TTI gets
        a newly seeded independent channel.

    IMPORTANT LIMITATION:
        This object does not by itself move UE
        geometry between samples.

        It models short-window velocity/Doppler
        evolution around the attached topology.
    """

    h_freq: torch.Tensor

    seed: int

    tti_duration_s: float


    def __post_init__(
        self,
    ) -> None:

        if self.h_freq.ndim != 7:
            raise ValueError(
                "Temporal h_freq must have shape "
                "[batch, UE, RX, BS, TX, time, SC]."
            )

        if (
            not torch.is_complex(
                self.h_freq
            )
        ):
            raise ValueError(
                "Temporal h_freq must be complex."
            )

        if (
            not math.isfinite(
                self.tti_duration_s
            )
            or self.tti_duration_s <= 0.0
        ):
            raise ValueError(
                "tti_duration_s must be finite "
                "and positive."
            )


    @property
    def num_ttis(
        self,
    ) -> int:
        return int(
            self.h_freq.shape[-2]
        )


    @property
    def num_subcarriers(
        self,
    ) -> int:
        return int(
            self.h_freq.shape[-1]
        )


    @property
    def sampling_frequency_hz(
        self,
    ) -> float:
        return (
            1.0
            / self.tti_duration_s
        )


    def tti_slice(
        self,
        tti_offset: int,
    ) -> torch.Tensor:
        """
        Return one TTI in the SAME tensor shape used
        by the existing PHY:

            [..., 1 OFDM/time symbol, subcarrier]
        """

        if not (
            0
            <= tti_offset
            < self.num_ttis
        ):
            raise ValueError(
                "tti_offset outside temporal "
                "channel window."
            )

        return self.h_freq[
            ...,
            tti_offset:tti_offset + 1,
            :,
        ]


def normalized_temporal_correlation(
    h_freq: torch.Tensor,
    *,
    reference_tti: int = 0,
) -> torch.Tensor:
    """
    Compute normalized complex channel correlation
    against one reference TTI.

    All dimensions except the temporal dimension are
    flattened.

    Output:
        [num_ttis]

    Each value lies approximately in [0, 1]:

        1:
            same channel direction up to a complex
            scalar phase.

        0:
            orthogonal after flattening.

    This is a diagnostic, not a scheduler KPI.
    """

    if h_freq.ndim < 3:
        raise ValueError(
            "h_freq must include a temporal axis "
            "at dimension -2."
        )

    num_ttis = int(
        h_freq.shape[-2]
    )

    if not (
        0
        <= reference_tti
        < num_ttis
    ):
        raise ValueError(
            "reference_tti outside temporal axis."
        )

    time_first = (
        h_freq
        .movedim(
            -2,
            0,
        )
    )

    flat = (
        time_first
        .reshape(
            num_ttis,
            -1,
        )
    )

    reference = flat[
        reference_tti
    ]

    numerator = torch.abs(
        torch.sum(
            torch.conj(reference)[None, :]
            * flat,
            dim=1,
        )
    )

    reference_norm = (
        torch.linalg.vector_norm(
            reference
        )
    )

    sample_norm = (
        torch.linalg.vector_norm(
            flat,
            dim=1,
        )
    )

    denominator = (
        reference_norm
        * sample_norm
    )

    zero = torch.zeros_like(
        numerator
    )

    correlation = torch.where(
        denominator > 0.0,
        numerator / denominator,
        zero,
    )

    return torch.clamp(
        correlation.real,
        min=0.0,
        max=1.0,
    )


class VelocityWindowFrequencyChannelRuntime:
    """
    Reusable Sionna runtime for short temporally
    correlated UMa channel windows.

    Existing independent runtime:

        seed(t)
            ->
        one independent H_t

    This runtime:

        seed(window)
            ->
        one Sionna CIR realization
            ->
        H_0, H_1, ..., H_(W-1)

    where samples are separated by tti_duration_s.

    UE velocities passed to UMa produce Doppler/time
    evolution inside the window.

    SCIENTIFIC LABEL:
        "velocity-driven temporal Sionna window"

    Do NOT yet call this full spatially consistent
    mobility because large-scale geometry and random
    channel state are not continuously transported
    across window boundaries.
    """

    def __init__(
        self,
        *,
        config: ChannelConfig,
        normalize_channel: bool = False,
    ) -> None:

        self.config = config

        self.normalize_channel = (
            normalize_channel
        )

        (
            self.bs_array,
            self.ut_array,
        ) = create_channel_arrays(
            config
        )

        self.channel_model = (
            create_uma_channel_model(
                config=config,
                bs_array=self.bs_array,
                ut_array=self.ut_array,
            )
        )

        self.frequencies = (
            subcarrier_frequencies(
                config.num_subcarriers,
                config.subcarrier_spacing_hz,
                precision=config.precision,
                device=config.device,
            )
        )

        self._topology_shape: (
            tuple[int, int, int]
            | None
        ) = None

        self._num_generations = 0

        self._num_topology_resets = 0


    @property
    def num_generations(
        self,
    ) -> int:
        return self._num_generations


    @property
    def num_topology_resets(
        self,
    ) -> int:
        return (
            self
            ._num_topology_resets
        )


    def generate_tti_slice(
        self,
        *,
        topology: TopologyData,
        seed: int,
        batch_size: int,
        num_ttis: int,
        tti_duration_s: float,
        tti_offset: int,
    ) -> torch.Tensor:
        """
        Generate one frequency-domain TTI from a
        temporally correlated Sionna CIR window.

        The important ordering is:

            generate W correlated CIR samples
                ->
            select ONE CIR temporal sample
                ->
            convert only that sample to OFDM H.

        This avoids materializing the full
        [W x subcarrier] paper-MIMO frequency tensor.

        Returned shape:

            [
                batch,
                UE,
                RX antenna,
                BS,
                TX antenna,
                1 time sample,
                subcarrier,
            ]

        The complete CIR window is still generated,
        so samples requested with the same topology,
        seed, num_ttis, and TTI spacing belong to the
        same velocity-driven Sionna realization.

        NOTE:
            Production uses normalize_channel=False.

            If normalization is enabled, conversion
            normalization is applied to the selected
            singleton temporal sample rather than to
            the complete frequency-domain window.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if num_ttis <= 0:
            raise ValueError(
                "num_ttis must be positive."
            )

        if not (
            0
            <= tti_offset
            < num_ttis
        ):
            raise ValueError(
                "tti_offset outside requested "
                "temporal window."
            )

        if (
            not math.isfinite(
                tti_duration_s
            )
            or tti_duration_s <= 0.0
        ):
            raise ValueError(
                "tti_duration_s must be finite "
                "and positive."
            )

        topology_batch_size = int(
            topology.ut_loc.shape[0]
        )

        num_ues = int(
            topology.ut_loc.shape[1]
        )

        num_bs = int(
            topology.bs_loc.shape[1]
        )

        if (
            batch_size
            != topology_batch_size
        ):
            raise ValueError(
                "batch_size does not match "
                "topology."
            )

        topology_shape = (
            topology_batch_size,
            num_ues,
            num_bs,
        )

        if (
            self._topology_shape
            is not None
            and topology_shape
            != self._topology_shape
        ):
            self.channel_model.reset_topology()

            self._num_topology_resets += 1

        set_channel_seed(
            seed
        )

        attach_topology_to_channel(
            channel_model=(
                self.channel_model
            ),
            topology=topology,
        )

        self._topology_shape = (
            topology_shape
        )

        sampling_frequency_hz = (
            1.0
            / tti_duration_s
        )

        #
        # Generate the SAME W-sample temporal CIR
        # realization used by generate_window().
        #
        channel_impulse_response = (
            self.channel_model(
                batch_size,
                num_ttis,
                sampling_frequency_hz,
            )
        )

        if (
            not isinstance(
                channel_impulse_response,
                tuple,
            )
            or len(
                channel_impulse_response
            ) < 2
        ):
            raise RuntimeError(
                "Unexpected Sionna system-level "
                "channel return value."
            )

        path_coefficients = (
            channel_impulse_response[0]
        )

        path_delays = (
            channel_impulse_response[1]
        )

        #
        # Sionna CIR coefficients place temporal
        # samples on the final coefficient axis:
        #
        # [..., path, time]
        #
        # Fail loudly if the installed Sionna
        # version does not match that contract.
        #
        if (
            path_coefficients.ndim < 1
            or int(
                path_coefficients.shape[-1]
            )
            != num_ttis
        ):
            raise RuntimeError(
                "Unexpected Sionna CIR temporal "
                "axis. Expected final coefficient "
                f"axis size {num_ttis}, got shape "
                f"{tuple(path_coefficients.shape)}."
            )

        #
        # CRITICAL MEMORY FIX
        # -------------------
        #
        # Slice BEFORE subcarrier expansion.
        #
        # clone() is intentional:
        # it gives the singleton sample independent
        # storage so the full W-sample coefficient
        # tensor can be released before the expensive
        # OFDM conversion.
        #
        selected_path_coefficients = (
            path_coefficients[
                ...,
                tti_offset:tti_offset + 1,
            ]
            .clone()
        )

        del path_coefficients
        del channel_impulse_response

        h_freq = cir_to_ofdm_channel(
            self.frequencies,
            selected_path_coefficients,
            path_delays,
            normalize=(
                self.normalize_channel
            ),
        )

        del selected_path_coefficients
        del path_delays

        if (
            h_freq.ndim != 7
        ):
            raise RuntimeError(
                "Unexpected singleton temporal "
                "frequency-channel rank."
            )

        if (
            int(h_freq.shape[-2])
            != 1
        ):
            raise RuntimeError(
                "Memory-safe temporal generation "
                "must return exactly one time "
                "sample."
            )

        if (
            int(h_freq.shape[-1])
            != self.config.num_subcarriers
        ):
            raise RuntimeError(
                "Sionna frequency axis does not "
                "match ChannelConfig."
            )

        if (
            not torch.isfinite(
                h_freq.real
            ).all()
            or not torch.isfinite(
                h_freq.imag
            ).all()
        ):
            raise RuntimeError(
                "Generated temporal frequency "
                "channel contains non-finite values."
            )

        self._num_generations += 1

        return h_freq


    def generate_window(
        self,
        *,
        topology: TopologyData,
        seed: int,
        batch_size: int,
        num_ttis: int,
        tti_duration_s: float,
    ) -> TemporalChannelWindow:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if num_ttis <= 0:
            raise ValueError(
                "num_ttis must be positive."
            )

        if (
            not math.isfinite(
                tti_duration_s
            )
            or tti_duration_s <= 0.0
        ):
            raise ValueError(
                "tti_duration_s must be finite "
                "and positive."
            )

        topology_batch_size = int(
            topology.ut_loc.shape[0]
        )

        num_ues = int(
            topology.ut_loc.shape[1]
        )

        num_bs = int(
            topology.bs_loc.shape[1]
        )

        if (
            batch_size
            != topology_batch_size
        ):
            raise ValueError(
                "batch_size does not match "
                "topology."
            )

        topology_shape = (
            topology_batch_size,
            num_ues,
            num_bs,
        )

        if (
            self._topology_shape
            is not None
            and topology_shape
            != self._topology_shape
        ):
            self.channel_model.reset_topology()

            self._num_topology_resets += 1

        set_channel_seed(
            seed
        )

        attach_topology_to_channel(
            channel_model=(
                self.channel_model
            ),
            topology=topology,
        )

        self._topology_shape = (
            topology_shape
        )

        sampling_frequency_hz = (
            1.0
            / tti_duration_s
        )

        #
        # System-level Sionna channel models are
        # called by GenerateOFDMChannel as:
        #
        #     model(
        #         batch_size,
        #         num_time_samples,
        #         sampling_frequency,
        #     )
        #
        # We call the same model directly so our time
        # spacing can equal the simulator TTI rather
        # than the OFDM-symbol duration.
        #
        channel_impulse_response = (
            self.channel_model(
                batch_size,
                num_ttis,
                sampling_frequency_hz,
            )
        )

        if (
            not isinstance(
                channel_impulse_response,
                tuple,
            )
            or len(
                channel_impulse_response
            ) < 2
        ):
            raise RuntimeError(
                "Unexpected Sionna system-level "
                "channel return value."
            )

        path_coefficients = (
            channel_impulse_response[0]
        )

        path_delays = (
            channel_impulse_response[1]
        )

        h_freq = cir_to_ofdm_channel(
            self.frequencies,
            path_coefficients,
            path_delays,
            normalize=(
                self.normalize_channel
            ),
        )

        if (
            h_freq.ndim != 7
        ):
            raise RuntimeError(
                "Unexpected temporal frequency "
                "channel rank."
            )

        if (
            int(h_freq.shape[-2])
            != num_ttis
        ):
            raise RuntimeError(
                "Sionna temporal axis does not "
                "match requested num_ttis."
            )

        if (
            int(h_freq.shape[-1])
            != self.config.num_subcarriers
        ):
            raise RuntimeError(
                "Sionna frequency axis does not "
                "match ChannelConfig."
            )

        if (
            not torch.isfinite(
                h_freq.real
            ).all()
            or not torch.isfinite(
                h_freq.imag
            ).all()
        ):
            raise RuntimeError(
                "Temporal channel contains "
                "non-finite values."
            )

        self._num_generations += 1

        return TemporalChannelWindow(
            h_freq=h_freq,

            seed=seed,

            tti_duration_s=(
                tti_duration_s
            ),
        )
