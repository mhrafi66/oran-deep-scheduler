from __future__ import annotations

from dataclasses import dataclass

import torch

from oran_scheduler.simulator.cell_association import (
    compute_channel_averaged_pathloss,
)

from oran_scheduler.simulator.channel import (
    ChannelConfig,
    generate_frequency_channel,
)

from oran_scheduler.simulator.mobility import (
    MobilityConfig,
    build_mobility_snapshot,
)

from oran_scheduler.simulator.topology import (
    TopologyConfig,
    TopologyData,
    subset_topology_ues,
)


@dataclass(frozen=True)
class SionnaHandoverMeasurementConfig:
    """
    Lightweight Sionna measurement configuration
    used ONLY for serving-cell / handover decisions.

    This is deliberately separate from the full
    paper-array scheduling PHY.

    The measurement path uses:

        moving UE geometry
            ->
        lightweight Sionna all-BS channel
            ->
        channel-averaged pathloss
            ->
        inverse pathloss as linear link strength

    OPEN-REPRODUCTION
    -----------------
    This is not claimed to reproduce a proprietary
    Nokia handover or exact 3GPP RSRP filtering
    implementation.

    Its purpose is to replace synthetic handover
    triggers with a physically generated,
    topology-dependent radio measurement while
    retaining the already tested hysteresis + TTT
    handover state machine.
    """

    carrier_frequency_hz: float = 4.0e9

    subcarrier_spacing_hz: float = 30.0e3

    num_rbs: int = 18

    subcarriers_per_rb: int = 12

    tti_duration_s: float = 0.5e-3

    channel_seed: int = 9100

    max_horizontal_displacement_m: (
        float | None
    ) = 20.0

    precision: str = "single"

    device: str = "cuda:0"


    def __post_init__(
        self,
    ) -> None:

        if self.carrier_frequency_hz <= 0.0:
            raise ValueError(
                "carrier_frequency_hz must "
                "be positive."
            )

        if self.subcarrier_spacing_hz <= 0.0:
            raise ValueError(
                "subcarrier_spacing_hz must "
                "be positive."
            )

        if self.num_rbs <= 0:
            raise ValueError(
                "num_rbs must be positive."
            )

        if self.subcarriers_per_rb <= 0:
            raise ValueError(
                "subcarriers_per_rb must "
                "be positive."
            )

        if self.tti_duration_s <= 0.0:
            raise ValueError(
                "tti_duration_s must be "
                "positive."
            )


class SionnaPathlossHandoverMeasurementProvider:
    """
    Produce persistent-UE/all-BS handover
    measurements from moving Sionna geometry.

    Output:

        [UE, BS]

    with larger values meaning stronger links,
    matching HandoverController's input convention.

    Internally:

        pathloss [UE, BS]
            ->
        link_strength = 1 / pathloss

    Since every gNB uses the same transmit power in
    the current reproduction, comparing inverse
    pathloss is equivalent to comparing average
    received-power ordering.

    IMPORTANT
    ---------
    `global_ue_indices` fixes row identity forever.

    Therefore reassociation does NOT reorder or
    resample handover measurements.

    The channel seed is intentionally fixed across
    TTIs. Geometry evolves, but we do not inject a
    completely unrelated random shadowing
    realization at each scheduler TTI.
    """

    def __init__(
        self,
        *,
        initial_topology: TopologyData,

        topology_config: TopologyConfig,

        global_ue_indices: torch.Tensor,

        config: (
            SionnaHandoverMeasurementConfig
            | None
        ) = None,
    ) -> None:

        if config is None:
            config = (
                SionnaHandoverMeasurementConfig()
            )

        if global_ue_indices.ndim != 1:
            raise ValueError(
                "global_ue_indices must have "
                "shape [UE]."
            )

        if global_ue_indices.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise TypeError(
                "global_ue_indices must use "
                "an integer dtype."
            )

        if global_ue_indices.numel() == 0:
            raise ValueError(
                "global_ue_indices cannot be "
                "empty."
            )

        if torch.any(
            global_ue_indices < 0
        ):
            raise ValueError(
                "global UE indices cannot "
                "be negative."
            )

        num_global_ues = int(
            initial_topology
            .ut_loc
            .shape[
                1
            ]
        )

        if torch.any(
            global_ue_indices
            >= num_global_ues
        ):
            raise ValueError(
                "global UE index exceeds the "
                "topology UE population."
            )

        if int(
            torch.unique(
                global_ue_indices
            ).numel()
        ) != int(
            global_ue_indices.numel()
        ):
            raise ValueError(
                "global_ue_indices must be unique."
            )

        self.initial_topology = (
            initial_topology
        )

        self.topology_config = (
            topology_config
        )

        self.global_ue_indices = (
            global_ue_indices
            .detach()
            .clone()
        )

        self.config = config

        self._last_tti_index: int | None = None

        self._last_link_power: (
            torch.Tensor | None
        ) = None

        self._last_pathloss: (
            torch.Tensor | None
        ) = None


    @property
    def last_pathloss(
        self,
    ) -> torch.Tensor | None:

        if self._last_pathloss is None:
            return None

        return (
            self
            ._last_pathloss
            .detach()
            .clone()
        )


    def __call__(
        self,
        tti_index: int,
    ) -> torch.Tensor:

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if (
            self._last_tti_index
            is not None
        ):

            if (
                tti_index
                < self._last_tti_index
            ):
                raise ValueError(
                    "Handover measurement time "
                    "cannot move backwards."
                )

            if (
                tti_index
                == self._last_tti_index
            ):
                assert (
                    self._last_link_power
                    is not None
                )

                return (
                    self
                    ._last_link_power
                    .detach()
                    .clone()
                )


        mobility = (
            build_mobility_snapshot(
                initial_topology=(
                    self.initial_topology
                ),

                config=MobilityConfig(
                    tti_duration_s=(
                        self
                        .config
                        .tti_duration_s
                    ),

                    trajectory_mode=(
                        "constant_velocity"
                    ),

                    #
                    # Mobility module itself still
                    # does not own association.
                    #
                    # The HandoverController owns
                    # dynamic serving-BS state.
                    #
                    association_mode="fixed",

                    max_horizontal_displacement_m=(
                        self
                        .config
                        .max_horizontal_displacement_m
                    ),
                ),

                tti_index=tti_index,
            )
        )


        measurement_topology = (
            subset_topology_ues(
                topology=(
                    mobility.topology
                ),

                global_ue_indices=(
                    self.global_ue_indices
                ),
            )
        )


        channel_config = ChannelConfig(
            carrier_frequency_hz=(
                self
                .config
                .carrier_frequency_hz
            ),

            subcarrier_spacing_hz=(
                self
                .config
                .subcarrier_spacing_hz
            ),

            num_rbs=(
                self
                .config
                .num_rbs
            ),

            subcarriers_per_rb=(
                self
                .config
                .subcarriers_per_rb
            ),

            num_ofdm_symbols=1,

            #
            # Handover measurement only.
            #
            # Full paper-array MIMO remains in the
            # scheduling PHY.
            #
            antenna_mode="sanity",

            direction="downlink",

            o2i_model="low",

            enable_pathloss=True,

            enable_shadow_fading=True,

            precision=(
                self.config.precision
            ),

            device=(
                self.config.device
            ),

            #
            # Fixed realization across TTIs:
            # geometry changes, UE identity does not.
            #
            seed=(
                self.config.channel_seed
            ),
        )


        channel = (
            generate_frequency_channel(
                topology=(
                    measurement_topology
                ),

                topology_config=(
                    self.topology_config
                ),

                channel_config=(
                    channel_config
                ),
            )
        )


        pathloss = (
            compute_channel_averaged_pathloss(
                h_freq=(
                    channel.h_freq
                ),

                precision=(
                    self.config.precision
                ),
            )[
                0
            ]
        )


        expected_shape = (
            int(
                self
                .global_ue_indices
                .numel()
            ),

            int(
                self
                .initial_topology
                .bs_loc
                .shape[
                    1
                ]
            ),
        )

        if tuple(
            pathloss.shape
        ) != expected_shape:
            raise RuntimeError(
                "Unexpected handover pathloss "
                "shape. Expected "
                f"{expected_shape}, got "
                f"{tuple(pathloss.shape)}."
            )

        if not torch.isfinite(
            pathloss
        ).all():
            raise RuntimeError(
                "Handover pathloss contains "
                "non-finite values."
            )

        if torch.any(
            pathloss <= 0.0
        ):
            raise RuntimeError(
                "Handover pathloss must be "
                "strictly positive."
            )


        link_power = (
            torch.reciprocal(
                pathloss
            )
        )


        if not torch.isfinite(
            link_power
        ).all():
            raise RuntimeError(
                "Handover link strength contains "
                "non-finite values."
            )


        self._last_tti_index = (
            tti_index
        )

        self._last_pathloss = (
            pathloss
            .detach()
            .clone()
        )

        self._last_link_power = (
            link_power
            .detach()
            .clone()
        )


        return (
            link_power
            .detach()
            .clone()
        )
