from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
import math

import torch

from oran_scheduler.rl.ppo_physical_score import (
    PPOPhysicalScoreInputs,
)
from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    PreparedOneLDSCellTTI,
)


InputProvider = Callable[
    [
        int,
        int,
    ],
    PPOTrainingTTIInputs,
]


@dataclass(frozen=True)
class PhysicalExecutionStressConfig:
    """
    Execution-time PHY shock.

    IMPORTANT:
        scheduler observation is NOT modified.

    Therefore these are post-CSI / post-measurement
    environment changes, not steady-state channel
    conditions.

    non_serving_interference_power_scale:
        Scale received POWER from every non-serving BS.

        1.0 = unchanged
        2.0 = +3.01 dB aggregate per interferer
        4.0 = +6.02 dB

    serving_signal_power_scale:
        Scale serving-link received power.

    failed_bs_index:
        Optional BS whose channel contribution becomes
        exactly zero at execution time.

        If the failed BS is the serving BS for a cell,
        that cell loses its serving signal.

        For other cells, that BS disappears as an
        interferer.
    """

    non_serving_interference_power_scale: float = 1.0

    serving_signal_power_scale: float = 1.0

    failed_bs_index: int | None = None

    failed_serving_power_scale: float = 1.0e-12


    def __post_init__(
        self,
    ) -> None:

        for name, value in (
            (
                "non_serving_interference_power_scale",
                self
                .non_serving_interference_power_scale,
            ),
            (
                "serving_signal_power_scale",
                self
                .serving_signal_power_scale,
            ),
        ):
            if (
                not math.isfinite(
                    value
                )
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and "
                    "non-negative."
                )

        if (
            self.failed_bs_index is not None
            and self.failed_bs_index < 0
        ):
            raise ValueError(
                "failed_bs_index cannot be negative."
            )

        if (
            not math.isfinite(
                self.failed_serving_power_scale
            )
            or self.failed_serving_power_scale <= 0.0
            or self.failed_serving_power_scale >= 1.0
        ):
            raise ValueError(
                "failed_serving_power_scale must be "
                "finite and strictly between 0 and 1."
            )


def stress_physical_inputs(
    *,
    inputs: PPOPhysicalScoreInputs,
    config: PhysicalExecutionStressConfig,
) -> PPOPhysicalScoreInputs:
    """
    Apply execution-time channel-amplitude scaling.

    Because received power is proportional to |H|^2:

        desired power factor p
            ->
        channel amplitude factor sqrt(p)
    """

    h_freq = (
        inputs
        .h_freq
    )

    if h_freq.ndim != 7:
        raise ValueError(
            "Expected h_freq shape "
            "[batch, UE, RX, BS, TX, symbol, subcarrier]."
        )

    num_bs = int(
        h_freq.shape[
            3
        ]
    )

    serving_bs = int(
        inputs
        .serving_cell_index
    )

    if not (
        0
        <= serving_bs
        < num_bs
    ):
        raise ValueError(
            "serving_cell_index is outside h_freq."
        )

    if (
        config.failed_bs_index is not None
        and config.failed_bs_index >= num_bs
    ):
        raise ValueError(
            "failed_bs_index is outside h_freq."
        )

    stressed = (
        h_freq
        .detach()
        .clone()
    )

    interferer_amp = math.sqrt(
        config
        .non_serving_interference_power_scale
    )

    serving_amp = math.sqrt(
        config
        .serving_signal_power_scale
    )

    #
    # Scale all non-serving links.
    #
    for bs_index in range(
        num_bs
    ):

        if bs_index == serving_bs:
            continue

        stressed[
            :,
            :,
            :,
            bs_index,
            :,
            :,
            :,
        ] *= interferer_amp

    #
    # Independently scale desired serving link.
    #
    stressed[
        :,
        :,
        :,
        serving_bs,
        :,
        :,
        :,
    ] *= serving_amp

    #
    # Optional sudden BS failure.
    #
    # For another cell, the failed BS disappears exactly
    # as an interferer.
    #
    # For the failed BS's own cell, an exactly-zero desired
    # channel is not a valid input to the downstream MRC /
    # MU-MIMO equations. We therefore preserve the original
    # channel direction but attenuate its power to an
    # effectively-outage numerical floor.
    #
    if config.failed_bs_index is not None:

        failed_bs = (
            config.failed_bs_index
        )

        if failed_bs == serving_bs:

            failed_amplitude_scale = math.sqrt(
                config.failed_serving_power_scale
            )

            stressed[
                :,
                :,
                :,
                failed_bs,
                :,
                :,
                :,
            ] = (
                h_freq[
                    :,
                    :,
                    :,
                    failed_bs,
                    :,
                    :,
                    :,
                ]
                * failed_amplitude_scale
            )

        else:

            stressed[
                :,
                :,
                :,
                failed_bs,
                :,
                :,
                :,
            ] = 0.0

    return replace(
        inputs,
        h_freq=stressed,
    )


class PhysicalExecutionStressInputProvider:
    """
    Preserve scheduler observation and packet arrivals,
    but wrap the candidate PHY builder.

    Thus:

        observation at scheduling time = unchanged

        execution-time physical truth = stressed
    """

    def __init__(
        self,
        *,
        base_provider: InputProvider,
        config: PhysicalExecutionStressConfig,
    ) -> None:

        self.base_provider = (
            base_provider
        )

        self.config = config

        self._last_key: (
            tuple[int, int]
            | None
        ) = None

        self._last_output: (
            PPOTrainingTTIInputs
            | None
        ) = None


    def __call__(
        self,
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:

        key = (
            tti_index,
            stream_index,
        )

        if (
            self._last_key == key
            and self._last_output is not None
        ):
            return self._last_output

        original = (
            self.base_provider(
                tti_index,
                stream_index,
            )
        )

        original_builder = (
            original
            .physical_inputs_builder
        )

        config = self.config


        def stressed_builder(
            prepared: PreparedOneLDSCellTTI,
        ) -> PPOPhysicalScoreInputs:

            inputs = original_builder(
                prepared
            )

            return stress_physical_inputs(
                inputs=inputs,
                config=config,
            )


        output = PPOTrainingTTIInputs(
            observation=(
                original
                .observation
            ),

            physical_inputs_builder=(
                stressed_builder
            ),

            packet_arrivals=(
                original
                .packet_arrivals
            ),
        )

        self._last_key = key
        self._last_output = output

        return output
