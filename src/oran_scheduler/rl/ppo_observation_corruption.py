from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch

from oran_scheduler.rl.ppo_training_runner import (
    PPOTrainingTTIInputs,
)
from oran_scheduler.simulator.one_lds_cell_tti import (
    OneLDSCellTTIObservation,
)


InputProvider = Callable[
    [int, int],
    PPOTrainingTTIInputs,
]


@dataclass(frozen=True)
class ObservationCorruptionConfig:
    """
    Evaluation-only scheduler-observation corruption.

    The physical_inputs_builder is NEVER modified.
    Therefore these perturbations change what the
    scheduler believes, not the actual current PHY.
    """

    cqi_bias: float = 0.0
    cqi_noise_std: float = 0.0
    cqi_quant_step: float = 0.0
    flatten_subband_cqi: bool = False

    rank_flip_prob: float = 0.0
    rank_force: int = 0

    precoder_ue_shuffle: bool = False
    precoder_rbg_shuffle: bool = False

    td_rate_log_noise_std: float = 0.0
    td_rate_dropout_prob: float = 0.0
    td_rate_shuffle: bool = False

    seed: int = 918273


    def __post_init__(self) -> None:

        for name, value in (
            (
                "cqi_noise_std",
                self.cqi_noise_std,
            ),
            (
                "cqi_quant_step",
                self.cqi_quant_step,
            ),
            (
                "td_rate_log_noise_std",
                self.td_rate_log_noise_std,
            ),
        ):
            if (
                not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be finite "
                    "and non-negative."
                )

        for name, value in (
            (
                "rank_flip_prob",
                self.rank_flip_prob,
            ),
            (
                "td_rate_dropout_prob",
                self.td_rate_dropout_prob,
            ),
        ):
            if not (
                0.0
                <= value
                <= 1.0
            ):
                raise ValueError(
                    f"{name} must lie in [0, 1]."
                )

        if self.rank_force not in (
            0,
            1,
            2,
        ):
            raise ValueError(
                "rank_force must be 0, 1, or 2."
            )


def _generator(
    *,
    device: torch.device,
    seed: int,
) -> torch.Generator:

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        seed
    )

    return generator


class ObservationCorruptionInputProvider:

    def __init__(
        self,
        *,
        base_provider: InputProvider,
        config: ObservationCorruptionConfig,
    ) -> None:

        self.base_provider = (
            base_provider
        )

        self.config = config

        self._last_key = None
        self._last_output = None


    def __call__(
        self,
        tti_index: int,
        stream_index: int,
    ) -> PPOTrainingTTIInputs:

        key = (
            tti_index,
            stream_index,
        )

        if key == self._last_key:
            return self._last_output

        current_inputs = (
            self.base_provider(
                tti_index,
                stream_index,
            )
        )

        obs = (
            current_inputs.observation
        )

        valid = (
            obs
            .serving_ue_valid_mask
            .to(
                dtype=torch.bool
            )
        )

        device = (
            obs
            .td_instantaneous_rate_bps
            .device
        )

        random_seed = (
            self.config.seed
            + 1_000_003 * tti_index
            + 9_176 * stream_index
        )

        generator = _generator(
            device=device,
            seed=random_seed,
        )

        # ======================================================
        # TD-rate / PF-TDS front end
        # ======================================================

        td_rate = (
            obs
            .td_instantaneous_rate_bps
            .detach()
            .clone()
        )

        sigma = (
            self.config
            .td_rate_log_noise_std
        )

        if sigma > 0.0:

            noise = torch.randn(
                td_rate.shape,
                dtype=td_rate.dtype,
                device=device,
                generator=generator,
            )

            #
            # Mean-preserving log-normal multiplier.
            #
            multiplier = torch.exp(
                sigma * noise
                - 0.5 * sigma * sigma
            )

            td_rate = (
                td_rate
                * multiplier
            )

        dropout_probability = (
            self.config
            .td_rate_dropout_prob
        )

        if dropout_probability > 0.0:

            dropout = (
                torch.rand(
                    td_rate.shape,
                    device=device,
                    generator=generator,
                )
                < dropout_probability
            )

            dropout = (
                dropout
                & valid
            )

            td_rate = torch.where(
                dropout,
                torch.zeros_like(
                    td_rate
                ),
                td_rate,
            )

        if self.config.td_rate_shuffle:

            valid_indices = (
                torch.nonzero(
                    valid,
                    as_tuple=False,
                )
                .flatten()
            )

            if valid_indices.numel() > 1:

                order = torch.randperm(
                    valid_indices.numel(),
                    device=device,
                    generator=generator,
                )

                sources = (
                    valid_indices[
                        order
                    ]
                )

                shuffled = (
                    td_rate.clone()
                )

                shuffled[
                    valid_indices
                ] = td_rate[
                    sources
                ]

                td_rate = shuffled


        # ======================================================
        # CQI
        # ======================================================

        subband_cqi = (
            obs
            .subband_cqi
            .detach()
            .clone()
            .to(
                dtype=torch.float32
            )
        )

        cqi_modified = (
            self.config.flatten_subband_cqi
            or self.config.cqi_bias != 0.0
            or self.config.cqi_noise_std > 0.0
            or self.config.cqi_quant_step > 0.0
        )

        if self.config.flatten_subband_cqi:

            subband_cqi = (
                obs
                .wideband_cqi
                .to(
                    dtype=torch.float32
                )
                .unsqueeze(-1)
                .expand_as(
                    subband_cqi
                )
                .clone()
            )

        if self.config.cqi_bias != 0.0:

            subband_cqi = (
                subband_cqi
                + self.config.cqi_bias
            )

        if self.config.cqi_noise_std > 0.0:

            cqi_noise = torch.randn(
                subband_cqi.shape,
                dtype=subband_cqi.dtype,
                device=device,
                generator=generator,
            )

            subband_cqi = (
                subband_cqi
                + (
                    self.config
                    .cqi_noise_std
                    * cqi_noise
                )
            )

        if self.config.cqi_quant_step > 0.0:

            step = (
                self.config
                .cqi_quant_step
            )

            subband_cqi = (
                torch.round(
                    subband_cqi
                    / step
                )
                * step
            )

        subband_cqi = torch.clamp(
            subband_cqi,
            min=0.0,
            max=15.0,
        )

        subband_cqi = torch.where(
            valid.unsqueeze(-1),
            subband_cqi,
            torch.zeros_like(
                subband_cqi
            ),
        )

        if cqi_modified:

            wideband_cqi = (
                torch.round(
                    subband_cqi.mean(
                        dim=-1
                    )
                )
            )

        else:

            wideband_cqi = (
                obs
                .wideband_cqi
                .detach()
                .clone()
            )


        # ======================================================
        # RI / rank
        # ======================================================

        rank = (
            obs.rank
            .detach()
            .clone()
        )

        if self.config.rank_force != 0:

            rank = torch.where(
                valid,
                torch.full_like(
                    rank,
                    self.config.rank_force,
                ),
                rank,
            )

        elif self.config.rank_flip_prob > 0.0:

            flip = (
                torch.rand(
                    rank.shape,
                    device=device,
                    generator=generator,
                )
                < (
                    self.config
                    .rank_flip_prob
                )
            )

            flip = (
                flip
                & valid
                & (
                    (rank == 1)
                    | (rank == 2)
                )
            )

            flipped_rank = (
                3 - rank
            )

            rank = torch.where(
                flip,
                flipped_rank,
                rank,
            )


        # ======================================================
        # PMI / precoder-direction information
        # ======================================================

        precoder = (
            obs
            .precoder_directions
            .detach()
            .clone()
        )

        if self.config.precoder_ue_shuffle:

            valid_indices = (
                torch.nonzero(
                    valid,
                    as_tuple=False,
                )
                .flatten()
            )

            if valid_indices.numel() > 1:

                order = torch.randperm(
                    valid_indices.numel(),
                    device=device,
                    generator=generator,
                )

                sources = (
                    valid_indices[
                        order
                    ]
                )

                shuffled_precoder = (
                    precoder.clone()
                )

                shuffled_precoder[
                    valid_indices
                ] = precoder[
                    sources
                ]

                precoder = (
                    shuffled_precoder
                )

        if (
            self.config.precoder_rbg_shuffle
            and precoder.ndim >= 2
            and precoder.shape[1] > 1
        ):

            rbg_order = torch.randperm(
                precoder.shape[1],
                device=device,
                generator=generator,
            )

            precoder = (
                precoder[
                    :,
                    rbg_order,
                    ...,
                ]
            )


        stressed_observation = (
            OneLDSCellTTIObservation(
                serving_global_ue_indices=(
                    obs
                    .serving_global_ue_indices
                ),

                serving_ue_valid_mask=(
                    obs
                    .serving_ue_valid_mask
                ),

                td_instantaneous_rate_bps=(
                    td_rate
                ),

                rank=rank,

                #
                # The real traffic manager later
                # supplies current buffer state.
                #
                dl_buffer=(
                    obs.dl_buffer
                ),

                wideband_cqi=(
                    wideband_cqi
                ),

                subband_cqi=(
                    subband_cqi
                ),

                precoder_directions=(
                    precoder
                ),
            )
        )

        output = PPOTrainingTTIInputs(
            observation=(
                stressed_observation
            ),

            #
            # Critical:
            # physical truth remains CURRENT.
            #
            physical_inputs_builder=(
                current_inputs
                .physical_inputs_builder
            ),

            packet_arrivals=(
                current_inputs
                .packet_arrivals
            ),
        )

        self._last_key = key
        self._last_output = output

        return output
