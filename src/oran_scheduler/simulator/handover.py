from __future__ import annotations

from dataclasses import dataclass
import math

import torch


NO_PENDING_TARGET = -1


@dataclass(frozen=True)
class HandoverConfig:
    """
    OPEN-REPRODUCTION handover decision model.

    This is not claimed to reproduce a proprietary
    Nokia handover implementation.

    A UE changes serving BS only when another BS is
    stronger than the current serving BS by at least

        hysteresis_db

    for

        time_to_trigger_ttis

    consecutive scheduler TTIs.

    This avoids treating every instantaneous
    strongest-BS fluctuation as a handover.
    """

    hysteresis_db: float = 3.0

    time_to_trigger_ttis: int = 2


    def __post_init__(
        self,
    ) -> None:

        if (
            not math.isfinite(
                self.hysteresis_db
            )
            or self.hysteresis_db < 0.0
        ):
            raise ValueError(
                "hysteresis_db must be finite "
                "and non-negative."
            )

        if self.time_to_trigger_ttis <= 0:
            raise ValueError(
                "time_to_trigger_ttis must be "
                "positive."
            )


@dataclass(frozen=True)
class HandoverStepResult:
    """
    Result of one handover-controller TTI.

    Shapes:
        [UE]
    """

    tti_index: int

    previous_serving_bs: torch.Tensor

    serving_bs: torch.Tensor

    strongest_bs: torch.Tensor

    strongest_gain_db: torch.Tensor

    pending_target_bs: torch.Tensor

    pending_count: torch.Tensor

    handover_mask: torch.Tensor


class HandoverController:
    """
    Stateful global-UE handover controller.

    The controller tracks serving BS by persistent UE
    identity position.

    Input link power:

        [UE, BS]

    Values are linear powers, not dB.

    The controller does NOT:
        - move traffic queues
        - move PF histories
        - rebuild scheduler cell layouts
        - model interruption time

    Those are separate integration stages.
    """

    def __init__(
        self,
        *,
        initial_serving_bs: torch.Tensor,
        num_bs: int,
        config: HandoverConfig,
    ) -> None:

        if (
            initial_serving_bs.ndim != 1
        ):
            raise ValueError(
                "initial_serving_bs must have "
                "shape [UE]."
            )

        if (
            initial_serving_bs.dtype
            == torch.bool
            or torch.is_floating_point(
                initial_serving_bs
            )
        ):
            raise ValueError(
                "initial_serving_bs must use "
                "an integer dtype."
            )

        if num_bs <= 0:
            raise ValueError(
                "num_bs must be positive."
            )

        if torch.any(
            initial_serving_bs < 0
        ):
            raise ValueError(
                "initial serving BS cannot "
                "be negative."
            )

        if torch.any(
            initial_serving_bs >= num_bs
        ):
            raise ValueError(
                "initial serving BS outside "
                "available BS range."
            )

        self.config = config

        self.num_bs = int(
            num_bs
        )

        self.device = (
            initial_serving_bs.device
        )

        self.serving_bs = (
            initial_serving_bs
            .detach()
            .clone()
            .to(
                dtype=torch.long,
            )
        )

        self.pending_target_bs = (
            torch.full_like(
                self.serving_bs,
                fill_value=(
                    NO_PENDING_TARGET
                ),
            )
        )

        self.pending_count = (
            torch.zeros_like(
                self.serving_bs,
                dtype=torch.long,
            )
        )

        self._last_tti_index: (
            int | None
        ) = None


    @property
    def num_ues(
        self,
    ) -> int:

        return int(
            self.serving_bs.numel()
        )


    def step(
        self,
        *,
        tti_index: int,
        link_power: torch.Tensor,
    ) -> HandoverStepResult:
        """
        Advance handover state by one TTI.

        link_power:
            linear received/link-power metric
            shape [UE, BS]

        A target must remain the strongest qualifying
        target for TTT consecutive calls.
        """

        if tti_index < 0:
            raise ValueError(
                "tti_index must be non-negative."
            )

        if (
            self._last_tti_index
            is not None
            and tti_index
            <= self._last_tti_index
        ):
            raise ValueError(
                "tti_index must increase "
                "strictly."
            )

        if link_power.ndim != 2:
            raise ValueError(
                "link_power must have "
                "shape [UE, BS]."
            )

        if tuple(
            link_power.shape
        ) != (
            self.num_ues,
            self.num_bs,
        ):
            raise ValueError(
                "link_power shape does not match "
                "handover population."
            )

        if (
            link_power.device
            != self.device
        ):
            raise ValueError(
                "link_power device does not match "
                "handover state."
            )

        if not torch.is_floating_point(
            link_power
        ):
            raise ValueError(
                "link_power must use a floating "
                "point dtype."
            )

        if not torch.isfinite(
            link_power
        ).all():
            raise ValueError(
                "link_power contains non-finite "
                "values."
            )

        if torch.any(
            link_power < 0.0
        ):
            raise ValueError(
                "link_power cannot be negative."
            )

        previous_serving_bs = (
            self.serving_bs
            .detach()
            .clone()
        )

        strongest_bs = torch.argmax(
            link_power,
            dim=1,
        )

        ue_index = torch.arange(
            self.num_ues,
            device=self.device,
        )

        current_power = link_power[
            ue_index,
            self.serving_bs,
        ]

        strongest_power = link_power[
            ue_index,
            strongest_bs,
        ]

        tiny = torch.finfo(
            link_power.dtype
        ).tiny

        strongest_gain_db = (
            10.0
            * torch.log10(
                torch.clamp(
                    strongest_power,
                    min=tiny,
                )
                /
                torch.clamp(
                    current_power,
                    min=tiny,
                )
            )
        )

        qualifies = (
            (
                strongest_bs
                != self.serving_bs
            )
            &
            (
                strongest_gain_db
                >= self.config.hysteresis_db
            )
        )

        same_pending_target = (
            qualifies
            &
            (
                self.pending_target_bs
                == strongest_bs
            )
        )

        next_pending_count = torch.where(
            qualifies,

            torch.where(
                same_pending_target,
                self.pending_count + 1,
                torch.ones_like(
                    self.pending_count
                ),
            ),

            torch.zeros_like(
                self.pending_count
            ),
        )

        next_pending_target = torch.where(
            qualifies,

            strongest_bs,

            torch.full_like(
                strongest_bs,
                fill_value=(
                    NO_PENDING_TARGET
                ),
            ),
        )

        handover_mask = (
            qualifies
            &
            (
                next_pending_count
                >= (
                    self.config
                    .time_to_trigger_ttis
                )
            )
        )

        self.serving_bs = torch.where(
            handover_mask,
            strongest_bs,
            self.serving_bs,
        )

        #
        # A completed handover clears its pending
        # candidate/TTT state.
        #
        self.pending_target_bs = (
            torch.where(
                handover_mask,

                torch.full_like(
                    next_pending_target,
                    fill_value=(
                        NO_PENDING_TARGET
                    ),
                ),

                next_pending_target,
            )
        )

        self.pending_count = torch.where(
            handover_mask,

            torch.zeros_like(
                next_pending_count
            ),

            next_pending_count,
        )

        self._last_tti_index = int(
            tti_index
        )

        return HandoverStepResult(
            tti_index=tti_index,

            previous_serving_bs=(
                previous_serving_bs
            ),

            serving_bs=(
                self.serving_bs
                .detach()
                .clone()
            ),

            strongest_bs=(
                strongest_bs
                .detach()
                .clone()
            ),

            strongest_gain_db=(
                strongest_gain_db
                .detach()
                .clone()
            ),

            pending_target_bs=(
                self.pending_target_bs
                .detach()
                .clone()
            ),

            pending_count=(
                self.pending_count
                .detach()
                .clone()
            ),

            handover_mask=(
                handover_mask
                .detach()
                .clone()
            ),
        )
