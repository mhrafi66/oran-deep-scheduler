from dataclasses import dataclass
import math
from typing import Literal

import torch


TDSBufferEligibilityMode = Literal[
    "all_serving_ues",
    "data_available_only",
]


@dataclass(frozen=True)
class TDSBufferEligibilityConfig:
    """
    Decide which associated serving UEs may enter
    PF time-domain scheduling.

    PAPER-UNSPECIFIED:
        The public paper does not fully specify
        whether finite-buffer UEs with an empty DL
        queue are filtered before PF-TDS.

    REPRODUCTION OPTIONS:

        "all_serving_ues":
            Every real associated UE is eligible,
            regardless of finite-buffer backlog.

        "data_available_only":
            Full-Buffer UEs are always eligible.

            Finite-buffer UEs are eligible only when
            their queue exceeds
            finite_buffer_min_bits.
    """

    mode: TDSBufferEligibilityMode

    finite_buffer_min_bits: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in (
            "all_serving_ues",
            "data_available_only",
        ):
            raise ValueError(
                "mode must be "
                "'all_serving_ues' or "
                "'data_available_only'."
            )

        if (
            not math.isfinite(
                self.finite_buffer_min_bits
            )
            or self.finite_buffer_min_bits < 0.0
        ):
            raise ValueError(
                "finite_buffer_min_bits must be "
                "non-negative and finite."
            )


@dataclass(frozen=True)
class TDSEligibilityData:
    """
    Serving-UE eligibility for one TTI.

    Shapes:

        serving_ue_valid_mask:
            [serving_ue]

        has_finite_buffer_data:
            [serving_ue]

        eligible_mask:
            [serving_ue]
    """

    serving_ue_valid_mask: torch.Tensor

    has_finite_buffer_data: torch.Tensor

    eligible_mask: torch.Tensor

    num_valid_serving_ues: int

    num_eligible_ues: int


def build_tds_eligibility(
    *,
    serving_ue_valid_mask: torch.Tensor,
    dl_buffer_bits: torch.Tensor,
    full_buffer_mask: torch.Tensor,
    config: TDSBufferEligibilityConfig,
) -> TDSEligibilityData:
    """
    Build the per-TTI PF-TDS eligibility mask.

    Association validity and scheduling eligibility
    remain separate concepts.
    """

    if serving_ue_valid_mask.ndim != 1:
        raise ValueError(
            "serving_ue_valid_mask must have shape "
            "[serving_ue]."
        )

    expected_shape = (
        serving_ue_valid_mask.shape
    )

    if tuple(
        dl_buffer_bits.shape
    ) != tuple(
        expected_shape
    ):
        raise ValueError(
            "dl_buffer_bits must match the serving "
            "UE layout."
        )

    if tuple(
        full_buffer_mask.shape
    ) != tuple(
        expected_shape
    ):
        raise ValueError(
            "full_buffer_mask must match the serving "
            "UE layout."
        )

    if serving_ue_valid_mask.dtype != torch.bool:
        raise ValueError(
            "serving_ue_valid_mask must use "
            "torch.bool."
        )

    if full_buffer_mask.dtype != torch.bool:
        raise ValueError(
            "full_buffer_mask must use torch.bool."
        )

    if not torch.is_floating_point(
        dl_buffer_bits
    ):
        raise ValueError(
            "dl_buffer_bits must use a "
            "floating-point dtype."
        )

    device = serving_ue_valid_mask.device

    if (
        dl_buffer_bits.device != device
        or full_buffer_mask.device != device
    ):
        raise ValueError(
            "Eligibility tensors must be on the "
            "same device."
        )

    if not torch.isfinite(
        dl_buffer_bits
    ).all():
        raise ValueError(
            "dl_buffer_bits contains non-finite "
            "values."
        )

    if torch.any(
        dl_buffer_bits < 0.0
    ):
        raise ValueError(
            "dl_buffer_bits cannot be negative."
        )

    has_finite_buffer_data = (
        dl_buffer_bits
        > config.finite_buffer_min_bits
    )

    if config.mode == "all_serving_ues":
        eligible_mask = (
            serving_ue_valid_mask.clone()
        )

    elif config.mode == "data_available_only":
        has_traffic_to_schedule = (
            full_buffer_mask
            | has_finite_buffer_data
        )

        eligible_mask = (
            serving_ue_valid_mask
            & has_traffic_to_schedule
        )

    else:
        raise RuntimeError(
            "Unexpected TDS eligibility mode."
        )

    return TDSEligibilityData(
        serving_ue_valid_mask=(
            serving_ue_valid_mask
            .detach()
            .clone()
        ),
        has_finite_buffer_data=(
            has_finite_buffer_data
            .detach()
            .clone()
        ),
        eligible_mask=(
            eligible_mask
            .detach()
            .clone()
        ),
        num_valid_serving_ues=int(
            serving_ue_valid_mask
            .sum()
            .item()
        ),
        num_eligible_ues=int(
            eligible_mask
            .sum()
            .item()
        ),
    )


