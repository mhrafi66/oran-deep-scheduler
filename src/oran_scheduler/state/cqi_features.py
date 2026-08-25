from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class CQISurrogateConfig:
    """
    Configuration for the open-reproduction CQI surrogate.

    The target paper assumes standard wideband and sub-band
    CQI reports but does not publicly specify the exact
    raw-channel -> CQI reporting procedure.

    Current surrogate:

        BLER-target failure
            -> CQI 0

        target-compliant MCS
            -> monotonically mapped to CQI 1..15

    The current PDSCH MCS Table 2 implementation uses
    usable MCS indices through 27.
    """

    max_mcs_index: int = 27

    min_reported_cqi: int = 1
    max_reported_cqi: int = 15

@dataclass
class CQIFeatures:
    """
    CQI features used by the deep scheduler.

    subband_cqi:
        [..., candidate, RBG]

    wideband_cqi:
        [..., candidate]

    Invalid/padded candidates are zeroed.
    """

    subband_cqi: torch.Tensor

    wideband_cqi: torch.Tensor


def validate_cqi_surrogate_config(
    config: CQISurrogateConfig,
) -> None:

    if config.max_mcs_index <= 0:
        raise ValueError(
            "max_mcs_index must be positive."
        )

    if config.min_reported_cqi <= 0:
        raise ValueError(
            "min_reported_cqi must be positive."
        )

    if (
        config.max_reported_cqi
        <= config.min_reported_cqi
    ):
        raise ValueError(
            "max_reported_cqi must be greater "
            "than min_reported_cqi."
        )

def build_cqi_surrogate(
    mcs_index: torch.Tensor,
    meets_bler_target: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    config: CQISurrogateConfig,
) -> CQIFeatures:
    """
    Build sub-band and wideband CQI surrogates.

    Inputs:

        mcs_index:
            [..., candidate, RBG]

        meets_bler_target:
            [..., candidate, RBG]

        candidate_valid_mask:
            [..., candidate]

    Output:

        subband_cqi:
            [..., candidate, RBG]

        wideband_cqi:
            [..., candidate]

    Open-reproduction rule:

        if BLER target is not met:
            CQI = 0

        otherwise:
            map MCS linearly and monotonically
            from 0..max_mcs_index
            onto min_CQI..max_CQI.
    """

    validate_cqi_surrogate_config(
        config
    )

    if mcs_index.ndim < 2:
        raise ValueError(
            "mcs_index must have shape "
            "[..., candidate, RBG]."
        )

    if (
        meets_bler_target.shape
        != mcs_index.shape
    ):
        raise ValueError(
            "meets_bler_target must have the "
            "same shape as mcs_index."
        )


    device = mcs_index.device

    if meets_bler_target.device != device:
        raise ValueError(
            "meets_bler_target and mcs_index "
            "must be on the same device."
        )

    if candidate_valid_mask.device != device:
        raise ValueError(
            "candidate_valid_mask and mcs_index "
            "must be on the same device."
        )

    valid_candidate_mask = (
        candidate_valid_mask.to(
            dtype=torch.bool
        )
    )

    valid_rbg_mask = (
        valid_candidate_mask
        .unsqueeze(-1)
        .expand_as(
            mcs_index
        )
    )

    target_mask = (
        meets_bler_target.to(
            dtype=torch.bool
        )
    )

    valid_mcs = (
        mcs_index[
            valid_rbg_mask
        ]
    )

    if torch.any(
        valid_mcs < 0
    ):
        raise ValueError(
            "A valid candidate has a negative MCS index."
        )

    if torch.any(
        valid_mcs
        > config.max_mcs_index
    ):
        raise ValueError(
            "A valid candidate has an MCS index above "
            "config.max_mcs_index."
        )


    mcs_float = mcs_index.to(
        dtype=torch.float32
    )

    cqi_span = (
        config.max_reported_cqi
        - config.min_reported_cqi
    )

    mapped_cqi = (
        config.min_reported_cqi
        + torch.round(
            mcs_float
            * float(cqi_span)
            / float(
                config.max_mcs_index
            )
        )
    )

    mapped_cqi = torch.clamp(
        mapped_cqi,
        min=float(
            config.min_reported_cqi
        ),
        max=float(
            config.max_reported_cqi
        ),
    )

    usable_link_mask = (
        valid_rbg_mask
        & target_mask
    )

    subband_cqi = torch.where(
        usable_link_mask,
        mapped_cqi,
        torch.zeros_like(
            mapped_cqi
        ),
    )

    subband_cqi = (
        subband_cqi.to(
            dtype=torch.long
        )
    )

    wideband_cqi_float = (
        subband_cqi
        .to(
            dtype=torch.float32
        )
        .mean(
            dim=-1
        )
    )

    wideband_cqi = torch.round(
        wideband_cqi_float
    ).to(
        dtype=torch.long
    )

    wideband_cqi = torch.where(
        valid_candidate_mask,
        wideband_cqi,
        torch.zeros_like(
            wideband_cqi
        ),
    )

    return CQIFeatures(
        subband_cqi=subband_cqi,
        wideband_cqi=wideband_cqi,
    )


