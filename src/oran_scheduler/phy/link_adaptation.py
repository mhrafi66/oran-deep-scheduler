from dataclasses import dataclass

import torch

from sionna.sys import (
    InnerLoopLinkAdaptation,
    PHYAbstraction,
)


@dataclass(frozen=True)
class LinkAdaptationConfig:
    """
    Configuration for the initial downlink link-adaptation model.

    Paper-specified:
        BLER target = 10%
        Downlink supports modulation through 256-QAM

    Open-reproduction choices:
        mcs_category = 1:
            PDSCH / downlink

        mcs_table_index = 2:
            NR PDSCH MCS Table 2, which includes 256-QAM.

        num_slot_ofdm_symbols = 14:
            The current generated channel contains one OFDM-symbol
            channel snapshot. For link adaptation, we temporarily
            assume that SINR is constant across a normal 14-symbol slot.

        All 12 x 14 REs of an RBG are currently treated as available.
        DMRS/control overhead is not yet removed.
    """

    bler_target: float = 0.10

    mcs_category: int = 1
    mcs_table_index: int = 2

    num_rbgs: int = 18
    subcarriers_per_rbg: int = 12

    num_slot_ofdm_symbols: int = 14

    precision: str = "single"
    device: str = "cuda:0"

@dataclass
class LinkAdaptationData:
    """
    Per-RBG downlink link-adaptation result.

    All scheduler-facing tensors use layout:

        [batch, UE, RBG]

    illa_sinr keeps the Sionna layout:

        [batch,
         RBG,
         slot_OFDM_symbol,
         subcarrier_within_RBG,
         UE,
         stream]
    """

    mcs_index: torch.Tensor

    lowest_available_mcs_index: torch.Tensor

    effective_sinr_linear: torch.Tensor

    tbler: torch.Tensor

    bler: torch.Tensor

    meets_bler_target: torch.Tensor

    illa_sinr: torch.Tensor

def validate_link_adaptation_input(
    sinr_linear: torch.Tensor,
    config: LinkAdaptationConfig,
) -> None:
    """
    Validate the physical SINR tensor before RBG conversion.

    Expected input:

        [batch, UE, 1, total_subcarriers]

    where:

        total_subcarriers
            =
        num_rbgs * subcarriers_per_rbg
    """

    if sinr_linear.ndim != 4:
        raise ValueError(
            "Expected SINR shape "
            "[batch, UE, OFDM_symbol, subcarrier], "
            f"got {tuple(sinr_linear.shape)}."
        )

    if sinr_linear.shape[2] != 1:
        raise ValueError(
            "Current link-adaptation implementation expects "
            "one generated OFDM-symbol channel snapshot."
        )

    expected_num_subcarriers = (
        config.num_rbgs
        * config.subcarriers_per_rbg
    )

    if sinr_linear.shape[-1] != expected_num_subcarriers:
        raise ValueError(
            f"Expected {expected_num_subcarriers} subcarriers, "
            f"got {sinr_linear.shape[-1]}."
        )

    if torch.any(sinr_linear < 0):
        raise ValueError(
            "SINR cannot be negative."
        )

    if not torch.isfinite(
        sinr_linear
    ).all():
        raise ValueError(
            "SINR contains non-finite values."
        )

def build_illa_rbg_sinr(
    sinr_linear: torch.Tensor,
    config: LinkAdaptationConfig,
) -> torch.Tensor:
    """
    Convert physical SINR into the layout required by Sionna ILLA.

    Input:

        [B, UE, 1, 216]

    First group frequency into 18 RBGs:

        [B, UE, 1, 18, 12]

    Reorder:

        [B, 18, 1, 12, UE]

    Add stream dimension:

        [B, 18, 1, 12, UE, 1]

    Then repeat the one-symbol SINR snapshot across the assumed
    14-symbol scheduling slot:

        [B, 18, 14, 12, UE, 1]

    Sionna interprets the final four dimensions as:

        [OFDM_symbol, subcarrier, UE, stream]

    while [B, RBG] are batch dimensions.
    """

    validate_link_adaptation_input(
        sinr_linear=sinr_linear,
        config=config,
    )

    batch_size = sinr_linear.shape[0]
    num_ues = sinr_linear.shape[1]


    rbg_sinr = sinr_linear.reshape(
        batch_size,
        num_ues,
        1,
        config.num_rbgs,
        config.subcarriers_per_rbg,
    )

    rbg_sinr = rbg_sinr.permute(
        0,
        3,
        2,
        4,
        1,
    )

    rbg_sinr = rbg_sinr.unsqueeze(
        dim=-1,
    )

    illa_sinr = rbg_sinr.expand(
        -1,
        -1,
        config.num_slot_ofdm_symbols,
        -1,
        -1,
        -1,
    )

    return illa_sinr

def create_link_adaptation_blocks(
    config: LinkAdaptationConfig,
) -> tuple[
    PHYAbstraction,
    InnerLoopLinkAdaptation,
]:
    """
    Create Sionna PHY abstraction and inner-loop link adaptation.
    """

    phy_abstraction = PHYAbstraction(
        precision=config.precision,
        device=config.device,
    )

    illa = InnerLoopLinkAdaptation(
        phy_abstraction=phy_abstraction,
        bler_target=config.bler_target,
        precision=config.precision,
        device=config.device,
    )

    return (
        phy_abstraction,
        illa,
    )

def select_rbg_mcs(
    sinr_linear: torch.Tensor,
    config: LinkAdaptationConfig,
) -> LinkAdaptationData:
    """
    Select one downlink MCS for every UE/RBG.

    Output:

        mcs_index:
            [batch, UE, RBG]
    """

    illa_sinr = build_illa_rbg_sinr(
        sinr_linear=sinr_linear,
        config=config,
    )

    phy_abstraction, illa = (
        create_link_adaptation_blocks(
            config
        )
    )

    (
        mcs_by_rbg,
        lowest_mcs_by_rbg,
    ) = illa(
        sinr=illa_sinr,
        mcs_table_index=config.mcs_table_index,
        mcs_category=config.mcs_category,
        return_lowest_available_mcs=True,
    )

    (
        _,
        _,
        effective_sinr_by_rbg,
        tbler_by_rbg,
        bler_by_rbg,
    ) = phy_abstraction(
        mcs_index=mcs_by_rbg,
        sinr=illa_sinr,
        mcs_table_index=config.mcs_table_index,
        mcs_category=config.mcs_category,
    )

    if mcs_by_rbg.ndim != 3:
        raise ValueError(
            "Unexpected ILLA output rank. "
            f"Expected [batch, RBG, UE], "
            f"got {tuple(mcs_by_rbg.shape)}."
        )

    mcs_index = mcs_by_rbg.permute(
        0,
        2,
        1,
    )

    lowest_available_mcs_index = (
        lowest_mcs_by_rbg.permute(
            0,
            2,
            1,
        )
    )

    effective_sinr_linear = (
        effective_sinr_by_rbg.permute(
            0,
            2,
            1,
        )
    )

    tbler = tbler_by_rbg.permute(
        0,
        2,
        1,
    )

    bler = bler_by_rbg.permute(
        0,
        2,
        1,
    )

    meets_bler_target = (
        tbler <= config.bler_target
    )

    return LinkAdaptationData(
        mcs_index=mcs_index,
        lowest_available_mcs_index=(
            lowest_available_mcs_index
        ),
        effective_sinr_linear=(
            effective_sinr_linear
        ),
        tbler=tbler,
        bler=bler,
        meets_bler_target=(
            meets_bler_target
        ),
        illa_sinr=illa_sinr,
    )


