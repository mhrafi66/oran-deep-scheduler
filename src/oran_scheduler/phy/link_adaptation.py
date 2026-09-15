from dataclasses import dataclass
from functools import lru_cache

import torch

from sionna.sys import (
    InnerLoopLinkAdaptation,
    PHYAbstraction,
)


from oran_scheduler.utils.perf_timing import (
    perf_region,
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


@dataclass
class MUMIMORBGLinkAdaptationData:
    """
    Link-adaptation result for the selected UEs on one RBG.

    All tensors have shape:

        [selected_UE]

    A rank-2 UE still receives one MCS and one transport
    block. Its two spatial layers are treated as two streams
    belonging to that UE.
    """

    mcs_index: torch.Tensor

    lowest_available_mcs_index: torch.Tensor

    effective_sinr_linear: torch.Tensor

    tbler: torch.Tensor

    bler: torch.Tensor

    meets_bler_target: torch.Tensor


def validate_mu_mimo_rbg_link_adaptation_input(
    layer_sinr_linear: torch.Tensor,
    layer_ue_indices: torch.Tensor,
    layer_index_within_ue: torch.Tensor,
    selected_ranks: torch.Tensor,
    config: LinkAdaptationConfig,
) -> None:
    """
    Validate layer-level SINR before regrouping layers by UE.

    Expected:

        layer_sinr_linear:
            [OFDM_symbol, subcarrier, layer]

        layer_ue_indices:
            [layer]

        layer_index_within_ue:
            [layer]

        selected_ranks:
            [selected_UE]
    """

    if layer_sinr_linear.ndim != 3:
        raise ValueError(
            "layer_sinr_linear must have shape "
            "[OFDM_symbol, subcarrier, layer]."
        )

    num_layers = layer_sinr_linear.shape[-1]

    if tuple(
        layer_ue_indices.shape
    ) != (
        num_layers,
    ):
        raise ValueError(
            "layer_ue_indices must have shape [layer]."
        )

    if tuple(
        layer_index_within_ue.shape
    ) != (
        num_layers,
    ):
        raise ValueError(
            "layer_index_within_ue must have shape [layer]."
        )

    if selected_ranks.ndim != 1:
        raise ValueError(
            "selected_ranks must have shape [selected_UE]."
        )

    if torch.any(
        (selected_ranks < 1)
        | (selected_ranks > 2)
    ):
        raise ValueError(
            "Selected UE ranks must be either 1 or 2."
        )

    if layer_sinr_linear.shape[1] != (
        config.subcarriers_per_rbg
    ):
        raise ValueError(
            "Unexpected number of subcarriers in the RBG."
        )

    num_generated_symbols = (
        layer_sinr_linear.shape[0]
    )

    if num_generated_symbols not in (
        1,
        config.num_slot_ofdm_symbols,
    ):
        raise ValueError(
            "MU-MIMO SINR must contain either one "
            "generated OFDM symbol or a complete slot."
        )

    if torch.any(
        layer_sinr_linear < 0
    ):
        raise ValueError(
            "SINR cannot be negative."
        )

    if not torch.isfinite(
        layer_sinr_linear
    ).all():
        raise ValueError(
            "SINR contains non-finite values."
        )

    num_selected_ues = (
        selected_ranks.numel()
    )

    if torch.any(
        layer_ue_indices < 0
    ):
        raise ValueError(
            "layer_ue_indices cannot be negative."
        )

    if torch.any(
        layer_ue_indices
        >= num_selected_ues
    ):
        raise ValueError(
            "layer_ue_indices contains an invalid UE."
        )

    if torch.any(
        layer_index_within_ue < 0
    ):
        raise ValueError(
            "layer_index_within_ue cannot be negative."
        )

    if torch.any(
        layer_index_within_ue >= 2
    ):
        raise ValueError(
            "Only UE ranks up to 2 are supported."
        )

    expected_num_layers = int(
        selected_ranks.sum().item()
    )

    if num_layers != expected_num_layers:
        raise ValueError(
            "Number of layer SINRs does not equal "
            "the sum of selected UE ranks."
        )

def build_layer_lookup(
    layer_ue_indices: torch.Tensor,
    layer_index_within_ue: torch.Tensor,
    selected_ranks: torch.Tensor,
) -> torch.Tensor:
    """
    Map:

        [UE, stream_within_UE]

    to the corresponding index in the flattened layer tensor.

    Example:

        selected_ranks = [2, 1, 1]

        flattened layers:
            A1 A2 B1 C1

        output:

            [
                [0,  1],
                [2, -1],
                [3, -1],
            ]
    """

    num_selected_ues = (
        selected_ranks.numel()
    )

    layer_lookup = torch.full(
        (
            num_selected_ues,
            2,
        ),
        fill_value=-1,
        dtype=torch.long,
        device=selected_ranks.device,
    )

    layer_numbers = torch.arange(
        layer_ue_indices.numel(),
        dtype=torch.long,
        device=selected_ranks.device,
    )

    layer_lookup[
        layer_ue_indices,
        layer_index_within_ue,
    ] = layer_numbers

    possible_streams = torch.arange(
        2,
        dtype=torch.long,
        device=selected_ranks.device,
    )

    expected_valid = (
        possible_streams.unsqueeze(0)
        < selected_ranks.unsqueeze(1)
    )

    if torch.any(
        layer_lookup[
            expected_valid
        ] < 0
    ):
        raise ValueError(
            "At least one expected UE layer is missing."
        )

    if torch.any(
        layer_lookup[
            ~expected_valid
        ] >= 0
    ):
        raise ValueError(
            "A layer exists beyond its UE's rank."
        )

    return layer_lookup

def select_mu_mimo_rbg_mcs(
    layer_sinr_linear: torch.Tensor,
    layer_ue_indices: torch.Tensor,
    layer_index_within_ue: torch.Tensor,
    selected_ranks: torch.Tensor,
    config: LinkAdaptationConfig,
) -> MUMIMORBGLinkAdaptationData:
    """
    Select one MCS per selected UE on one MU-MIMO RBG.

    Rank-1 and rank-2 UEs are processed separately because
    Sionna's stream dimension is common within one SINR tensor.

    A rank-r UE is represented to Sionna as:

        [OFDM symbol, subcarrier, UE, r streams]

    and receives one MCS / transport block.
    """

    validate_mu_mimo_rbg_link_adaptation_input(
        layer_sinr_linear=layer_sinr_linear,
        layer_ue_indices=layer_ue_indices,
        layer_index_within_ue=(
            layer_index_within_ue
        ),
        selected_ranks=selected_ranks,
        config=config,
    )

    layer_lookup = build_layer_lookup(
        layer_ue_indices=layer_ue_indices,
        layer_index_within_ue=(
            layer_index_within_ue
        ),
        selected_ranks=selected_ranks,
    )

    num_selected_ues = (
        selected_ranks.numel()
    )

    device = layer_sinr_linear.device
    real_dtype = layer_sinr_linear.dtype

    mcs_index = torch.empty(
        num_selected_ues,
        dtype=torch.long,
        device=device,
    )

    lowest_available_mcs_index = torch.empty(
        num_selected_ues,
        dtype=torch.long,
        device=device,
    )

    effective_sinr_linear = torch.empty(
        num_selected_ues,
        dtype=real_dtype,
        device=device,
    )

    tbler = torch.empty(
        num_selected_ues,
        dtype=real_dtype,
        device=device,
    )

    bler = torch.empty(
        num_selected_ues,
        dtype=real_dtype,
        device=device,
    )

    with perf_region(
        "mu_la.blocks",
        device=device,
    ):
        phy_abstraction, illa = (
            create_link_adaptation_blocks(
                config
            )
        )

    for rank_value in (
        1,
        2,
    ):

        ue_indices = torch.nonzero(
            selected_ranks
            == rank_value,
            as_tuple=False,
        ).flatten()

        if ue_indices.numel() == 0:
            continue

        group_layer_indices = (
            layer_lookup[
                ue_indices,
                :rank_value,
            ]
        )

        group_sinr = (
            layer_sinr_linear[
                ...,
                group_layer_indices,
            ]
        )

        if group_sinr.shape[0] == 1:

            group_sinr = group_sinr.expand(
                config.num_slot_ofdm_symbols,
                -1,
                -1,
                -1,
            )

        illa_sinr = (
            group_sinr.unsqueeze(0)
        )

        with perf_region(
            "mu_la.illa",
            device=device,
        ):
            (
                group_mcs,
                group_lowest_mcs,
            ) = illa(
                sinr=illa_sinr,
                mcs_table_index=(
                    config.mcs_table_index
                ),
                mcs_category=(
                    config.mcs_category
                ),
                return_lowest_available_mcs=True,
            )
        with perf_region(
            "mu_la.selected_phy",
            device=device,
        ):
            (
                _,
                _,
                group_effective_sinr,
                group_tbler,
                group_bler,
            ) = phy_abstraction(
                mcs_index=group_mcs,
                sinr=illa_sinr,
                mcs_table_index=(
                    config.mcs_table_index
                ),
                mcs_category=(
                    config.mcs_category
                ),
            )
        group_mcs = (
            group_mcs.squeeze(0)
        )

        group_lowest_mcs = (
            group_lowest_mcs.squeeze(0)
        )

        group_effective_sinr = (
            group_effective_sinr.squeeze(0)
        )

        group_tbler = (
            group_tbler.squeeze(0)
        )

        group_bler = (
            group_bler.squeeze(0)
        )

        mcs_index[
            ue_indices
        ] = group_mcs

        lowest_available_mcs_index[
            ue_indices
        ] = group_lowest_mcs

        effective_sinr_linear[
            ue_indices
        ] = group_effective_sinr

        tbler[
            ue_indices
        ] = group_tbler

        bler[
            ue_indices
        ] = group_bler

    meets_bler_target = (
        tbler
        <= config.bler_target
    )

    return MUMIMORBGLinkAdaptationData(
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
    )




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

def _build_link_adaptation_blocks(
    *,
    bler_target: float,
    precision: str,
    device: str,
) -> tuple[
    PHYAbstraction,
    InnerLoopLinkAdaptation,
]:
    """
    Construct one Sionna link-adaptation runtime.

    This function deliberately contains the actual
    object construction so the cached public factory
    below can reuse the resulting stateless inference
    blocks.

    The MCS table/category are runtime inputs to the
    blocks and therefore do not belong to the object
    construction key.
    """

    phy_abstraction = PHYAbstraction(
        precision=precision,
        device=device,
    )

    illa = InnerLoopLinkAdaptation(
        phy_abstraction=phy_abstraction,
        bler_target=bler_target,
        precision=precision,
        device=device,
    )

    return (
        phy_abstraction,
        illa,
    )


@lru_cache(maxsize=16)
def _cached_link_adaptation_blocks(
    bler_target: float,
    precision: str,
    device: str,
) -> tuple[
    PHYAbstraction,
    InnerLoopLinkAdaptation,
]:
    """
    Reuse configuration-invariant Sionna
    link-adaptation inference objects.
    """

    return _build_link_adaptation_blocks(
        bler_target=bler_target,
        precision=precision,
        device=device,
    )


def create_link_adaptation_blocks(
    config: LinkAdaptationConfig,
) -> tuple[
    PHYAbstraction,
    InnerLoopLinkAdaptation,
]:
    """
    Return the persistent Sionna PHY-abstraction /
    ILLA runtime corresponding to this configuration.

    Reuse changes object lifetime only. SINR, MCS
    table, MCS category, BLER target and all PHY
    calculations remain unchanged.
    """

    return _cached_link_adaptation_blocks(
        float(config.bler_target),
        str(config.precision),
        str(config.device),
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


