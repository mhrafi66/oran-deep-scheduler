from dataclasses import dataclass

import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
    create_link_adaptation_blocks,
)
from oran_scheduler.phy.rate import (
    RateConfig,
    RateData,
    compute_rbg_rates,
)


@dataclass
class SingleUserLayerSINRData:
    """
    Vectorized SU-MIMO physical result.

    All layer tensors use:

        [batch,
         global_UE,
         RBG,
         generated_symbol,
         subcarrier_within_RBG,
         layer]

    layer_valid_mask:
        [batch, global_UE, RBG, layer]

    A rank-1 UE has:
        [True, False]

    A rank-2 UE has:
        [True, True]
    """

    layer_valid_mask: torch.Tensor

    mrc_combiner: torch.Tensor

    desired_power: torch.Tensor

    intra_stream_interference_power: torch.Tensor

    inter_cell_interference_power: torch.Tensor

    noise_power: torch.Tensor

    sinr_linear: torch.Tensor

@dataclass
class SingleUserLinkAdaptationData:
    """
    Per-UE/per-RBG link-adaptation result.

    Every tensor has shape:

        [batch, global_UE, RBG]
    """

    mcs_index: torch.Tensor

    lowest_available_mcs_index: torch.Tensor

    effective_sinr_linear: torch.Tensor

    tbler: torch.Tensor

    bler: torch.Tensor

    meets_bler_target: torch.Tensor

@dataclass
class SingleUserPHYReportData:
    """
    Complete SU-MIMO report used by TDS/FDS and RL state.

    sinr:
        vectorized physical layer result

    link_adaptation:
        MCS/TBLER per UE/RBG

    rate:
        physical rate per UE/RBG
    """

    sinr: SingleUserLayerSINRData

    link_adaptation: SingleUserLinkAdaptationData

    rate: RateData

def compute_single_user_layer_sinr(
    h_freq: torch.Tensor,
    serving_bs: torch.Tensor,
    recommended_rank: torch.Tensor,
    precoder_directions: torch.Tensor,
    num_rbgs: int,
    subcarriers_per_rbg: int,
    tx_power_per_subcarrier_w: float | torch.Tensor,
    noise_power_per_subcarrier_w: float | torch.Tensor,
) -> SingleUserLayerSINRData:
    """
    Evaluate every UE independently on every RBG.

    h_freq:
        [B, UE, RX, BS, TX, symbol, total_subcarrier]

    serving_bs:
        [B, UE]

    recommended_rank:
        [B, UE]

    precoder_directions:
        [B, UE, RBG, 2, TX]

        Current ideal-SVD/PMI-direction surrogate.

    Important:
        Each UE is evaluated as the sole serving-cell UE.

        Other cells remain active using the current isotropic
        inter-cell-interference approximation.
    """

    if h_freq.ndim != 7:
        raise ValueError(
            "h_freq must have shape "
            "[B, UE, RX, BS, TX, symbol, subcarrier]."
        )

    batch_size = h_freq.shape[0]
    num_ues = h_freq.shape[1]
    num_rx_ant = h_freq.shape[2]
    num_bs = h_freq.shape[3]
    num_tx_ant = h_freq.shape[4]
    num_symbols = h_freq.shape[5]
    num_subcarriers = h_freq.shape[6]

    if tuple(
        serving_bs.shape
    ) != (
        batch_size,
        num_ues,
    ):
        raise ValueError(
            "serving_bs must have shape [B, UE]."
        )

    if tuple(
        recommended_rank.shape
    ) != (
        batch_size,
        num_ues,
    ):
        raise ValueError(
            "recommended_rank must have shape [B, UE]."
        )

    expected_num_subcarriers = (
        num_rbgs
        * subcarriers_per_rbg
    )

    if num_subcarriers != expected_num_subcarriers:
        raise ValueError(
            "h_freq subcarrier count does not match "
            "num_rbgs * subcarriers_per_rbg."
        )

    expected_precoder_shape = (
        batch_size,
        num_ues,
        num_rbgs,
        2,
        num_tx_ant,
    )

    if tuple(
        precoder_directions.shape
    ) != expected_precoder_shape:
        raise ValueError(
            "precoder_directions must have shape "
            "[B, UE, RBG, 2, TX]."
        )

    if not torch.is_complex(
        h_freq
    ):
        raise ValueError(
            "h_freq must be complex-valued."
        )

    if not torch.is_complex(
        precoder_directions
    ):
        raise ValueError(
            "precoder_directions must be complex-valued."
        )

    if torch.any(
        (recommended_rank < 1)
        | (recommended_rank > 2)
    ):
        raise ValueError(
            "recommended_rank must contain only 1 or 2."
        )

    if torch.any(
        serving_bs < 0
    ) or torch.any(
        serving_bs >= num_bs
    ):
        raise ValueError(
            "serving_bs contains an invalid BS index."
        )


    all_bs_rbg_channel = h_freq.reshape(
        batch_size,
        num_ues,
        num_rx_ant,
        num_bs,
        num_tx_ant,
        num_symbols,
        num_rbgs,
        subcarriers_per_rbg,
    )

    all_bs_rbg_channel = (
        all_bs_rbg_channel.permute(
            0,
            1,
            6,
            5,
            7,
            3,
            2,
            4,
        )
        .contiguous()
    )

    serving_index = (
        serving_bs[
            :,
            :,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        .expand(
            batch_size,
            num_ues,
            num_rbgs,
            num_symbols,
            subcarriers_per_rbg,
            1,
            num_rx_ant,
            num_tx_ant,
        )
    )

    serving_channel = torch.gather(
        all_bs_rbg_channel,
        dim=5,
        index=serving_index,
    ).squeeze(
        dim=5
    )

    layer_numbers = torch.arange(
        2,
        dtype=torch.long,
        device=h_freq.device,
    )

    layer_valid_mask = (
        layer_numbers[
            None,
            None,
            None,
            :,
        ]
        < recommended_rank[
            :,
            :,
            None,
            None,
        ]
    )

    layer_valid_mask = (
        layer_valid_mask.expand(
            batch_size,
            num_ues,
            num_rbgs,
            2,
        )
    )

    beam_norm = torch.linalg.vector_norm(
        precoder_directions,
        dim=-1,
        keepdim=True,
    )

    valid_beam_norm = beam_norm[
        layer_valid_mask.unsqueeze(-1)
    ]

    if torch.any(
        valid_beam_norm <= 0
    ):
        raise ValueError(
            "A valid CSI precoder direction has zero norm."
        )

    safe_beam_norm = torch.where(
        layer_valid_mask.unsqueeze(-1),
        beam_norm,
        torch.ones_like(
            beam_norm
        ),
    )

    beams = (
        precoder_directions
        / safe_beam_norm
    )

    beams = torch.where(
        layer_valid_mask.unsqueeze(-1),
        beams,
        torch.zeros_like(
            beams
        ),
    )

    serving_precoded = torch.einsum(
        "bugsfrt,buglt->bugsfrl",
        serving_channel,
        beams,
    )

    desired_spatial_channel = (
        serving_precoded.movedim(
            -1,
            -2,
        )
    )

    desired_norm = torch.linalg.vector_norm(
        desired_spatial_channel,
        dim=-1,
        keepdim=True,
    )

    expanded_layer_valid = (
        layer_valid_mask[
            :,
            :,
            :,
            None,
            None,
            :,
        ]
    )

    valid_desired_norm = (
        desired_norm[
            expanded_layer_valid
            .unsqueeze(-1)
            .expand_as(
                desired_norm
            )
        ]
    )

    if torch.any(
        valid_desired_norm <= 0
    ):
        raise ValueError(
            "A valid desired precoded channel "
            "has zero norm."
        )

    safe_desired_norm = torch.where(
        expanded_layer_valid.unsqueeze(-1),
        desired_norm,
        torch.ones_like(
            desired_norm
        ),
    )

    mrc_combiner = (
        desired_spatial_channel
        / safe_desired_norm
    )

    mrc_combiner = torch.where(
        expanded_layer_valid.unsqueeze(-1),
        mrc_combiner,
        torch.zeros_like(
            mrc_combiner
        ),
    )

    combined_channel = torch.einsum(
        "bugsflr,bugsfrj->bugsflj",
        mrc_combiner.conj(),
        serving_precoded,
    )


    total_tx_power = torch.as_tensor(
        tx_power_per_subcarrier_w,
        dtype=h_freq.real.dtype,
        device=h_freq.device,
    )

    if torch.any(
        total_tx_power <= 0
    ):
        raise ValueError(
            "tx_power_per_subcarrier_w must be positive."
        )

    rank_by_rbg = (
        recommended_rank[
            :,
            :,
            None,
        ]
        .expand(
            batch_size,
            num_ues,
            num_rbgs,
        )
    )

    stream_power = (
        total_tx_power
        / rank_by_rbg.to(
            dtype=h_freq.real.dtype
        )
    )

    stream_power = (
        stream_power.unsqueeze(-1)
        * layer_valid_mask.to(
            dtype=h_freq.real.dtype
        )
    )

    combined_power = (
        torch.abs(
            combined_channel
        ) ** 2
    )

    combined_power = (
        combined_power
        * stream_power[
            :,
            :,
            :,
            None,
            None,
            None,
            :,
        ]
    )

    desired_power = torch.diagonal(
        combined_power,
        dim1=-2,
        dim2=-1,
    )

    total_serving_power = (
        combined_power.sum(
            dim=-1
        )
    )

    intra_stream_interference_power = (
        total_serving_power
        - desired_power
    )

    projected_all_bs = torch.einsum(
        "bugsflr,bugsfqrt->bugsflqt",
        mrc_combiner.conj(),
        all_bs_rbg_channel,
    )

    interfering_power_per_tx_port = (
        total_tx_power
        / float(
            num_tx_ant
        )
    )

    interference_by_bs = (
        torch.abs(
            projected_all_bs
        ) ** 2
    ).sum(
        dim=-1
    )

    interference_by_bs = (
        interference_by_bs
        * interfering_power_per_tx_port
    )

    bs_indices = torch.arange(
        num_bs,
        dtype=torch.long,
        device=h_freq.device,
    )

    interfering_bs_mask = (
        bs_indices[
            None,
            None,
            :,
        ]
        != serving_bs[
            :,
            :,
            None,
        ]
    )

    interference_by_bs = torch.where(
        interfering_bs_mask[
            :,
            :,
            None,
            None,
            None,
            None,
            :,
        ],
        interference_by_bs,
        torch.zeros_like(
            interference_by_bs
        ),
    )

    inter_cell_interference_power = (
        interference_by_bs.sum(
            dim=-1
        )
    )

    noise_power_scalar = torch.as_tensor(
        noise_power_per_subcarrier_w,
        dtype=h_freq.real.dtype,
        device=h_freq.device,
    )

    if torch.any(
        noise_power_scalar < 0
    ):
        raise ValueError(
            "noise_power_per_subcarrier_w "
            "cannot be negative."
        )

    noise_power = (
        noise_power_scalar
        * expanded_layer_valid.to(
            dtype=h_freq.real.dtype
        )
    )

    noise_power = torch.broadcast_to(
        noise_power,
        desired_power.shape,
    )

    denominator = (
        intra_stream_interference_power
        + inter_cell_interference_power
        + noise_power
    )

    valid_full_mask = (
        expanded_layer_valid.expand_as(
            desired_power
        )
    )

    safe_denominator = torch.where(
        valid_full_mask,
        denominator,
        torch.ones_like(
            denominator
        ),
    )

    if torch.any(
        safe_denominator[
            valid_full_mask
        ] <= 0
    ):
        raise ValueError(
            "A valid SINR denominator is non-positive."
        )

    sinr_linear = (
        desired_power
        / safe_denominator
    )

    sinr_linear = torch.where(
        valid_full_mask,
        sinr_linear,
        torch.zeros_like(
            sinr_linear
        ),
    )

    return SingleUserLayerSINRData(
        layer_valid_mask=layer_valid_mask,
        mrc_combiner=mrc_combiner,
        desired_power=desired_power,
        intra_stream_interference_power=(
            intra_stream_interference_power
        ),
        inter_cell_interference_power=(
            inter_cell_interference_power
        ),
        noise_power=noise_power,
        sinr_linear=sinr_linear,
    )


def select_single_user_mcs(
    layer_sinr_linear: torch.Tensor,
    recommended_rank: torch.Tensor,
    config: LinkAdaptationConfig,
) -> SingleUserLinkAdaptationData:
    """
    Run batched Sionna ILLA for every single-user/RBG case.

    layer_sinr_linear:
        [B, UE, RBG, generated_symbol, SC, 2]

    recommended_rank:
        [B, UE]

    Rank-1 and rank-2 cases are processed as two large
    batches instead of one Sionna call per UE/RBG.
    """

    if layer_sinr_linear.ndim != 6:
        raise ValueError(
            "layer_sinr_linear must have shape "
            "[B, UE, RBG, symbol, SC, layer]."
        )

    batch_size = layer_sinr_linear.shape[0]
    num_ues = layer_sinr_linear.shape[1]
    num_rbgs = layer_sinr_linear.shape[2]
    num_generated_symbols = (
        layer_sinr_linear.shape[3]
    )
    num_subcarriers = (
        layer_sinr_linear.shape[4]
    )

    if tuple(
        recommended_rank.shape
    ) != (
        batch_size,
        num_ues,
    ):
        raise ValueError(
            "recommended_rank must have shape [B, UE]."
        )

    if num_subcarriers != (
        config.subcarriers_per_rbg
    ):
        raise ValueError(
            "Unexpected number of RBG subcarriers."
        )

    if num_generated_symbols not in (
        1,
        config.num_slot_ofdm_symbols,
    ):
        raise ValueError(
            "Expected either one generated symbol "
            "or one complete slot."
        )

    rank_by_rbg = (
        recommended_rank[
            :,
            :,
            None,
        ]
        .expand(
            batch_size,
            num_ues,
            num_rbgs,
        )
    )

    flat_sinr = (
        layer_sinr_linear.reshape(
            -1,
            num_generated_symbols,
            num_subcarriers,
            2,
        )
    )

    flat_rank = rank_by_rbg.reshape(
        -1
    )

    num_cases = int(
        flat_rank.numel()
    )

    device = layer_sinr_linear.device
    real_dtype = layer_sinr_linear.dtype

    flat_mcs = torch.empty(
        num_cases,
        dtype=torch.long,
        device=device,
    )

    flat_lowest_mcs = torch.empty(
        num_cases,
        dtype=torch.long,
        device=device,
    )

    flat_effective_sinr = torch.empty(
        num_cases,
        dtype=real_dtype,
        device=device,
    )

    flat_tbler = torch.empty(
        num_cases,
        dtype=real_dtype,
        device=device,
    )

    flat_bler = torch.empty(
        num_cases,
        dtype=real_dtype,
        device=device,
    )

    phy_abstraction, illa = (
        create_link_adaptation_blocks(
            config
        )
    )

    for rank_value in (
        1,
        2,
    ):

        case_indices = torch.nonzero(
            flat_rank == rank_value,
            as_tuple=False,
        ).flatten()

        if case_indices.numel() == 0:
            continue

        group_sinr = (
            flat_sinr[
                case_indices,
                :,
                :,
                :rank_value,
            ]
        )

        if num_generated_symbols == 1:

            group_sinr = group_sinr.expand(
                -1,
                config.num_slot_ofdm_symbols,
                -1,
                -1,
            )

        illa_sinr = (
            group_sinr.unsqueeze(
                dim=-2
            )
        )

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
            group_mcs.squeeze(-1)
        )

        group_lowest_mcs = (
            group_lowest_mcs.squeeze(-1)
        )

        group_effective_sinr = (
            group_effective_sinr.squeeze(-1)
        )

        group_tbler = (
            group_tbler.squeeze(-1)
        )

        group_bler = (
            group_bler.squeeze(-1)
        )

        flat_mcs[
            case_indices
        ] = group_mcs

        flat_lowest_mcs[
            case_indices
        ] = group_lowest_mcs

        flat_effective_sinr[
            case_indices
        ] = group_effective_sinr

        flat_tbler[
            case_indices
        ] = group_tbler

        flat_bler[
            case_indices
        ] = group_bler

    output_shape = (
        batch_size,
        num_ues,
        num_rbgs,
    )

    mcs_index = flat_mcs.reshape(
        output_shape
    )

    lowest_available_mcs_index = (
        flat_lowest_mcs.reshape(
            output_shape
        )
    )

    effective_sinr_linear = (
        flat_effective_sinr.reshape(
            output_shape
        )
    )

    tbler = flat_tbler.reshape(
        output_shape
    )

    bler = flat_bler.reshape(
        output_shape
    )

    meets_bler_target = (
        tbler
        <= config.bler_target
    )

    return SingleUserLinkAdaptationData(
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

def build_single_user_phy_reports(
    h_freq: torch.Tensor,
    serving_bs: torch.Tensor,
    recommended_rank: torch.Tensor,
    precoder_directions: torch.Tensor,
    num_rbgs: int,
    subcarriers_per_rbg: int,
    tx_power_per_subcarrier_w: float | torch.Tensor,
    noise_power_per_subcarrier_w: float | torch.Tensor,
    link_adaptation_config: LinkAdaptationConfig,
    rate_config: RateConfig,
) -> SingleUserPHYReportData:
    """
    Generate all SU-MIMO physical scheduler reports in one
    vectorized pipeline.
    """

    sinr_data = compute_single_user_layer_sinr(
        h_freq=h_freq,
        serving_bs=serving_bs,
        recommended_rank=recommended_rank,
        precoder_directions=(
            precoder_directions
        ),
        num_rbgs=num_rbgs,
        subcarriers_per_rbg=(
            subcarriers_per_rbg
        ),
        tx_power_per_subcarrier_w=(
            tx_power_per_subcarrier_w
        ),
        noise_power_per_subcarrier_w=(
            noise_power_per_subcarrier_w
        ),
    )

    link_data = select_single_user_mcs(
        layer_sinr_linear=(
            sinr_data.sinr_linear
        ),
        recommended_rank=(
            recommended_rank
        ),
        config=link_adaptation_config,
    )

    num_rbgs_actual = (
        link_data.mcs_index.shape[-1]
    )

    stream_count = (
        recommended_rank[
            :,
            :,
            None,
        ]
        .expand(
            -1,
            -1,
            num_rbgs_actual,
        )
    )

    rate_data = compute_rbg_rates(
        mcs_index=link_data.mcs_index,
        tbler=link_data.tbler,
        config=rate_config,
        num_streams_per_ue=(
            stream_count
        ),
    )

    return SingleUserPHYReportData(
        sinr=sinr_data,
        link_adaptation=link_data,
        rate=rate_data,
    )



    