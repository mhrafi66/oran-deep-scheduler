from dataclasses import dataclass

import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
    build_layer_lookup,
    create_link_adaptation_blocks,
)
from oran_scheduler.phy.mu_mimo import (
    compute_rzf_matrix,
)
from oran_scheduler.phy.rate import (
    RateConfig,
    compute_rbg_rates,
)


@dataclass(frozen=True)
class BatchedCandidateHypothesisData:
    """
    Physical scores for multiple hypothetical
    candidate sets evaluated in parallel.

    All hypotheses in one call must have the same
    selected-UE rank pattern.

    Example:

        hypothesis 0: candidate (1, 5, 7)
                      ranks     (1, 2, 1)

        hypothesis 1: candidate (3, 4, 9)
                      ranks     (1, 2, 1)

    These can share one rectangular GPU batch.

    selected_candidate_indices:
        [hypothesis, selected_UE]

    rbg_indices:
        [hypothesis]

    selected_ranks:
        [selected_UE]

    target_compliant_rate_bps:
        [hypothesis, selected_UE]

    total_target_compliant_rate_bps:
        [hypothesis]

    rzf_alpha:
        [hypothesis]
    """

    selected_candidate_indices: torch.Tensor

    rbg_indices: torch.Tensor

    selected_ranks: torch.Tensor

    target_compliant_rate_bps: torch.Tensor

    total_target_compliant_rate_bps: torch.Tensor

    rzf_alpha: torch.Tensor


def _build_layer_structure(
    selected_ranks: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Expand UE ranks into physical-layer ownership.

    Example:

        ranks = [2, 1]

    gives:

        layer UE:
            [0, 0, 1]

        layer within UE:
            [0, 1, 0]
    """

    possible_layers = torch.arange(
        2,
        dtype=torch.long,
        device=selected_ranks.device,
    )

    valid_layer_mask = (
        possible_layers.unsqueeze(0)
        < selected_ranks.unsqueeze(1)
    )

    layer_pairs = torch.nonzero(
        valid_layer_mask,
        as_tuple=False,
    )

    return (
        layer_pairs[:, 0],
        layer_pairs[:, 1],
    )


def _estimate_batched_rzf_alpha(
    *,
    selected_ranks: torch.Tensor,
    selected_rx_combiners: torch.Tensor,
    selected_inter_cell_covariance: torch.Tensor,
    total_tx_power_w: float | torch.Tensor,
    noise_power_w: float | torch.Tensor,
) -> torch.Tensor:
    """
    Batched equivalent of estimate_rzf_alpha().

    selected_rx_combiners:
        [H, UE, 2, RX]

    selected_inter_cell_covariance:
        [H, UE, symbol, subcarrier, RX, RX]

    output:
        [H]
    """

    (
        layer_ue_indices,
        layer_index_within_ue,
    ) = _build_layer_structure(
        selected_ranks
    )

    layer_combiners = (
        selected_rx_combiners[
            :,
            layer_ue_indices,
            layer_index_within_ue,
            :,
        ]
    )

    combiner_norm = torch.linalg.vector_norm(
        layer_combiners,
        dim=-1,
        keepdim=True,
    )

    if torch.any(
        combiner_norm <= 0
    ):
        raise ValueError(
            "A receive combiner has zero norm."
        )

    layer_combiners = (
        layer_combiners
        / combiner_norm
    )

    layer_covariance = (
        selected_inter_cell_covariance[
            :,
            layer_ue_indices,
            :,
            :,
            :,
            :,
        ]
    )

    projected_interference = torch.einsum(
        "hlr,hlscrq,hlq->hlsc",
        layer_combiners.conj(),
        layer_covariance,
        layer_combiners,
    ).real

    projected_interference = torch.clamp(
        projected_interference,
        min=0.0,
    )

    #
    # Scalar implementation does:
    #
    #     mean across layer, symbol and subcarrier.
    #
    # Here that mean is retained independently for
    # every hypothesis.
    #
    mean_interference = (
        projected_interference.mean(
            dim=(
                1,
                2,
                3,
            )
        )
    )

    tx_power = torch.as_tensor(
        total_tx_power_w,
        dtype=mean_interference.dtype,
        device=mean_interference.device,
    )

    noise_power = torch.as_tensor(
        noise_power_w,
        dtype=mean_interference.dtype,
        device=mean_interference.device,
    )

    if torch.any(
        tx_power <= 0
    ):
        raise ValueError(
            "total_tx_power_w must be positive."
        )

    if torch.any(
        noise_power < 0
    ):
        raise ValueError(
            "noise_power_w cannot be negative."
        )

    num_physical_layers = int(
        selected_ranks.sum().item()
    )

    return (
        float(num_physical_layers)
        * (
            mean_interference
            + noise_power
        )
        / tx_power
    )


def _batched_link_adaptation(
    *,
    layer_sinr_linear: torch.Tensor,
    layer_ue_indices: torch.Tensor,
    layer_index_within_ue: torch.Tensor,
    selected_ranks: torch.Tensor,
    config: LinkAdaptationConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Run Sionna ILLA for multiple PHY hypotheses.

    layer_sinr_linear:
        [H, symbol, subcarrier, layer]

    Returns:

        mcs_index:
            [H, UE]

        tbler:
            [H, UE]

    Rank-1 and rank-2 UEs are still separated exactly
    as in select_mu_mimo_rbg_mcs(). The new leading H
    dimension is simply a Sionna batch dimension.
    """

    if layer_sinr_linear.ndim != 4:
        raise ValueError(
            "layer_sinr_linear must have shape "
            "[hypothesis, symbol, subcarrier, layer]."
        )

    num_hypotheses = int(
        layer_sinr_linear.shape[0]
    )

    num_selected_ues = int(
        selected_ranks.numel()
    )

    layer_lookup = build_layer_lookup(
        layer_ue_indices=(
            layer_ue_indices
        ),
        layer_index_within_ue=(
            layer_index_within_ue
        ),
        selected_ranks=selected_ranks,
    )

    device = layer_sinr_linear.device

    mcs_index = torch.empty(
        (
            num_hypotheses,
            num_selected_ues,
        ),
        dtype=torch.long,
        device=device,
    )

    tbler = torch.empty(
        (
            num_hypotheses,
            num_selected_ues,
        ),
        dtype=layer_sinr_linear.dtype,
        device=device,
    )

    #
    # Important optimization:
    #
    # Before:
    #     construct these blocks for every hypothetical
    #     candidate-set evaluation.
    #
    # Now:
    #     construct once for the entire batch.
    #
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

        #
        # [H, symbol, subcarrier, UE, stream]
        #
        group_sinr = (
            layer_sinr_linear[
                ...,
                group_layer_indices,
            ]
        )

        if group_sinr.shape[1] == 1:
            group_sinr = (
                group_sinr.expand(
                    -1,
                    config.num_slot_ofdm_symbols,
                    -1,
                    -1,
                    -1,
                )
            )

        (
            group_mcs,
            _,
        ) = illa(
            sinr=group_sinr,
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
            _,
            group_tbler,
            _,
        ) = phy_abstraction(
            mcs_index=group_mcs,
            sinr=group_sinr,
            mcs_table_index=(
                config.mcs_table_index
            ),
            mcs_category=(
                config.mcs_category
            ),
        )

        mcs_index[
            :,
            ue_indices,
        ] = group_mcs

        tbler[
            :,
            ue_indices,
        ] = group_tbler

    return (
        mcs_index,
        tbler,
    )


def evaluate_candidate_hypothesis_batch_same_rank_pattern(
    *,
    selected_candidate_indices: torch.Tensor,
    rbg_indices: torch.Tensor,
    candidate_ranks: torch.Tensor,
    candidate_rx_combiners: torch.Tensor,
    candidate_serving_channel: torch.Tensor,
    candidate_inter_cell_covariance: torch.Tensor,
    csi_subcarrier_index: int,
    subcarriers_per_rbg: int,
    tx_power_per_subcarrier_w: float | torch.Tensor,
    noise_power_per_subcarrier_w: float | torch.Tensor,
    link_adaptation_config: LinkAdaptationConfig,
    rate_config: RateConfig,
) -> BatchedCandidateHypothesisData:
    """
    Physically evaluate multiple counterfactual scheduler
    hypotheses in parallel.

    OPEN-REPRODUCTION ENGINEERING OPTIMIZATION.

    The PHY model is unchanged:

        candidate set
            ->
        rank expansion
            ->
        RZF
            ->
        post-RZF MRC
            ->
        SINR
            ->
        ILLA / TBLER
            ->
        target-compliant rate

    What changes is only execution layout:

        OLD:
            hypothesis 0 -> GPU -> wait
            hypothesis 1 -> GPU -> wait
            hypothesis 2 -> GPU -> wait
            ...

        NEW:
            [hypothesis 0
             hypothesis 1
             hypothesis 2
             ...] -> GPU together

    All hypotheses in one invocation must have the same
    selected rank pattern so that their physical layer
    dimension is rectangular.
    """

    if (
        selected_candidate_indices.ndim
        != 2
    ):
        raise ValueError(
            "selected_candidate_indices must have shape "
            "[hypothesis, selected_UE]."
        )

    num_hypotheses = int(
        selected_candidate_indices.shape[0]
    )

    num_selected_ues = int(
        selected_candidate_indices.shape[1]
    )

    if num_hypotheses < 1:
        raise ValueError(
            "At least one hypothesis is required."
        )

    if num_selected_ues < 1:
        raise ValueError(
            "At least one selected UE is required."
        )

    if tuple(
        rbg_indices.shape
    ) != (
        num_hypotheses,
    ):
        raise ValueError(
            "rbg_indices must have shape "
            "[hypothesis]."
        )

    if candidate_ranks.ndim != 1:
        raise ValueError(
            "candidate_ranks must have shape "
            "[candidate]."
        )

    if candidate_rx_combiners.ndim != 4:
        raise ValueError(
            "candidate_rx_combiners must have shape "
            "[candidate, RBG, 2, RX]."
        )

    if candidate_serving_channel.ndim != 5:
        raise ValueError(
            "candidate_serving_channel must have shape "
            "[candidate, symbol, subcarrier, RX, TX]."
        )

    if (
        candidate_inter_cell_covariance.ndim
        != 5
    ):
        raise ValueError(
            "candidate_inter_cell_covariance must "
            "have shape "
            "[candidate, symbol, subcarrier, RX, RX]."
        )

    device = (
        candidate_serving_channel.device
    )

    if (
        selected_candidate_indices.device
        != device
    ):
        raise ValueError(
            "Candidate hypotheses are on the "
            "wrong device."
        )

    if rbg_indices.device != device:
        raise ValueError(
            "RBG indices are on the wrong device."
        )

    if torch.any(
        selected_candidate_indices < 0
    ):
        raise ValueError(
            "Selected candidate index cannot "
            "be negative."
        )

    num_candidates = int(
        candidate_ranks.numel()
    )

    if torch.any(
        selected_candidate_indices
        >= num_candidates
    ):
        raise ValueError(
            "Selected candidate index is invalid."
        )

    if subcarriers_per_rbg <= 0:
        raise ValueError(
            "subcarriers_per_rbg must be positive."
        )

    num_symbols = int(
        candidate_serving_channel.shape[1]
    )

    num_subcarriers = int(
        candidate_serving_channel.shape[2]
    )

    num_rx_ant = int(
        candidate_serving_channel.shape[-2]
    )

    num_tx_ant = int(
        candidate_serving_channel.shape[-1]
    )

    if (
        num_subcarriers
        % subcarriers_per_rbg
        != 0
    ):
        raise ValueError(
            "Subcarrier count is not divisible "
            "by the RBG size."
        )

    num_rbgs = (
        num_subcarriers
        // subcarriers_per_rbg
    )

    if torch.any(
        rbg_indices < 0
    ) or torch.any(
        rbg_indices >= num_rbgs
    ):
        raise ValueError(
            "RBG index is invalid."
        )

    #
    # Convert candidate-wide storage into:
    #
    #     [candidate, RBG, symbol, SC, RX, TX]
    #
    serving_by_rbg = (
        candidate_serving_channel
        .reshape(
            num_candidates,
            num_symbols,
            num_rbgs,
            subcarriers_per_rbg,
            num_rx_ant,
            num_tx_ant,
        )
        .permute(
            0,
            2,
            1,
            3,
            4,
            5,
        )
    )

    covariance_by_rbg = (
        candidate_inter_cell_covariance
        .reshape(
            num_candidates,
            num_symbols,
            num_rbgs,
            subcarriers_per_rbg,
            num_rx_ant,
            num_rx_ant,
        )
        .permute(
            0,
            2,
            1,
            3,
            4,
            5,
        )
    )

    rbg_column = (
        rbg_indices.unsqueeze(1)
    )

    #
    # Advanced indexing produces:
    #
    #     [H, selected_UE, symbol, SC, RX, TX]
    #
    selected_serving_channel = (
        serving_by_rbg[
            selected_candidate_indices,
            rbg_column,
        ]
    )

    selected_inter_cell_covariance = (
        covariance_by_rbg[
            selected_candidate_indices,
            rbg_column,
        ]
    )

    selected_ranks_by_hypothesis = (
        candidate_ranks[
            selected_candidate_indices
        ]
    )

    selected_ranks = (
        selected_ranks_by_hypothesis[0]
    )

    if not torch.all(
        selected_ranks_by_hypothesis
        == selected_ranks.unsqueeze(0)
    ):
        raise ValueError(
            "Every hypothesis in one GPU batch "
            "must share the same rank pattern."
        )

    if torch.any(
        (selected_ranks < 1)
        | (selected_ranks > 2)
    ):
        raise ValueError(
            "Selected UE ranks must be 1 or 2."
        )

    #
    # [H, selected_UE, 2, RX]
    #
    selected_rx_combiners = (
        candidate_rx_combiners[
            selected_candidate_indices,
            rbg_column,
        ]
    )

    rzf_alpha = (
        _estimate_batched_rzf_alpha(
            selected_ranks=(
                selected_ranks
            ),
            selected_rx_combiners=(
                selected_rx_combiners
            ),
            selected_inter_cell_covariance=(
                selected_inter_cell_covariance
            ),
            total_tx_power_w=(
                tx_power_per_subcarrier_w
            ),
            noise_power_w=(
                noise_power_per_subcarrier_w
            ),
        )
    )

    (
        layer_ue_indices,
        layer_index_within_ue,
    ) = _build_layer_structure(
        selected_ranks
    )

    num_layers = int(
        layer_ue_indices.numel()
    )

    #
    # CSI-side receive direction used for RZF design.
    #
    # [H, layer, RX]
    #
    layer_rx_combiner = (
        selected_rx_combiners[
            :,
            layer_ue_indices,
            layer_index_within_ue,
            :,
        ]
    )

    if not (
        0
        <= csi_subcarrier_index
        < subcarriers_per_rbg
    ):
        raise ValueError(
            "csi_subcarrier_index is invalid."
        )

    #
    # [H, UE, RX, TX]
    #
    csi_channel = (
        selected_serving_channel[
            :,
            :,
            0,
            csi_subcarrier_index,
            :,
            :,
        ]
    )

    #
    # [H, layer, RX, TX]
    #
    layer_csi_channel = (
        csi_channel[
            :,
            layer_ue_indices,
            :,
            :,
        ]
    )

    #
    # [H, layer, TX]
    #
    effective_csi_channel = torch.einsum(
        "hlr,hlrt->hlt",
        layer_rx_combiner.conj(),
        layer_csi_channel,
    )

    #
    # THIS IS THE IMPORTANT GPU BATCH.
    #
    # Input:
    #     [H, layer, TX]
    #
    # alpha:
    #     [H]
    #
    # Output:
    #     [H, TX, layer]
    #
    precoding_matrix = compute_rzf_matrix(
        effective_channel=(
            effective_csi_channel
        ),
        alpha=rzf_alpha,
    )

    #
    # Physical channel:
    #
    #     [H, symbol, SC, layer, RX, TX]
    #
    layer_physical_channel = (
        selected_serving_channel[
            :,
            layer_ue_indices,
            :,
            :,
            :,
            :,
        ]
        .permute(
            0,
            2,
            3,
            1,
            4,
            5,
        )
        .contiguous()
    )

    layer_inter_cell_covariance = (
        selected_inter_cell_covariance[
            :,
            layer_ue_indices,
            :,
            :,
            :,
            :,
        ]
        .permute(
            0,
            2,
            3,
            1,
            4,
            5,
        )
        .contiguous()
    )

    #
    # Expand each hypothesis's RZF matrix over
    # symbols, subcarriers, and physical layer-owner.
    #
    precoder_for_matmul = (
        precoding_matrix[
            :,
            None,
            None,
            None,
            :,
            :,
        ]
    )

    #
    # [H, symbol, SC, layer, RX, transmit_layer]
    #
    precoded_spatial_channel = (
        torch.matmul(
            layer_physical_channel,
            precoder_for_matmul,
        )
    )

    desired_spatial_channel = (
        torch.diagonal(
            precoded_spatial_channel,
            dim1=-3,
            dim2=-1,
        )
        .movedim(
            -1,
            -2,
        )
    )

    desired_norm = (
        torch.linalg.vector_norm(
            desired_spatial_channel,
            dim=-1,
            keepdim=True,
        )
    )

    if torch.any(
        desired_norm <= 0
    ):
        raise ValueError(
            "A desired post-precoder channel "
            "has zero norm."
        )

    #
    # PAPER-SPECIFIED receiver: MRC.
    #
    mrc_combiner = (
        desired_spatial_channel
        / desired_norm
    )

    combined_channel = torch.einsum(
        "...lr,...lrj->...lj",
        mrc_combiner.conj(),
        precoded_spatial_channel,
    )

    tx_power = torch.as_tensor(
        tx_power_per_subcarrier_w,
        dtype=combined_channel.real.dtype,
        device=device,
    )

    if torch.any(
        tx_power <= 0
    ):
        raise ValueError(
            "Transmit power must be positive."
        )

    per_layer_power = (
        tx_power
        / float(num_layers)
    )

    combined_power = (
        torch.abs(
            combined_channel
        )
        ** 2
    )

    combined_power = (
        combined_power
        * per_layer_power
    )

    desired_power = torch.diagonal(
        combined_power,
        dim1=-2,
        dim2=-1,
    )

    intra_cell_interference_power = (
        combined_power.sum(
            dim=-1
        )
        - desired_power
    )

    inter_cell_interference_power = (
        torch.einsum(
            "...lr,...lrs,...ls->...l",
            mrc_combiner.conj(),
            layer_inter_cell_covariance,
            mrc_combiner,
        )
        .real
    )

    inter_cell_interference_power = (
        torch.clamp(
            inter_cell_interference_power,
            min=0.0,
        )
    )

    noise_power = torch.as_tensor(
        noise_power_per_subcarrier_w,
        dtype=desired_power.dtype,
        device=device,
    )

    denominator = (
        intra_cell_interference_power
        + inter_cell_interference_power
        + noise_power
    )

    if torch.any(
        denominator <= 0
    ):
        raise ValueError(
            "SINR denominator must be positive."
        )

    #
    # [H, symbol, SC, layer]
    #
    layer_sinr_linear = (
        desired_power
        / denominator
    )

    (
        mcs_index,
        tbler,
    ) = _batched_link_adaptation(
        layer_sinr_linear=(
            layer_sinr_linear
        ),
        layer_ue_indices=(
            layer_ue_indices
        ),
        layer_index_within_ue=(
            layer_index_within_ue
        ),
        selected_ranks=selected_ranks,
        config=link_adaptation_config,
    )

    #
    # Existing rate function already supports a real
    # leading batch dimension.
    #
    streams_for_rate = (
        selected_ranks
        .reshape(
            1,
            num_selected_ues,
            1,
        )
        .expand(
            num_hypotheses,
            -1,
            -1,
        )
    )

    rate_data = compute_rbg_rates(
        mcs_index=(
            mcs_index.unsqueeze(-1)
        ),
        tbler=(
            tbler.unsqueeze(-1)
        ),
        config=rate_config,
        num_streams_per_ue=(
            streams_for_rate
        ),
    )

    target_compliant_rate_bps = (
        rate_data
        .target_compliant_rate_bps[
            :,
            :,
            0,
        ]
    )

    total_target_compliant_rate_bps = (
        target_compliant_rate_bps.sum(
            dim=-1
        )
    )

    return BatchedCandidateHypothesisData(
        selected_candidate_indices=(
            selected_candidate_indices
        ),
        rbg_indices=rbg_indices,
        selected_ranks=selected_ranks,
        target_compliant_rate_bps=(
            target_compliant_rate_bps
        ),
        total_target_compliant_rate_bps=(
            total_target_compliant_rate_bps
        ),
        rzf_alpha=rzf_alpha,
    )



