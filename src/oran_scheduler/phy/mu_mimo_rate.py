from dataclasses import dataclass

import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
    MUMIMORBGLinkAdaptationData,
    select_mu_mimo_rbg_mcs,
)
from oran_scheduler.phy.rate import (
    RateConfig,
    RateData,
    compute_rbg_rates,
)


@dataclass
class MUMIMORBGRateData:
    """
    Physical rate result for selected UEs on one RBG.

    ue_rate_bps:
        [selected_UE]

    expected_ue_goodput_bps:
        [selected_UE]

    target_compliant_ue_rate_bps:
        [selected_UE]

    total_target_compliant_rate_bps:
        scalar
    """

    link_adaptation: (
        MUMIMORBGLinkAdaptationData
    )

    rate: RateData

    ue_rate_bps: torch.Tensor

    expected_ue_goodput_bps: torch.Tensor

    target_compliant_ue_rate_bps: torch.Tensor

    total_target_compliant_rate_bps: torch.Tensor

def compute_mu_mimo_rbg_rates(
    layer_sinr_linear: torch.Tensor,
    layer_ue_indices: torch.Tensor,
    layer_index_within_ue: torch.Tensor,
    selected_ranks: torch.Tensor,
    link_adaptation_config: LinkAdaptationConfig,
    rate_config: RateConfig,
) -> MUMIMORBGRateData:
    """
    Convert layer-level MU-MIMO SINR into one physical
    rate per selected UE.

    Multiple layers belonging to the same UE are treated
    as streams of one transport block.
    """

    link_data = select_mu_mimo_rbg_mcs(
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

    mcs_for_rate = (
        link_data.mcs_index
        .reshape(
            1,
            -1,
            1,
        )
    )

    tbler_for_rate = (
        link_data.tbler
        .reshape(
            1,
            -1,
            1,
        )
    )

    streams_for_rate = (
        selected_ranks
        .reshape(
            1,
            -1,
            1,
        )
    )

    rate_data = compute_rbg_rates(
        mcs_index=mcs_for_rate,
        tbler=tbler_for_rate,
        config=rate_config,
        num_streams_per_ue=(
            streams_for_rate
        ),
    )

    ue_rate_bps = (
        rate_data.nominal_rate_bps[
            0,
            :,
            0,
        ]
    )

    expected_ue_goodput_bps = (
        rate_data.expected_goodput_bps[
            0,
            :,
            0,
        ]
    )

    target_compliant_ue_rate_bps = (
        rate_data.target_compliant_rate_bps[
            0,
            :,
            0,
        ]
    )

    total_target_compliant_rate_bps = (
        target_compliant_ue_rate_bps.sum()
    )

    return MUMIMORBGRateData(
        link_adaptation=link_data,
        rate=rate_data,
        ue_rate_bps=ue_rate_bps,
        expected_ue_goodput_bps=(
            expected_ue_goodput_bps
        ),
        target_compliant_ue_rate_bps=(
            target_compliant_ue_rate_bps
        ),
        total_target_compliant_rate_bps=(
            total_target_compliant_rate_bps
        ),
    )
