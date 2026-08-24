import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
    build_layer_lookup,
)
from oran_scheduler.phy.mu_mimo_rate import (
    compute_mu_mimo_rbg_rates,
)
from oran_scheduler.phy.rate import (
    RateConfig,
    compute_rbg_rates,
)


def test_build_layer_lookup_mixed_rank():
    device = "cuda:0"

    layer_ue_indices = torch.tensor(
        [0, 0, 1, 2],
        dtype=torch.long,
        device=device,
    )

    layer_index_within_ue = torch.tensor(
        [0, 1, 0, 0],
        dtype=torch.long,
        device=device,
    )

    ranks = torch.tensor(
        [2, 1, 1],
        dtype=torch.long,
        device=device,
    )

    lookup = build_layer_lookup(
        layer_ue_indices=layer_ue_indices,
        layer_index_within_ue=(
            layer_index_within_ue
        ),
        selected_ranks=ranks,
    )

    expected = torch.tensor(
        [
            [0, 1],
            [2, -1],
            [3, -1],
        ],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        lookup,
        expected,
    )

def test_rate_uses_rank_dependent_stream_count():
    device = "cuda:0"

    mcs = torch.tensor(
        [
            [
                [10],
                [10],
            ]
        ],
        dtype=torch.int32,
        device=device,
    )

    tbler = torch.zeros(
        (
            1,
            2,
            1,
        ),
        device=device,
    )

    ranks = torch.tensor(
        [
            [
                [1],
                [2],
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    config = RateConfig(
        device=device,
    )

    result = compute_rbg_rates(
        mcs_index=mcs,
        tbler=tbler,
        config=config,
        num_streams_per_ue=ranks,
    )

    rank1_tb_size = int(
        result.tb_size_bits[
            0,
            0,
            0,
        ].item()
    )

    rank2_tb_size = int(
        result.tb_size_bits[
            0,
            1,
            0,
        ].item()
    )

    assert rank2_tb_size > rank1_tb_size

def test_mu_mimo_rbg_rate_mixed_rank():
    device = "cuda:0"

    ranks = torch.tensor(
        [2, 1],
        dtype=torch.long,
        device=device,
    )

    layer_ue_indices = torch.tensor(
        [0, 0, 1],
        dtype=torch.long,
        device=device,
    )

    layer_index_within_ue = torch.tensor(
        [0, 1, 0],
        dtype=torch.long,
        device=device,
    )

    layer_sinr = torch.full(
        (
            1,
            12,
            3,
        ),
        fill_value=100.0,
        dtype=torch.float32,
        device=device,
    )

    link_config = LinkAdaptationConfig(
        device=device,
    )

    rate_config = RateConfig(
        device=device,
    )

    result = compute_mu_mimo_rbg_rates(
        layer_sinr_linear=layer_sinr,
        layer_ue_indices=layer_ue_indices,
        layer_index_within_ue=(
            layer_index_within_ue
        ),
        selected_ranks=ranks,
        link_adaptation_config=(
            link_config
        ),
        rate_config=rate_config,
    )

    assert result.ue_rate_bps.shape == (
        2,
    )

    assert (
        result.target_compliant_ue_rate_bps.shape
        == (2,)
    )

    assert (
        result.link_adaptation.mcs_index.shape
        == (2,)
    )

    assert torch.all(
        result.target_compliant_ue_rate_bps
        > 0
    )

    torch.testing.assert_close(
        result.total_target_compliant_rate_bps,
        result.target_compliant_ue_rate_bps.sum(),
    )

    assert (
        result.ue_rate_bps[0]
        > result.ue_rate_bps[1]
    )