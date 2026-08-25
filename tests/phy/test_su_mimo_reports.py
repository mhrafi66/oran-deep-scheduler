import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
)
from oran_scheduler.phy.rate import (
    RateConfig,
)
from oran_scheduler.phy.su_mimo_reports import (
    build_single_user_phy_reports,
    compute_single_user_layer_sinr,
)


def test_rank1_single_user_sinr():
    device = "cuda:0"

    h_freq = torch.zeros(
        (
            1,   # batch
            1,   # UE
            1,   # RX
            1,   # BS
            2,   # TX
            1,   # symbol
            12,  # one RBG
        ),
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0,
        0,
        0,
        0,
        0,
        0,
        :,
    ] = 1.0


    directions = torch.zeros(
        (
            1,
            1,
            1,
            2,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        0,
        0,
        0,
        0,
        0,
    ] = 1.0

    serving_bs = torch.tensor(
        [[0]],
        dtype=torch.long,
        device=device,
    )

    rank = torch.tensor(
        [[1]],
        dtype=torch.long,
        device=device,
    )

    result = compute_single_user_layer_sinr(
        h_freq=h_freq,
        serving_bs=serving_bs,
        recommended_rank=rank,
        precoder_directions=directions,
        num_rbgs=1,
        subcarriers_per_rbg=12,
        tx_power_per_subcarrier_w=1.0,
        noise_power_per_subcarrier_w=1.0,
    )

    expected = torch.ones(
        (
            1,
            1,
            1,
            1,
            12,
        ),
        device=device,
    )

    torch.testing.assert_close(
        result.sinr_linear[
            ...,
            0,
        ],
        expected,
    )

    torch.testing.assert_close(
        result.sinr_linear[
            ...,
            1,
        ],
        torch.zeros_like(
            result.sinr_linear[
                ...,
                1,
            ]
        ),
    )

def test_rank2_orthogonal_single_user_streams():
    device = "cuda:0"

    h_freq = torch.zeros(
        (
            1,
            1,
            2,
            1,
            2,
            1,
            12,
        ),
        dtype=torch.complex64,
        device=device,
    )


    h_freq[
        0,
        0,
        0,
        0,
        0,
        0,
        :,
    ] = 2.0

    h_freq[
        0,
        0,
        1,
        0,
        1,
        0,
        :,
    ] = 1.0

    directions = torch.zeros(
        (
            1,
            1,
            1,
            2,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        0,
        0,
        0,
        0,
        0,
    ] = 1.0

    directions[
        0,
        0,
        0,
        1,
        1,
    ] = 1.0

    rank = torch.tensor(
        [[2]],
        dtype=torch.long,
        device=device,
    )

    serving_bs = torch.tensor(
        [[0]],
        dtype=torch.long,
        device=device,
    )


    result = compute_single_user_layer_sinr(
        h_freq=h_freq,
        serving_bs=serving_bs,
        recommended_rank=rank,
        precoder_directions=directions,
        num_rbgs=1,
        subcarriers_per_rbg=12,
        tx_power_per_subcarrier_w=2.0,
        noise_power_per_subcarrier_w=1.0,
    )

    expected_layer_0 = torch.full(
        (
            1,
            1,
            1,
            1,
            12,
        ),
        4.0,
        device=device,
    )

    expected_layer_1 = torch.ones(
        (
            1,
            1,
            1,
            1,
            12,
        ),
        device=device,
    )

    torch.testing.assert_close(
        result.sinr_linear[
            ...,
            0,
        ],
        expected_layer_0,
    )

    torch.testing.assert_close(
        result.sinr_linear[
            ...,
            1,
        ],
        expected_layer_1,
    )

    torch.testing.assert_close(
        result.intra_stream_interference_power,
        torch.zeros_like(
            result.intra_stream_interference_power
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

def test_single_user_inter_cell_interference():
    device = "cuda:0"

    h_freq = torch.zeros(
        (
            1,
            1,
            1,
            2,   # two BSs
            2,
            1,
            12,
        ),
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0,
        0,
        0,
        0,
        0,
        0,
        :,
    ] = 1.0

    h_freq[
        0,
        0,
        0,
        1,
        0,
        0,
        :,
    ] = 1.0

    directions = torch.zeros(
        (
            1,
            1,
            1,
            2,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        0,
        0,
        0,
        0,
        0,
    ] = 1.0

    result = compute_single_user_layer_sinr(
        h_freq=h_freq,
        serving_bs=torch.tensor(
            [[0]],
            dtype=torch.long,
            device=device,
        ),
        recommended_rank=torch.tensor(
            [[1]],
            dtype=torch.long,
            device=device,
        ),
        precoder_directions=directions,
        num_rbgs=1,
        subcarriers_per_rbg=12,
        tx_power_per_subcarrier_w=2.0,
        noise_power_per_subcarrier_w=1.0,
    )

    expected_interference = torch.ones(
        (
            1,
            1,
            1,
            1,
            12,
        ),
        device=device,
    )

    torch.testing.assert_close(
        result.inter_cell_interference_power[
            ...,
            0,
        ],
        expected_interference,
    )

    torch.testing.assert_close(
        result.sinr_linear[
            ...,
            0,
        ],
        torch.ones_like(
            result.sinr_linear[
                ...,
                0,
            ]
        ),
    )

def test_complete_single_user_phy_reports():
    device = "cuda:0"

    num_ues = 2
    num_rbgs = 2

    h_freq = torch.zeros(
        (
            1,
            num_ues,
            2,
            1,
            2,
            1,
            num_rbgs * 12,
        ),
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0,
        0,
        0,
        0,
        0,
        0,
        :,
    ] = 10.0

    h_freq[
        0,
        1,
        0,
        0,
        0,
        0,
        :,
    ] = 10.0

    h_freq[
        0,
        1,
        1,
        0,
        1,
        0,
        :,
    ] = 10.0

    directions = torch.zeros(
        (
            1,
            num_ues,
            num_rbgs,
            2,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    directions[
        :,
        :,
        :,
        0,
        0,
    ] = 1.0

    directions[
        0,
        1,
        :,
        1,
        1,
    ] = 1.0

    ranks = torch.tensor(
        [
            [1, 2]
        ],
        dtype=torch.long,
        device=device,
    )

    result = build_single_user_phy_reports(
        h_freq=h_freq,
        serving_bs=torch.zeros(
            (
                1,
                num_ues,
            ),
            dtype=torch.long,
            device=device,
        ),
        recommended_rank=ranks,
        precoder_directions=directions,
        num_rbgs=num_rbgs,
        subcarriers_per_rbg=12,
        tx_power_per_subcarrier_w=10.0,
        noise_power_per_subcarrier_w=1.0,
        link_adaptation_config=(
            LinkAdaptationConfig(
                num_rbgs=num_rbgs,
                device=device,
            )
        ),
        rate_config=RateConfig(
            device=device,
        ),
    )

    assert result.sinr.sinr_linear.shape == (
        1,
        2,
        2,
        1,
        12,
        2,
    )

    assert (
        result.link_adaptation.mcs_index.shape
        == (
            1,
            2,
            2,
        )
    )

    assert result.rate.nominal_rate_bps.shape == (
        1,
        2,
        2,
    )

    assert torch.all(
        result.rate.target_compliant_rate_bps
        > 0
    )

    assert torch.isfinite(
        result.rate.nominal_rate_bps
    ).all()



