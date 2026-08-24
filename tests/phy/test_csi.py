import torch

from oran_scheduler.phy.csi import (
    compute_rbg_spatial_modes,
    extract_serving_mimo_rbg_channel,
)

from oran_scheduler.phy.csi import (
    compute_ideal_svd_csi,
)

def test_extract_serving_mimo_rbg_channel():
    device = "cuda:0"

    h_freq = torch.zeros(
        (
            1,  # batch
            2,  # UE
            1,  # RX antenna
            2,  # BS
            1,  # TX antenna
            1,  # OFDM symbol
            4,  # subcarriers
        ),
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0, 0, 0, 0, 0, 0, :
    ] = torch.tensor(
        [10.0, 11.0, 12.0, 13.0],
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0, 0, 0, 1, 0, 0, :
    ] = torch.tensor(
        [20.0, 21.0, 22.0, 23.0],
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0, 1, 0, 0, 0, 0, :
    ] = torch.tensor(
        [30.0, 31.0, 32.0, 33.0],
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0, 1, 0, 1, 0, 0, :
    ] = torch.tensor(
        [40.0, 41.0, 42.0, 43.0],
        dtype=torch.complex64,
        device=device,
    )

    serving_bs = torch.tensor(
        [
            [1, 0]
        ],
        dtype=torch.long,
        device=device,
    )

    result = (
        extract_serving_mimo_rbg_channel(
            h_freq=h_freq,
            serving_bs=serving_bs,
            num_rbgs=2,
        )
    )

    assert result.h_serving_rbg.shape == (
        1,
        2,
        2,
        1,
        2,
        1,
        1,
    )

    assert result.subcarriers_per_rbg == 2

    ue_0_rbg_0 = (
        result.h_serving_rbg[
            0,
            0,
            0,
            0,
            :,
            0,
            0,
        ]
    )

    expected_ue_0_rbg_0 = torch.tensor(
        [20.0, 21.0],
        dtype=torch.complex64,
        device=device,
    )

    torch.testing.assert_close(
        ue_0_rbg_0,
        expected_ue_0_rbg_0,
    )

    ue_1_rbg_1 = (
        result.h_serving_rbg[
            0,
            1,
            1,
            0,
            :,
            0,
            0,
        ]
    )

    expected_ue_1_rbg_1 = torch.tensor(
        [32.0, 33.0],
        dtype=torch.complex64,
        device=device,
    )

    torch.testing.assert_close(
        ue_1_rbg_1,
        expected_ue_1_rbg_1,
    )

def test_compute_rbg_spatial_modes():
    device = "cuda:0"

    h = torch.zeros(
        (
            1,  # batch
            1,  # UE
            1,  # RBG
            1,  # OFDM symbol
            2,  # subcarriers
            2,  # RX antennas
            2,  # TX antennas
        ),
        dtype=torch.complex64,
        device=device,
    )

    h[
        0, 0, 0, 0, 0, :, :
    ] = torch.tensor(
        [
            [3.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    h[
        0, 0, 0, 0, 1, :, :
    ] = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    result = compute_rbg_spatial_modes(
        h_serving_rbg=h
    )

    expected_singular_values = torch.tensor(
        [
            [
                [
                    [
                        [
                            [3.0, 1.0],
                            [4.0, 2.0],
                        ]
                    ]
                ]
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        result.singular_values,
        expected_singular_values,
    )

    expected_mode_power = torch.tensor(
        [
            [
                [
                    [12.5, 2.5]
                ]
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        result.rbg_mode_power,
        expected_mode_power,
    )

    expected_ratio = torch.tensor(
        [
            [
                [0.2]
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        result.second_to_first_power_ratio,
        expected_ratio,
    )

def test_rank_one_channel_has_tiny_second_mode():
    device = "cuda:0"

    h = torch.tensor(
        [
            [
                [
                    [
                        [
                            [
                                [1.0, 1.0],
                                [2.0, 2.0],
                            ]
                        ]
                    ]
                ]
            ]
        ],
        dtype=torch.complex64,
        device=device,
    )

    result = compute_rbg_spatial_modes(
        h_serving_rbg=h
    )

    ratio = (
        result.second_to_first_power_ratio[
            0,
            0,
            0,
        ]
    )

    assert ratio < 1.0e-6



def test_compute_ideal_svd_csi_rank_two():
    device = "cuda:0"

    h = torch.zeros(
        (
            1,  # batch
            1,  # UE
            2,  # RBG
            1,  # OFDM symbol
            2,  # subcarriers
            2,  # RX
            2,  # TX
        ),
        dtype=torch.complex64,
        device=device,
    )


    channel_matrix = torch.tensor(
        [
            [3.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    h[
        0,
        0,
        :,
        0,
        :,
        :,
        :,
    ] = channel_matrix

    rank1_score = torch.tensor(
        [
            [
                [2.0, 2.0]
            ]
        ],
        device=device,
    )

    rank2_score = torch.tensor(
        [
            [
                [3.0, 3.0]
            ]
        ],
        device=device,
    )

    result = compute_ideal_svd_csi(
        h_serving_rbg=h,
        rank1_rbg_score=rank1_score,
        rank2_rbg_score=rank2_score,
    )

    assert result.csi_snapshot.shape == (
        1,
        1,
        2,
        2,
        2,
    )

    assert result.singular_values.shape == (
        1,
        1,
        2,
        2,
    )

    assert result.rx_combiners.shape == (
        1,
        1,
        2,
        2,
        2,
    )

    assert result.precoder_directions.shape == (
        1,
        1,
        2,
        2,
        2,
    )

    assert result.effective_channels.shape == (
        1,
        1,
        2,
        2,
        2,
    )

    assert result.layer_valid_mask.shape == (
        1,
        1,
        2,
        2,
    )

    assert int(
        result.recommended_rank[
            0,
            0,
        ].item()
    ) == 2

    assert torch.all(
        result.layer_valid_mask[
            0,
            0,
            :,
            :,
        ]
    )

    expected_effective = (
        result.singular_values.unsqueeze(-1)
        * result.precoder_directions.conj()
    )

    torch.testing.assert_close(
        result.effective_channels,
        expected_effective,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    rx_norms = torch.linalg.vector_norm(
        result.rx_combiners,
        dim=-1,
    )

    tx_norms = torch.linalg.vector_norm(
        result.precoder_directions,
        dim=-1,
    )

    torch.testing.assert_close(
        rx_norms,
        torch.ones_like(
            rx_norms
        ),
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    torch.testing.assert_close(
        tx_norms,
        torch.ones_like(
            tx_norms
        ),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_compute_ideal_svd_csi_rank_one_mask():
    device = "cuda:0"

    h = torch.zeros(
        (
            1,
            1,
            1,
            1,
            2,
            2,
            2,
        ),
        dtype=torch.complex64,
        device=device,
    )

    h[
        0,
        0,
        0,
        0,
        :,
        :,
        :,
    ] = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 0.5],
        ],
        dtype=torch.complex64,
        device=device,
    )

    rank1_score = torch.tensor(
        [
            [
                [5.0]
            ]
        ],
        device=device,
    )

    rank2_score = torch.tensor(
        [
            [
                [4.0]
            ]
        ],
        device=device,
    )

    result = compute_ideal_svd_csi(
        h_serving_rbg=h,
        rank1_rbg_score=rank1_score,
        rank2_rbg_score=rank2_score,
    )

    assert int(
        result.recommended_rank[
            0,
            0,
        ].item()
    ) == 1

    assert bool(
        result.layer_valid_mask[
            0,
            0,
            0,
            0,
        ].item()
    )

    assert not bool(
        result.layer_valid_mask[
            0,
            0,
            0,
            1,
        ].item()
    )


