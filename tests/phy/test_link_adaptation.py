import torch

from oran_scheduler.phy.link_adaptation import (
    LinkAdaptationConfig,
    build_illa_rbg_sinr,
    select_rbg_mcs,
)

def test_build_illa_rbg_sinr_shape():
    device = "cuda:0"

    config = LinkAdaptationConfig(
        num_rbgs=2,
        subcarriers_per_rbg=3,
        num_slot_ofdm_symbols=4,
        device=device,
    )

    sinr = torch.ones(
        1,
        5,
        1,
        6,
        device=device,
    )

    illa_sinr = build_illa_rbg_sinr(
        sinr_linear=sinr,
        config=config,
    )

    assert illa_sinr.shape == (
        1,
        2,
        4,
        3,
        5,
        1,
    )

    assert illa_sinr.device.type == "cuda"

def test_build_illa_rbg_sinr_preserves_frequency_mapping():
    device = "cuda:0"

    config = LinkAdaptationConfig(
        num_rbgs=2,
        subcarriers_per_rbg=3,
        num_slot_ofdm_symbols=2,
        device=device,
    )

    sinr = torch.tensor(
        [
            [
                [
                    [
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                    ]
                ]
            ]
        ],
        device=device,
    )

    illa_sinr = build_illa_rbg_sinr(
        sinr_linear=sinr,
        config=config,
    )

    expected_rbg_0 = torch.tensor(
        [1.0, 2.0, 3.0],
        device=device,
    )

    expected_rbg_1 = torch.tensor(
        [4.0, 5.0, 6.0],
        device=device,
    )

    torch.testing.assert_close(
        illa_sinr[
            0,
            0,
            0,
            :,
            0,
            0,
        ],
        expected_rbg_0,
    )

    torch.testing.assert_close(
        illa_sinr[
            0,
            1,
            0,
            :,
            0,
            0,
        ],
        expected_rbg_1,
    )

    torch.testing.assert_close(
        illa_sinr[
            0,
            0,
            0,
            :,
            0,
            0,
        ],
        illa_sinr[
            0,
            0,
            1,
            :,
            0,
            0,
        ],
    )

def test_mcs_increases_with_sinr():
    device = "cuda:0"

    config = LinkAdaptationConfig(
        num_rbgs=1,
        subcarriers_per_rbg=12,
        num_slot_ofdm_symbols=14,
        bler_target=0.10,
        mcs_category=1,
        mcs_table_index=2,
        device=device,
    )

    weak_sinr = 10.0 ** (
        -5.0 / 10.0
    )

    strong_sinr = 10.0 ** (
        20.0 / 10.0
    )

    sinr = torch.empty(
        1,
        2,
        1,
        12,
        device=device,
    )

    sinr[
        0,
        0,
        0,
        :,
    ] = weak_sinr

    sinr[
        0,
        1,
        0,
        :,
    ] = strong_sinr

    result = select_rbg_mcs(
        sinr_linear=sinr,
        config=config,
    )

    assert result.tbler.shape == (
        1,
        2,
        1,
    )

    assert result.bler.shape == (
        1,
        2,
        1,
    )

    assert result.effective_sinr_linear.shape == (
        1,
        2,
        1,
    )

    assert result.meets_bler_target.shape == (
        1,
        2,
        1,
    )

    assert torch.all(
        result.tbler >= 0.0
    )

    assert torch.all(
        result.tbler <= 1.0
    )

    assert torch.all(
        result.bler >= 0.0
    )

    assert torch.all(
        result.bler <= 1.0
    )

    assert bool(
        result.meets_bler_target[
            0,
            1,
            0,
        ].item()
    )

    assert result.mcs_index.shape == (
        1,
        2,
        1,
    )

    weak_mcs = result.mcs_index[
        0,
        0,
        0,
    ]

    strong_mcs = result.mcs_index[
        0,
        1,
        0,
    ]

    assert strong_mcs >= weak_mcs

    assert torch.all(
        result.mcs_index >= 0
    )

