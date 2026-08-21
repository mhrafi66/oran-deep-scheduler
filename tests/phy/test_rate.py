import torch

from oran_scheduler.phy.rate import (
    RateConfig,
    compute_rbg_rates,
)

def test_compute_rbg_rates_shapes():
    device = "cuda:0"

    config = RateConfig(
        device=device,
    )

    mcs_index = torch.tensor(
        [
            [
                [2, 10, 20],
                [3, 12, 25],
            ]
        ],
        dtype=torch.int32,
        device=device,
    )

    tbler = torch.tensor(
        [
            [
                [0.05, 0.05, 0.05],
                [0.05, 0.05, 0.05],
            ]
        ],
        device=device,
    )

    result = compute_rbg_rates(
        mcs_index=mcs_index,
        tbler=tbler,
        config=config,
    )

    expected_shape = (
        1,
        2,
        3,
    )

    assert result.tb_size_bits.shape == (
        expected_shape
    )

    assert result.nominal_rate_bps.shape == (
        expected_shape
    )

    assert result.expected_goodput_bps.shape == (
        expected_shape
    )

    assert result.target_compliant_rate_bps.shape == (
        expected_shape
    )

    assert torch.all(
        result.tb_size_bits > 0
    )

    assert torch.all(
        result.nominal_rate_bps > 0
    )

def test_expected_goodput_accounts_for_tbler():
    device = "cuda:0"

    config = RateConfig(
        device=device,
    )

    mcs_index = torch.tensor(
        [
            [
                [10],
                [10],
            ]
        ],
        dtype=torch.int32,
        device=device,
    )

    tbler = torch.tensor(
        [
            [
                [0.0],
                [0.5],
            ]
        ],
        device=device,
    )

    result = compute_rbg_rates(
        mcs_index=mcs_index,
        tbler=tbler,
        config=config,
    )

    torch.testing.assert_close(
        result.nominal_rate_bps[
            0,
            0,
            0,
        ],
        result.nominal_rate_bps[
            0,
            1,
            0,
        ],
    )

    expected_half_rate = (
        0.5
        * result.nominal_rate_bps[
            0,
            1,
            0,
        ]
    )

    torch.testing.assert_close(
        result.expected_goodput_bps[
            0,
            1,
            0,
        ],
        expected_half_rate,
    )

    torch.testing.assert_close(
        result.expected_goodput_bps[
            0,
            0,
            0,
        ],
        result.nominal_rate_bps[
            0,
            0,
            0,
        ],
    )

def test_target_compliant_rate_zeroes_outage():
    device = "cuda:0"

    config = RateConfig(
        bler_target=0.10,
        device=device,
    )

    mcs_index = torch.tensor(
        [
            [
                [2],
                [2],
            ]
        ],
        dtype=torch.int32,
        device=device,
    )

    tbler = torch.tensor(
        [
            [
                [0.05],
                [0.80],
            ]
        ],
        device=device,
    )

    result = compute_rbg_rates(
        mcs_index=mcs_index,
        tbler=tbler,
        config=config,
    )

    assert bool(
        result.meets_bler_target[
            0,
            0,
            0,
        ].item()
    )

    assert not bool(
        result.meets_bler_target[
            0,
            1,
            0,
        ].item()
    )

    assert (
        result.target_compliant_rate_bps[
            0,
            0,
            0,
        ].item()
        > 0.0
    )

    assert (
        result.target_compliant_rate_bps[
            0,
            1,
            0,
        ].item()
        == 0.0
    )

