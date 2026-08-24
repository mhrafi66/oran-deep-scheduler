import torch

from oran_scheduler.phy.rank_diagnostic import (
    compute_ideal_svd_rank_diagnostic,
)

def test_low_snr_prefers_rank_one():
    device = "cuda:0"

    h_freq = torch.zeros(
        (
            1,
            1,
            2,
            1,
            2,
            1,
            1,
        ),
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0,
        0,
        :,
        0,
        :,
        0,
        0,
    ] = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    serving_bs = torch.tensor(
        [[0]],
        dtype=torch.long,
        device=device,
    )

    result = compute_ideal_svd_rank_diagnostic(
        h_freq=h_freq,
        serving_bs=serving_bs,
        num_rbgs=1,
        tx_power_per_subcarrier_w=1.0,
        noise_power_per_subcarrier_w=1.0,
    )

    expected_rank1_sinr = torch.tensor(
        [[[[[4.0]]]]],
        device=device,
    )

    torch.testing.assert_close(
        result.rank1_sinr,
        expected_rank1_sinr,
    )

    assert int(
        result.diagnostic_best_rank[
            0,
            0,
            0,
        ].item()
    ) == 1

def test_high_snr_prefers_rank_two():
    device = "cuda:0"

    h_freq = torch.zeros(
        (
            1,
            1,
            2,
            1,
            2,
            1,
            1,
        ),
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0,
        0,
        :,
        0,
        :,
        0,
        0,
    ] = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    serving_bs = torch.tensor(
        [[0]],
        dtype=torch.long,
        device=device,
    )

    result = compute_ideal_svd_rank_diagnostic(
        h_freq=h_freq,
        serving_bs=serving_bs,
        num_rbgs=1,
        tx_power_per_subcarrier_w=1.0,
        noise_power_per_subcarrier_w=0.01,
    )

    assert int(
        result.diagnostic_best_rank[
            0,
            0,
            0,
        ].item()
    ) == 2

def test_isotropic_inter_cell_interference():
    device = "cuda:0"

    h_freq = torch.zeros(
        (
            1,
            1,
            2,
            2,
            2,
            1,
            1,
        ),
        dtype=torch.complex64,
        device=device,
    )

    identity = torch.eye(
        2,
        dtype=torch.complex64,
        device=device,
    )

    h_freq[
        0,
        0,
        :,
        0,
        :,
        0,
        0,
    ] = identity

    h_freq[
        0,
        0,
        :,
        1,
        :,
        0,
        0,
    ] = identity

    serving_bs = torch.tensor(
        [[0]],
        dtype=torch.long,
        device=device,
    )

    result = compute_ideal_svd_rank_diagnostic(
        h_freq=h_freq,
        serving_bs=serving_bs,
        num_rbgs=1,
        tx_power_per_subcarrier_w=1.0,
        noise_power_per_subcarrier_w=1.0,
    )

    expected_interference = torch.tensor(
        [
            [
                [
                    [
                        [
                            [0.5, 0.5]
                        ]
                    ]
                ]
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        result.interference_power_per_mode,
        expected_interference,
    )


