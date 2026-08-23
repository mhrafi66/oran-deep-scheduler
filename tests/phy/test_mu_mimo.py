import torch

from oran_scheduler.phy.mu_mimo import (
    build_rank1_effective_channel,
    compute_rank1_mrc_post_sinr,
    compute_rzf_matrix,
)

def test_build_rank1_effective_channel():
    device = "cuda:0"

    h = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=torch.complex64,
        device=device,
    )

    combiner = torch.tensor(
        [
            [1.0, 0.0]
        ],
        dtype=torch.complex64,
        device=device,
    )

    effective = build_rank1_effective_channel(
        h=h,
        rx_combiner=combiner,
    )

    expected = torch.tensor(
        [
            [1.0, 2.0]
        ],
        dtype=torch.complex64,
        device=device,
    )

    torch.testing.assert_close(
        effective,
        expected,
    )

def test_rzf_matrix_shape_and_column_norm():
    device = "cuda:0"

    h_eff = torch.tensor(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    g = compute_rzf_matrix(
        effective_channel=h_eff,
        alpha=0.1,
    )

    assert g.shape == (
        4,
        2,
    )

    column_norms = torch.linalg.vector_norm(
        g,
        dim=-2,
    )

    torch.testing.assert_close(
        column_norms,
        torch.ones_like(
            column_norms
        ),
        rtol=1e-5,
        atol=1e-5,
    )

def test_zero_forcing_suppresses_intra_cell_interference():
    device = "cuda:0"

    h = torch.tensor(
        [
            [
                [1.0, 0.5, 0.0, 0.0]
            ],
            [
                [0.5, 1.0, 0.0, 0.0]
            ],
        ],
        dtype=torch.complex64,
        device=device,
    )

    combiner = torch.ones(
        2,
        1,
        dtype=torch.complex64,
        device=device,
    )

    h_eff = build_rank1_effective_channel(
        h=h,
        rx_combiner=combiner,
    )

    g = compute_rzf_matrix(
        effective_channel=h_eff,
        alpha=0.0,
    )

    result = compute_rank1_mrc_post_sinr(
        h=h,
        precoding_matrix=g,
        stream_power_w=1.0,
        noise_power_w=1.0e-3,
    )

    assert torch.all(
        result.intra_cell_interference_power
        < 1.0e-8
    )

    assert torch.all(
        result.sinr_linear > 0.0
    )

def test_mrc_combining_gain():
    device = "cuda:0"

    h = torch.tensor(
        [
            [
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        ],
        dtype=torch.complex64,
        device=device,
    )

    g = torch.tensor(
        [
            [1.0],
            [0.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    result = compute_rank1_mrc_post_sinr(
        h=h,
        precoding_matrix=g,
        stream_power_w=1.0,
        noise_power_w=1.0,
    )

    expected_desired_power = torch.tensor(
        [5.0],
        device=device,
    )

    expected_sinr = torch.tensor(
        [5.0],
        device=device,
    )

    torch.testing.assert_close(
        result.desired_power,
        expected_desired_power,
    )

    torch.testing.assert_close(
        result.sinr_linear,
        expected_sinr,
    )

