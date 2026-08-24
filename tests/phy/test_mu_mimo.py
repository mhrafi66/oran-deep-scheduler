import torch

from oran_scheduler.phy.mu_mimo import (
    build_rank1_effective_channel,
    compute_rank1_mrc_post_sinr,
    compute_rzf_matrix,
)

from oran_scheduler.phy.mu_mimo import (
    compute_mu_mimo_layer_sinr,
)

from oran_scheduler.phy.mu_mimo import (
    evaluate_mu_mimo_layer_sinr_with_precoder,
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



def test_general_mu_mimo_single_layer():
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

    combiner = torch.tensor(
        [
            [1.0, 2.0]
        ],
        dtype=torch.complex64,
        device=device,
    )

    result = compute_mu_mimo_layer_sinr(
        layer_physical_channel=h,
        layer_rx_combiner=combiner,
        total_tx_power_w=1.0,
        noise_power_w=1.0,
        rzf_alpha=0.0,
    )
    assert result.precoding_matrix.shape == (
        2,
        1,
    )

    assert result.combined_channel.shape == (
        1,
        1,
    )

    torch.testing.assert_close(
        result.intra_cell_interference_power,
        torch.zeros_like(
            result.intra_cell_interference_power
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert torch.all(
        result.sinr_linear > 0
    )

def test_two_orthogonal_layers_have_zero_intra_cell_interference():
    device = "cuda:0"

    h = torch.tensor(
        [
            [
                [1.0, 0.0]
            ],
            [
                [0.0, 1.0]
            ],
        ],
        dtype=torch.complex64,
        device=device,
    )

    combiner = torch.ones(
        (
            2,
            1,
        ),
        dtype=torch.complex64,
        device=device,
    )

    result = compute_mu_mimo_layer_sinr(
        layer_physical_channel=h,
        layer_rx_combiner=combiner,
        total_tx_power_w=2.0,
        noise_power_w=1.0,
        rzf_alpha=0.0,
    )

    expected_stream_power = torch.tensor(
        [1.0, 1.0],
        device=device,
    )

    torch.testing.assert_close(
        result.stream_power_w,
        expected_stream_power,
    )

    torch.testing.assert_close(
        result.intra_cell_interference_power,
        torch.zeros_like(
            result.intra_cell_interference_power
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    expected_desired = torch.tensor(
        [1.0, 1.0],
        device=device,
    )

    torch.testing.assert_close(
        result.desired_power,
        expected_desired,
        atol=1.0e-5,
        rtol=1.0e-5,
    )

    expected_sinr = torch.tensor(
        [1.0, 1.0],
        device=device,
    )

    torch.testing.assert_close(
        result.sinr_linear,
        expected_sinr,
        atol=1.0e-5,
        rtol=1.0e-5,
    )

def test_two_layers_can_belong_to_same_physical_ue():
    device = "cuda:0"

    physical_channel = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    h = torch.stack(
        [
            physical_channel,
            physical_channel,
        ],
        dim=0,
    )

    combiner = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    result = compute_mu_mimo_layer_sinr(
        layer_physical_channel=h,
        layer_rx_combiner=combiner,
        total_tx_power_w=2.0,
        noise_power_w=1.0,
        rzf_alpha=0.0,
    )

    assert result.effective_channel.shape == (
        2,
        2,
    )

    assert result.precoding_matrix.shape == (
        2,
        2,
    )

    torch.testing.assert_close(
        result.intra_cell_interference_power,
        torch.zeros_like(
            result.intra_cell_interference_power
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

def test_mu_mimo_inter_cell_covariance():
    device = "cuda:0"

    h = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
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

    inter_cell_covariance = torch.tensor(
        [
            [
                [3.0, 0.0],
                [0.0, 5.0],
            ]
        ],
        dtype=torch.complex64,
        device=device,
    )

    result = compute_mu_mimo_layer_sinr(
        layer_physical_channel=h,
        layer_rx_combiner=combiner,
        total_tx_power_w=1.0,
        noise_power_w=1.0,
        rzf_alpha=0.0,
        inter_cell_covariance=(
            inter_cell_covariance
        ),
    )

    expected_interference = torch.tensor(
        [3.0],
        device=device,
    )

    torch.testing.assert_close(
        result.inter_cell_interference_power,
        expected_interference,
    )

    expected_sinr = torch.tensor(
        [0.25],
        device=device,
    )

    torch.testing.assert_close(
        result.sinr_linear,
        expected_sinr,
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_fixed_precoder_over_multiple_subcarriers():
    device = "cuda:0"

    h = torch.zeros(
        (
            1,  # OFDM symbol
            2,  # subcarrier
            2,  # layer
            1,  # RX antenna
            2,  # TX antenna
        ),
        dtype=torch.complex64,
        device=device,
    )

    h[
        :,
        :,
        0,
        0,
        :,
    ] = torch.tensor(
        [1.0, 0.0],
        dtype=torch.complex64,
        device=device,
    )

    h[
        :,
        :,
        1,
        0,
        :,
    ] = torch.tensor(
        [0.0, 1.0],
        dtype=torch.complex64,
        device=device,
    )

    combiner = torch.ones(
        (
            2,
            1,
        ),
        dtype=torch.complex64,
        device=device,
    )

    precoder = torch.eye(
        2,
        dtype=torch.complex64,
        device=device,
    )

    result = (
        evaluate_mu_mimo_layer_sinr_with_precoder(
            layer_physical_channel=h,
            layer_rx_combiner=combiner,
            precoding_matrix=precoder,
            total_tx_power_w=2.0,
            noise_power_w=1.0,
        )
    )

    assert result.sinr_linear.shape == (
        1,
        2,
        2,
    )

    expected_sinr = torch.ones(
        (
            1,
            2,
            2,
        ),
        device=device,
    )

    torch.testing.assert_close(
        result.sinr_linear,
        expected_sinr,
        atol=1.0e-5,
        rtol=1.0e-5,
    )

    torch.testing.assert_close(
        result.intra_cell_interference_power,
        torch.zeros_like(
            result.intra_cell_interference_power
        ),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

