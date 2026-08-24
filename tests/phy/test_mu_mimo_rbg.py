import torch

from oran_scheduler.phy.mu_mimo_rbg import (
    compute_isotropic_inter_cell_covariance,
    evaluate_selected_users_on_rbg,
)

def test_rank_two_plus_rank_one_rbg():
    device = "cuda:0"

    selected_channel = torch.zeros(
        (
            2,  # UEs
            1,  # symbol
            2,  # subcarriers
            2,  # RX
            3,  # TX
        ),
        dtype=torch.complex64,
        device=device,
    )

    ue0_channel = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    ue1_channel = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.complex64,
        device=device,
    )

    selected_channel[
        0,
        :,
        :,
        :,
        :,
    ] = ue0_channel

    selected_channel[
        1,
        :,
        :,
        :,
        :,
    ] = ue1_channel

    ranks = torch.tensor(
        [2, 1],
        dtype=torch.long,
        device=device,
    )

    combiners = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ],
        dtype=torch.complex64,
        device=device,
    )

    result = evaluate_selected_users_on_rbg(
        selected_ue_channel=selected_channel,
        selected_ranks=ranks,
        selected_rx_combiners=combiners,
        total_tx_power_w=3.0,
        noise_power_w=1.0,
        rzf_alpha=0.0,
        csi_subcarrier_index=0,
    )

    expected_layer_ues = torch.tensor(
        [0, 0, 1],
        dtype=torch.long,
        device=device,
    )

    expected_layer_numbers = torch.tensor(
        [0, 1, 0],
        dtype=torch.long,
        device=device,
    )

    torch.testing.assert_close(
        result.layer_ue_indices,
        expected_layer_ues,
    )

    torch.testing.assert_close(
        result.layer_index_within_ue,
        expected_layer_numbers,
    )

    assert result.sinr_data.sinr_linear.shape == (
        1,
        2,
        3,
    )

    expected_sinr = torch.tensor(
        [
            [
                [4.0, 1.0, 1.0],
                [4.0, 1.0, 1.0],
            ]
        ],
        device=device,
    )

    torch.testing.assert_close(
        result.sinr_data.sinr_linear,
        expected_sinr,
        atol=1.0e-5,
        rtol=1.0e-5,
    )

def test_isotropic_inter_cell_covariance():
    device = "cuda:0"

    h = torch.zeros(
        (
            1,  # UE
            1,  # symbol
            1,  # subcarrier
            2,  # BS
            2,  # RX
            2,  # TX
        ),
        dtype=torch.complex64,
        device=device,
    )

    identity = torch.eye(
        2,
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
    ] = identity

    h[
        0,
        0,
        0,
        1,
        :,
        :,
    ] = identity

    covariance = (
        compute_isotropic_inter_cell_covariance(
            all_bs_channel=h,
            serving_cell_index=0,
            tx_power_per_subcarrier_w=2.0,
        )
    )

    expected = torch.eye(
        2,
        dtype=torch.complex64,
        device=device,
    ).reshape(
        1,
        1,
        1,
        2,
        2,
    )

    torch.testing.assert_close(
        covariance,
        expected,
    )

