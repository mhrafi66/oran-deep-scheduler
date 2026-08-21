import torch

from oran_scheduler.phy.sinr import (
    SINRConfig,
    compute_noise_power_per_subcarrier_w,
    compute_siso_multicell_sinr,
    compute_subcarrier_tx_power_w,
    compute_total_tx_power_w,
)

def test_tx_power_distribution():
    device = "cuda:0"

    config = SINRConfig(
        tx_power_dbm=44.0,
        device=device,
    )

    total_power_w = compute_total_tx_power_w(
        config
    )

    subcarrier_power_w = (
        compute_subcarrier_tx_power_w(
            config=config,
            num_subcarriers=216,
        )
    )

    torch.testing.assert_close(
        subcarrier_power_w * 216,
        total_power_w,
    )

    assert total_power_w.device.type == "cuda"


def test_thermal_noise_is_positive():
    device = "cuda:0"

    config = SINRConfig(
        temperature_k=294.0,
        subcarrier_spacing_hz=30e3,
        receiver_noise_figure_db=None,
        device=device,
    )

    noise_power_w = (
        compute_noise_power_per_subcarrier_w(
            config
        )
    )

    assert noise_power_w.item() > 0.0

    assert noise_power_w.device.type == "cuda"

def test_siso_multicell_sinr_numerical_example():
    device = "cuda:0"

    h_freq = torch.zeros(
        1,  # batch
        1,  # UE
        1,  # RX antenna
        3,  # BS
        1,  # TX antenna
        1,  # OFDM symbol
        1,  # subcarrier
        dtype=torch.complex64,
        device=device,
    )

    h_freq[0, 0, 0, 0, 0, 0, 0] = (
        1.0 + 0.0j
    )

    h_freq[0, 0, 0, 1, 0, 0, 0] = (
        2.0 + 0.0j
    )

    h_freq[0, 0, 0, 2, 0, 0, 0] = (
        0.5 + 0.0j
    )

    serving_bs = torch.tensor(
        [
            [1]
        ],
        device=device,
    )


    config = SINRConfig(
        tx_power_dbm=0.0,
        temperature_k=294.0,
        subcarrier_spacing_hz=30e3,
        receiver_noise_figure_db=None,
        device=device,
    )


    result = compute_siso_multicell_sinr(
        h_freq=h_freq,
        serving_bs=serving_bs,
        config=config,
    )


    tx_power_w = (
        compute_subcarrier_tx_power_w(
            config=config,
            num_subcarriers=1,
        )
    )

    noise_power_w = (
        compute_noise_power_per_subcarrier_w(
            config
        )
    )


    expected_desired = (
        tx_power_w
        * 4.0
    )

    expected_interference = (
        tx_power_w
        * (1.0 + 0.25)
    )


    expected_sinr = (
        expected_desired
        / (
            expected_interference
            + noise_power_w
        )
    )

    torch.testing.assert_close(
        result.desired_power_w[
            0,
            0,
            0,
            0,
        ],
        expected_desired,
    )

    torch.testing.assert_close(
        result.interference_power_w[
            0,
            0,
            0,
            0,
        ],
        expected_interference,
    )

    torch.testing.assert_close(
        result.sinr_linear[
            0,
            0,
            0,
            0,
        ],
        expected_sinr,
    )

def test_inactive_bs_does_not_interfere():
    device = "cuda:0"

    h_freq = torch.ones(
        1,
        1,
        1,
        2,
        1,
        1,
        1,
        dtype=torch.complex64,
        device=device,
    )

    serving_bs = torch.tensor(
        [
            [0]
        ],
        device=device,
    )

    config = SINRConfig(
        tx_power_dbm=0.0,
        device=device,
    )

    all_active = torch.tensor(
        [
            [
                [[True]],
                [[True]],
            ]
        ],
        device=device,
    )

    interferer_off = torch.tensor(
        [
            [
                [[True]],
                [[False]],
            ]
        ],
        device=device,
    )

    result_all_active = (
        compute_siso_multicell_sinr(
            h_freq=h_freq,
            serving_bs=serving_bs,
            config=config,
            bs_activity_mask=all_active,
        )
    )

    result_interferer_off = (
        compute_siso_multicell_sinr(
            h_freq=h_freq,
            serving_bs=serving_bs,
            config=config,
            bs_activity_mask=interferer_off,
        )
    )

    assert torch.all(
        result_interferer_off.sinr_linear
        >
        result_all_active.sinr_linear
    )


