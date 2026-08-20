import torch

from oran_scheduler.simulator.cell_association import (
    associate_by_minimum_pathloss,
    compute_channel_averaged_pathloss,
    validate_cell_association,
)

def test_minimum_pathloss_association():
    device = "cuda:0"

    pathloss = torch.tensor(
        [
            [
                [10.0, 20.0, 30.0],
                [50.0, 15.0, 40.0],
                [80.0, 60.0, 12.0],
                [70.0, 14.0, 25.0],
            ]
        ],
        device=device,
    )

    association = associate_by_minimum_pathloss(
        pathloss=pathloss,
    )

    expected_serving_bs = torch.tensor(
        [
            [0, 1, 2, 1]
        ],
        device=device,
    )

    torch.testing.assert_close(
        association.serving_bs,
        expected_serving_bs,
    )

    expected_counts = torch.tensor(
        [
            [1, 2, 1]
        ],
        device=device,
    )

    torch.testing.assert_close(
        association.num_ues_per_cell,
        expected_counts,
    )

    validate_cell_association(
        association=association,
        expected_batch_size=1,
        expected_num_ues=4,
        expected_num_bs=3,
    )

def test_channel_averaged_pathloss():
    device = "cuda:0"

    h_freq = torch.zeros(
        1,   # batch
        2,   # UE
        1,   # RX antenna
        2,   # BS
        1,   # TX antenna
        1,   # OFDM symbol
        4,   # subcarriers
        dtype=torch.complex64,
        device=device,
    )

    h_freq[0, 0, 0, 0, 0, 0, :] = 2.0 + 0.0j
    h_freq[0, 0, 0, 1, 0, 0, :] = 1.0 + 0.0j

    h_freq[0, 1, 0, 0, 0, 0, :] = 0.5 + 0.0j
    h_freq[0, 1, 0, 1, 0, 0, :] = 1.0 + 0.0j

    pathloss = compute_channel_averaged_pathloss(
        h_freq=h_freq,
        precision="single",
    )

    assert pathloss.shape == (
        1,
        2,
        2,
    )

    association = associate_by_minimum_pathloss(
        pathloss=pathloss,
    )

    expected = torch.tensor(
        [
            [0, 1]
        ],
        device=device,
    )

    torch.testing.assert_close(
        association.serving_bs,
        expected,
    )

