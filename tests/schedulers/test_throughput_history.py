import torch

from oran_scheduler.schedulers.throughput_history import (
    map_candidate_rates_to_serving_ues,
    update_cell_throughput_history,
)

def test_map_candidate_rates_to_serving_ues():
    candidate_indices = torch.tensor(
        [
            2,
            0,
            1,
        ],
        dtype=torch.long,
    )

    candidate_valid_mask = torch.tensor(
        [
            True,
            True,
            False,
        ],
        dtype=torch.bool,
    )

    candidate_rates = torch.tensor(
        [
            4.0,
            2.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    delivered = (
        map_candidate_rates_to_serving_ues(
            candidate_indices=(
                candidate_indices
            ),
            candidate_valid_mask=(
                candidate_valid_mask
            ),
            candidate_delivered_rate_bps=(
                candidate_rates
            ),
            num_serving_ue_slots=4,
        )
    )

    expected = torch.tensor(
        [
            2.0,
            0.0,
            4.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        delivered,
        expected,
    )

def test_update_cell_throughput_history():
    previous_average = torch.tensor(
        [
            10.0,
            20.0,
            30.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    serving_valid_mask = torch.tensor(
        [
            True,
            True,
            True,
            False,
        ],
        dtype=torch.bool,
    )

    candidate_indices = torch.tensor(
        [
            2,
            0,
            1,
        ],
        dtype=torch.long,
    )

    candidate_valid_mask = torch.tensor(
        [
            True,
            True,
            False,
        ],
        dtype=torch.bool,
    )

    candidate_rates = torch.tensor(
        [
            4.0,
            2.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    result = update_cell_throughput_history(
        previous_average_throughput_bps=(
            previous_average
        ),
        serving_ue_valid_mask=(
            serving_valid_mask
        ),
        candidate_indices=(
            candidate_indices
        ),
        candidate_valid_mask=(
            candidate_valid_mask
        ),
        candidate_delivered_rate_bps=(
            candidate_rates
        ),
        forgetting_factor=0.9,
    )

    expected_delivered = torch.tensor(
        [
            2.0,
            0.0,
            4.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    expected_updated_average = torch.tensor(
        [
            9.2,
            18.0,
            27.4,
            0.0,
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        result.delivered_rate_bps,
        expected_delivered,
    )

    torch.testing.assert_close(
        result.updated_average_throughput_bps,
        expected_updated_average,
    )

def test_non_candidate_real_ue_history_decays():
    previous_average = torch.tensor(
        [
            10.0,
            20.0,
            30.0,
        ],
        dtype=torch.float32,
    )

    result = update_cell_throughput_history(
        previous_average_throughput_bps=(
            previous_average
        ),
        serving_ue_valid_mask=torch.tensor(
            [
                True,
                True,
                True,
            ],
            dtype=torch.bool,
        ),
        candidate_indices=torch.tensor(
            [
                0,
                2,
            ],
            dtype=torch.long,
        ),
        candidate_valid_mask=torch.tensor(
            [
                True,
                True,
            ],
            dtype=torch.bool,
        ),
        candidate_delivered_rate_bps=torch.tensor(
            [
                5.0,
                7.0,
            ],
            dtype=torch.float32,
        ),
        forgetting_factor=0.5,
    )

    # UE slot 1 was not even in the TDS candidate set,
    # so its delivered rate is zero.
    assert float(
        result.delivered_rate_bps[
            1
        ].item()
    ) == 0.0

    # But its history still advances through the TTI:
    #
    # 0.5 * 0 + 0.5 * 20 = 10.
    assert float(
        result.updated_average_throughput_bps[
            1
        ].item()
    ) == 10.0


