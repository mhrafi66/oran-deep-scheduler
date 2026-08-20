import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    run_pf_tds,
)

def test_pf_tds_fewer_than_ten_valid_ues():
    device = "cuda:0"

    config = PFTimeDomainConfig(
        num_candidates=10,
    )

    instantaneous_rate = torch.tensor(
        [
            [
                [
                    1.0,
                    6.0,
                    2.0,
                    5.0,
                    3.0,
                    4.0,
                    999.0,
                    999.0,
                    999.0,
                    999.0,
                    999.0,
                    999.0,
                ]
            ]
        ],
        device=device,
    )

    history = torch.ones_like(
        instantaneous_rate
    )

    valid_ue_mask = torch.tensor(
        [
            [
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            ]
        ],
        device=device,
    )

    result = run_pf_tds(
        instantaneous_rate=instantaneous_rate,
        past_average_throughput=history,
        config=config,
        valid_ue_mask=valid_ue_mask,
    )

    expected_real_indices = torch.tensor(
        [1, 3, 5, 4, 2, 0],
        device=device,
    )

    torch.testing.assert_close(
        result.candidate_indices[
            0,
            0,
            :6,
        ],
        expected_real_indices,
    )

    assert (
        result.candidate_valid_mask[
            0,
            0,
        ].sum().item()
        == 6
    )

    assert torch.all(
        ~result.candidate_valid_mask[
            0,
            0,
            6:,
        ]
    )

def test_pf_tds_zero_valid_ues():
    device = "cuda:0"

    config = PFTimeDomainConfig(
        num_candidates=10,
    )

    instantaneous_rate = torch.ones(
        1,
        1,
        12,
        device=device,
    )

    history = torch.ones_like(
        instantaneous_rate
    )

    valid_ue_mask = torch.zeros(
        1,
        1,
        12,
        dtype=torch.bool,
        device=device,
    )

    result = run_pf_tds(
        instantaneous_rate=instantaneous_rate,
        past_average_throughput=history,
        config=config,
        valid_ue_mask=valid_ue_mask,
    )

    assert result.candidate_indices.shape == (
        1,
        1,
        10,
    )

    assert not torch.any(
        result.candidate_valid_mask
    )

    assert torch.all(
        result.candidate_indices == 0
    )

    assert torch.all(
        result.candidate_metrics == 0
    )

def test_pf_tds_pool_dimension_smaller_than_ten():
    device = "cuda:0"

    config = PFTimeDomainConfig(
        num_candidates=10,
    )

    instantaneous_rate = torch.tensor(
        [
            [
                [4.0, 1.0, 3.0, 2.0]
            ]
        ],
        device=device,
    )

    history = torch.ones_like(
        instantaneous_rate
    )

    valid_ue_mask = torch.ones_like(
        instantaneous_rate,
        dtype=torch.bool,
    )

    result = run_pf_tds(
        instantaneous_rate=instantaneous_rate,
        past_average_throughput=history,
        config=config,
        valid_ue_mask=valid_ue_mask,
    )

    assert result.candidate_indices.shape == (
        1,
        1,
        10,
    )

    assert (
        result.candidate_valid_mask.sum().item()
        == 4
    )

    expected = torch.tensor(
        [0, 2, 3, 1],
        device=device,
    )

    torch.testing.assert_close(
        result.candidate_indices[
            0,
            0,
            :4,
        ],
        expected,
    )

