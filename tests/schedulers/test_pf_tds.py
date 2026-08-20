import torch

from oran_scheduler.schedulers.pf_tds import (
    PFTimeDomainConfig,
    compute_pf_metric,
    run_pf_tds,
    update_past_average_throughput,
)

def test_pf_metric_favors_underserved_user() -> None:
    config = PFTimeDomainConfig(
        num_candidates=2,
    )

    instantaneous_rate = torch.tensor(
        [[[10.0, 8.0]]]
    )

    past_average_throughput = torch.tensor(
        [[[20.0, 4.0]]]
    )

    metric = compute_pf_metric(
        instantaneous_rate=instantaneous_rate,
        past_average_throughput=past_average_throughput,
        config=config,
    )

    expected = torch.tensor(
        [[[0.5, 2.0]]]
    )

    torch.testing.assert_close(
        metric,
        expected,
    )

def test_pf_tds_selects_top_candidates() -> None:
    config = PFTimeDomainConfig(
        num_candidates=3,
    )

    instantaneous_rate = torch.tensor(
        [[[
            10.0,
            8.0,
            9.0,
            5.0,
            6.0,
        ]]]
    )

    past_average_throughput = torch.tensor(
        [[[
            10.0,
            2.0,
            3.0,
            5.0,
            12.0,
        ]]]
    )

    candidate_indices, candidate_metrics = run_pf_tds(
        instantaneous_rate=instantaneous_rate,
        past_average_throughput=past_average_throughput,
        config=config,
    )

    expected_indices = torch.tensor(
        [[[1, 2, 0]]]
    )

    torch.testing.assert_close(
        candidate_indices,
        expected_indices,
    )

def test_training_shape_selects_10_of_20_ues() -> None:
    config = PFTimeDomainConfig(
        num_candidates=10,
    )

    batch_size = 1
    num_cells = 21
    num_ues_per_cell = 20

    instantaneous_rate = torch.rand(
        batch_size,
        num_cells,
        num_ues_per_cell,
        device="cuda:0",
    )

    past_average_throughput = torch.rand(
        batch_size,
        num_cells,
        num_ues_per_cell,
        device="cuda:0",
    ) + 0.1

    candidate_indices, candidate_metrics = run_pf_tds(
        instantaneous_rate=instantaneous_rate,
        past_average_throughput=past_average_throughput,
        config=config,
    )

    assert tuple(candidate_indices.shape) == (
        1,
        21,
        10,
    )

    assert tuple(candidate_metrics.shape) == (
        1,
        21,
        10,
    )

    assert candidate_indices.device.type == "cuda"

def test_past_average_throughput_update() -> None:
    previous_average = torch.tensor(
        [10.0, 20.0]
    )

    delivered_rate = torch.tensor(
        [30.0, 0.0]
    )

    updated = update_past_average_throughput(
        previous_average=previous_average,
        delivered_rate=delivered_rate,
        forgetting_factor=0.9,
    )

    expected = torch.tensor(
        [
            12.0,
            18.0,
        ]
    )

    torch.testing.assert_close(
        updated,
        expected,
    )

