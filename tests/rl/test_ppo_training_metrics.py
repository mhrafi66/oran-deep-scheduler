from types import SimpleNamespace

import torch

from oran_scheduler.rl.ppo_training_metrics import (
    PPOTrainingMetricsObserver,
)


def build_fake_result(
    *,
    delivered_rate_bps,
    history_bps,
    reward_geomean_bps,
):
    valid_mask = torch.ones(
        len(
            delivered_rate_bps
        ),
        dtype=torch.bool,
    )

    greedy_indicator = torch.tensor(
        [
            [
                1.0,
                -1.0,
            ],
            [
                1.0,
                1.0,
            ],
        ],
        dtype=torch.float32,
    )

    reward_data = SimpleNamespace(
        geometric_mean_throughput_bps=(
            torch.tensor(
                reward_geomean_bps,
                dtype=torch.float32,
            )
        ),
        normalized_geometric_mean=(
            torch.tensor(
                reward_geomean_bps
                / 100.0e6,
                dtype=torch.float32,
            )
        ),
        greedy_indicator=(
            greedy_indicator
        ),
    )

    reward = SimpleNamespace(
        reward_data=reward_data,
        reduced_reward=torch.tensor(
            [
                0.1,
                0.2,
            ],
            dtype=torch.float32,
        ),
    )

    return SimpleNamespace(
        scheduler_observation=(
            SimpleNamespace(
                serving_ue_valid_mask=(
                    valid_mask
                ),
            )
        ),
        traffic_service=(
            SimpleNamespace(
                delivered_rate_bps=(
                    torch.tensor(
                        delivered_rate_bps,
                        dtype=torch.float32,
                    )
                ),
            )
        ),
        history_update=(
            SimpleNamespace(
                updated_average_throughput_bps=(
                    torch.tensor(
                        history_bps,
                        dtype=torch.float32,
                    )
                ),
            )
        ),
        reward=reward,
    )


def test_metrics_observer_writes_one_network_row(
    tmp_path,
):
    output_path = (
        tmp_path
        / "metrics.csv"
    )

    observer = (
        PPOTrainingMetricsObserver(
            num_cells=2,
            csv_path=output_path,
        )
    )

    observer(
        0,
        0,
        build_fake_result(
            delivered_rate_bps=[
                10.0e6,
                20.0e6,
            ],
            history_bps=[
                5.0e6,
                10.0e6,
            ],
            reward_geomean_bps=(
                4.0e6
            ),
        ),
    )

    assert len(
        observer.records
    ) == 0

    observer(
        0,
        1,
        build_fake_result(
            delivered_rate_bps=[
                30.0e6,
                40.0e6,
            ],
            history_bps=[
                15.0e6,
                20.0e6,
            ],
            reward_geomean_bps=(
                8.0e6
            ),
        ),
    )

    assert len(
        observer.records
    ) == 1

    record = observer.records[
        0
    ]

    assert record.tti_index == 0

    assert record.num_cells == 2

    assert record.num_valid_ues == 4

    assert abs(
        record.mean_delivered_mbps
        - 25.0
    ) < 1.0e-6

    assert output_path.exists()

    lines = (
        output_path
        .read_text()
        .strip()
        .splitlines()
    )

    assert len(
        lines
    ) == 2


