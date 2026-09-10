from __future__ import annotations

import csv

from dataclasses import dataclass
from pathlib import Path

import torch

from oran_scheduler.rl.ppo_traffic_cell_step import (
    TrafficAwarePPOCellTTIStepResult,
)


@dataclass(frozen=True)
class PPONetworkTTIMetrics:
    """
    Network-level diagnostics for one completed TTI.

    IMPORTANT:

    history_* metrics use the exponentially smoothed
    throughput history maintained by the PF scheduler.

    They are therefore useful for monitoring the
    long-term proportional-fairness objective.

    They are NOT a substitute for the final paper
    evaluation over complete evaluation simulations.
    """

    tti_index: int

    num_cells: int

    num_valid_ues: int

    mean_delivered_mbps: float

    mean_history_mbps: float

    p05_history_mbps: float

    median_history_mbps: float

    history_geomean_mbps: float

    paper_floor_history_geomean_mbps: float

    reward_population_geomean_mbps: (
        float | None
    )

    normalized_reward_geomean: (
        float | None
    )

    mean_layer_reward: (
        float | None
    )

    greedy_better_fraction: (
        float | None
    )


def _geometric_mean(
    values: torch.Tensor,
) -> float:
    """
    Exact geometric mean.

    If any value is zero, the exact geometric mean
    is zero.
    """

    if values.ndim != 1:
        raise ValueError(
            "values must have shape [sample]."
        )

    if values.numel() == 0:
        raise ValueError(
            "Cannot calculate an empty geometric mean."
        )

    if torch.any(
        values < 0.0
    ):
        raise ValueError(
            "Geometric-mean values cannot be negative."
        )

    if torch.any(
        values == 0.0
    ):
        return 0.0

    return float(
        torch.exp(
            torch.log(
                values
            ).mean()
        ).item()
    )


def _paper_floor_geometric_mean(
    values_bps: torch.Tensor,
) -> float:
    """
    Evaluation-style diagnostic.

    The v3 paper states that zero UE throughputs are
    replaced with one when reporting evaluation
    geometric means.

    This function keeps that rule separate from the
    exact training-reward geometric mean.
    """

    floored = torch.where(
        values_bps == 0.0,
        torch.ones_like(
            values_bps
        ),
        values_bps,
    )

    return _geometric_mean(
        floored
    )


class PPOTrainingMetricsObserver:
    """
    Consume completed cell TTIs without retaining
    expensive physical tensors.

    The multicell runner calls this object once per
    completed cell.

    Once all expected cells of a TTI have arrived,
    one network-level CSV row is produced.
    """

    FIELDNAMES = [
        "tti",
        "num_cells",
        "num_valid_ues",
        "mean_delivered_mbps",
        "mean_history_mbps",
        "p05_history_mbps",
        "median_history_mbps",
        "history_geomean_mbps",
        "paper_floor_history_geomean_mbps",
        "reward_population_geomean_mbps",
        "normalized_reward_geomean",
        "mean_layer_reward",
        "greedy_better_fraction",
    ]

    def __init__(
        self,
        *,
        num_cells: int,
        csv_path: str | Path,
    ) -> None:
        if num_cells <= 0:
            raise ValueError(
                "num_cells must be positive."
            )

        self.num_cells = int(
            num_cells
        )

        self.csv_path = Path(
            csv_path
        )

        self.csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._pending_tti_index: (
            int | None
        ) = None

        self._seen_cells: set[
            int
        ] = set()

        self._delivered_rate_parts: list[
            torch.Tensor
        ] = []

        self._history_parts: list[
            torch.Tensor
        ] = []

        self._reward_geomean: list[
            float
        ] = []

        self._normalized_geomean: list[
            float
        ] = []

        self._mean_layer_reward: list[
            float
        ] = []

        self._greedy_better_fraction: list[
            float
        ] = []

        self.records: list[
            PPONetworkTTIMetrics
        ] = []

        with self.csv_path.open(
            "w",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    self.FIELDNAMES
                ),
            )
            writer.writeheader()

    def _reset_pending(
        self,
        tti_index: int,
    ) -> None:
        self._pending_tti_index = (
            int(
                tti_index
            )
        )

        self._seen_cells.clear()

        self._delivered_rate_parts.clear()

        self._history_parts.clear()

        self._reward_geomean.clear()

        self._normalized_geomean.clear()

        self._mean_layer_reward.clear()

        self._greedy_better_fraction.clear()

    def __call__(
        self,
        tti_index: int,
        cell_index: int,
        result: (
            TrafficAwarePPOCellTTIStepResult
        ),
    ) -> None:
        if self._pending_tti_index is None:
            self._reset_pending(
                tti_index
            )

        if (
            tti_index
            != self._pending_tti_index
        ):
            raise RuntimeError(
                "A new TTI arrived before all "
                "cells of the previous TTI."
            )

        if cell_index in self._seen_cells:
            raise RuntimeError(
                "The same cell was observed twice "
                "for one TTI."
            )

        self._seen_cells.add(
            int(
                cell_index
            )
        )

        valid_mask = (
            result
            .scheduler_observation
            .serving_ue_valid_mask
        )

        delivered_rate = (
            result
            .traffic_service
            .delivered_rate_bps[
                valid_mask
            ]
            .detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )

        history = (
            result
            .history_update
            .updated_average_throughput_bps[
                valid_mask
            ]
            .detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )

        if delivered_rate.numel() == 0:
            raise RuntimeError(
                "A measured cell contains no valid UEs."
            )

        if history.numel() != delivered_rate.numel():
            raise RuntimeError(
                "Delivered-rate/history UE counts disagree."
            )

        self._delivered_rate_parts.append(
            delivered_rate
        )

        self._history_parts.append(
            history
        )

        reward = result.reward

        if reward is not None:
            reward_data = (
                reward.reward_data
            )

            self._reward_geomean.append(
                float(
                    reward_data
                    .geometric_mean_throughput_bps
                    .detach()
                    .item()
                )
            )

            self._normalized_geomean.append(
                float(
                    reward_data
                    .normalized_geometric_mean
                    .detach()
                    .item()
                )
            )

            self._mean_layer_reward.append(
                float(
                    reward
                    .reduced_reward
                    .detach()
                    .mean()
                    .item()
                )
            )

            better = (
                reward_data
                .greedy_indicator
                .detach()
                < 0
            )

            self._greedy_better_fraction.append(
                float(
                    better
                    .to(
                        dtype=torch.float32
                    )
                    .mean()
                    .item()
                )
            )

        if (
            len(
                self._seen_cells
            )
            == self.num_cells
        ):
            self._finish_tti()

    def _finish_tti(
        self,
    ) -> None:
        if self._pending_tti_index is None:
            raise RuntimeError(
                "No pending TTI exists."
            )

        delivered_rate = torch.cat(
            self._delivered_rate_parts,
            dim=0,
        )

        history = torch.cat(
            self._history_parts,
            dim=0,
        )

        num_valid_ues = int(
            history.numel()
        )

        if num_valid_ues < 1:
            raise RuntimeError(
                "Network metric set is empty."
            )

        p05 = torch.quantile(
            history,
            0.05,
        )

        median = torch.quantile(
            history,
            0.50,
        )

        history_geomean_bps = (
            _geometric_mean(
                history
            )
        )

        floor_geomean_bps = (
            _paper_floor_geometric_mean(
                history
            )
        )

        reward_geomean = (
            None
            if not self._reward_geomean
            else sum(
                self._reward_geomean
            )
            / len(
                self._reward_geomean
            )
        )

        normalized_geomean = (
            None
            if not self._normalized_geomean
            else sum(
                self._normalized_geomean
            )
            / len(
                self._normalized_geomean
            )
        )

        mean_layer_reward = (
            None
            if not self._mean_layer_reward
            else sum(
                self._mean_layer_reward
            )
            / len(
                self._mean_layer_reward
            )
        )

        greedy_better_fraction = (
            None
            if not self._greedy_better_fraction
            else sum(
                self._greedy_better_fraction
            )
            / len(
                self._greedy_better_fraction
            )
        )

        record = PPONetworkTTIMetrics(
            tti_index=(
                self._pending_tti_index
            ),
            num_cells=self.num_cells,
            num_valid_ues=(
                num_valid_ues
            ),
            mean_delivered_mbps=(
                float(
                    delivered_rate
                    .mean()
                    .item()
                )
                / 1.0e6
            ),
            mean_history_mbps=(
                float(
                    history
                    .mean()
                    .item()
                )
                / 1.0e6
            ),
            p05_history_mbps=(
                float(
                    p05.item()
                )
                / 1.0e6
            ),
            median_history_mbps=(
                float(
                    median.item()
                )
                / 1.0e6
            ),
            history_geomean_mbps=(
                history_geomean_bps
                / 1.0e6
            ),
            paper_floor_history_geomean_mbps=(
                floor_geomean_bps
                / 1.0e6
            ),
            reward_population_geomean_mbps=(
                None
                if reward_geomean is None
                else reward_geomean
                / 1.0e6
            ),
            normalized_reward_geomean=(
                normalized_geomean
            ),
            mean_layer_reward=(
                mean_layer_reward
            ),
            greedy_better_fraction=(
                greedy_better_fraction
            ),
        )

        self.records.append(
            record
        )

        with self.csv_path.open(
            "a",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=(
                    self.FIELDNAMES
                ),
            )

            writer.writerow(
                {
                    "tti": (
                        record.tti_index
                    ),
                    "num_cells": (
                        record.num_cells
                    ),
                    "num_valid_ues": (
                        record.num_valid_ues
                    ),
                    "mean_delivered_mbps": (
                        record.mean_delivered_mbps
                    ),
                    "mean_history_mbps": (
                        record.mean_history_mbps
                    ),
                    "p05_history_mbps": (
                        record.p05_history_mbps
                    ),
                    "median_history_mbps": (
                        record.median_history_mbps
                    ),
                    "history_geomean_mbps": (
                        record.history_geomean_mbps
                    ),
                    "paper_floor_history_geomean_mbps": (
                        record
                        .paper_floor_history_geomean_mbps
                    ),
                    "reward_population_geomean_mbps": (
                        record
                        .reward_population_geomean_mbps
                    ),
                    "normalized_reward_geomean": (
                        record
                        .normalized_reward_geomean
                    ),
                    "mean_layer_reward": (
                        record.mean_layer_reward
                    ),
                    "greedy_better_fraction": (
                        record
                        .greedy_better_fraction
                    ),
                }
            )

        self._pending_tti_index = None
        self._seen_cells.clear()
        self._delivered_rate_parts.clear()
        self._history_parts.clear()
        self._reward_geomean.clear()
        self._normalized_geomean.clear()
        self._mean_layer_reward.clear()
        self._greedy_better_fraction.clear()


