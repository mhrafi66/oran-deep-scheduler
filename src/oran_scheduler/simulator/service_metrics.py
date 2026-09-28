from __future__ import annotations

import math

import torch

from oran_scheduler.rl.ppo_traffic_cell_step import (
    TrafficAwarePPOCellTTIStepResult,
)


SERVICE_CELL_FIELDS = (
    "queue_mean_bits",
    "queue_p50_bits",
    "queue_p95_bits",
    "queue_p99_bits",
    "queue_max_bits",
    "queue_nonempty_fraction",
    "demand_ue_count",
    "served_ue_count",
    "starved_ue_count",
    "starvation_fraction",
    "no_service_streak_mean",
    "no_service_streak_max",
    "starvation_ge5_fraction",
    "starvation_ge10_fraction",
    "starvation_ge20_fraction",
)


SERVICE_AGGREGATE_FIELDS = tuple(
    f"{field}_mean"
    for field
    in SERVICE_CELL_FIELDS
)


def _quantile(
    values: torch.Tensor,
    q: float,
) -> float:

    if values.numel() == 0:
        return 0.0

    return float(
        torch.quantile(
            values.to(
                dtype=torch.float32
            ),
            q,
        ).item()
    )


def _mean_or_zero(
    values: torch.Tensor,
) -> float:

    if values.numel() == 0:
        return 0.0

    return float(
        values
        .to(
            dtype=torch.float32
        )
        .mean()
        .item()
    )


class RuntimeServiceMetrics:
    """
    Track queue backlog and service starvation.

    Queue statistics intentionally exclude Full-Buffer UEs,
    whose finite buffer value is only a scheduler-state proxy.

    Starvation statistics include every real UE that currently
    has data to receive, including Full-Buffer UEs.
    """

    def __init__(
        self,
        *,
        num_cells: int,
    ) -> None:

        if num_cells <= 0:
            raise ValueError(
                "num_cells must be positive."
            )

        self.num_cells = num_cells

        self._no_service_streak: dict[
            int,
            torch.Tensor,
        ] = {}

        self._rows_by_tti: dict[
            int,
            list[
                dict[
                    str,
                    float | int,
                ]
            ],
        ] = {}


    def observe(
        self,
        *,
        tti_index: int,
        cell_index: int,
        result: TrafficAwarePPOCellTTIStepResult,
        full_buffer_mask: torch.Tensor,
    ) -> dict[
        str,
        float | int,
    ]:

        if not (
            0
            <= cell_index
            < self.num_cells
        ):
            raise ValueError(
                "cell_index is invalid."
            )

        valid_mask = (
            result
            .scheduler_observation
            .serving_ue_valid_mask
        )

        delivered_rate = (
            result
            .traffic_service
            .delivered_rate_bps
        )

        had_data = (
            result
            .traffic_service
            .had_data_to_receive
        )

        queue_after = (
            result
            .traffic_service
            .buffer_after_service_bits
        )

        expected_shape = tuple(
            valid_mask.shape
        )

        for name, tensor in {
            "full_buffer_mask": full_buffer_mask,
            "delivered_rate": delivered_rate,
            "had_data": had_data,
            "queue_after": queue_after,
        }.items():

            if tuple(
                tensor.shape
            ) != expected_shape:
                raise ValueError(
                    f"{name} has wrong shape."
                )

        if full_buffer_mask.dtype != torch.bool:
            raise ValueError(
                "full_buffer_mask must use bool."
            )

        ftp_mask = (
            valid_mask
            & (~full_buffer_mask)
        )

        ftp_queue = (
            queue_after[
                ftp_mask
            ]
        )

        demand_mask = (
            valid_mask
            & had_data
        )

        served_mask = (
            demand_mask
            & (
                delivered_rate > 0.0
            )
        )

        starved_mask = (
            demand_mask
            & (~served_mask)
        )

        if cell_index not in (
            self._no_service_streak
        ):
            self._no_service_streak[
                cell_index
            ] = torch.zeros(
                expected_shape,
                dtype=torch.long,
                device=(
                    delivered_rate.device
                ),
            )

        streak = (
            self._no_service_streak[
                cell_index
            ]
        )

        #
        # A UE accumulates starvation only while it has
        # traffic demand and receives zero service.
        #
        streak = torch.where(
            starved_mask,
            streak + 1,
            torch.zeros_like(
                streak
            ),
        )

        self._no_service_streak[
            cell_index
        ] = streak

        valid_streak = streak[
            valid_mask
        ]

        demand_count = int(
            demand_mask
            .sum()
            .item()
        )

        served_count = int(
            served_mask
            .sum()
            .item()
        )

        starved_count = int(
            starved_mask
            .sum()
            .item()
        )

        starvation_fraction = (
            starved_count
            / demand_count
            if demand_count > 0
            else 0.0
        )

        valid_count = int(
            valid_mask
            .sum()
            .item()
        )

        def streak_fraction(
            threshold: int,
        ) -> float:

            if valid_count == 0:
                return 0.0

            return float(
                (
                    valid_streak
                    >= threshold
                )
                .to(
                    dtype=torch.float32
                )
                .mean()
                .item()
            )

        row: dict[
            str,
            float | int,
        ] = {
            "tti": tti_index,
            "cell_index": cell_index,

            "queue_mean_bits": (
                _mean_or_zero(
                    ftp_queue
                )
            ),

            "queue_p50_bits": (
                _quantile(
                    ftp_queue,
                    0.50,
                )
            ),

            "queue_p95_bits": (
                _quantile(
                    ftp_queue,
                    0.95,
                )
            ),

            "queue_p99_bits": (
                _quantile(
                    ftp_queue,
                    0.99,
                )
            ),

            "queue_max_bits": (
                float(
                    ftp_queue.max().item()
                )
                if ftp_queue.numel() > 0
                else 0.0
            ),

            "queue_nonempty_fraction": (
                float(
                    (
                        ftp_queue > 0.0
                    )
                    .to(
                        dtype=torch.float32
                    )
                    .mean()
                    .item()
                )
                if ftp_queue.numel() > 0
                else 0.0
            ),

            "demand_ue_count": (
                demand_count
            ),

            "served_ue_count": (
                served_count
            ),

            "starved_ue_count": (
                starved_count
            ),

            "starvation_fraction": (
                starvation_fraction
            ),

            "no_service_streak_mean": (
                _mean_or_zero(
                    valid_streak
                )
            ),

            "no_service_streak_max": (
                float(
                    valid_streak
                    .max()
                    .item()
                )
                if valid_streak.numel() > 0
                else 0.0
            ),

            "starvation_ge5_fraction": (
                streak_fraction(
                    5
                )
            ),

            "starvation_ge10_fraction": (
                streak_fraction(
                    10
                )
            ),

            "starvation_ge20_fraction": (
                streak_fraction(
                    20
                )
            ),
        }

        for field in SERVICE_CELL_FIELDS:
            value = float(
                row[
                    field
                ]
            )

            if not math.isfinite(
                value
            ):
                raise RuntimeError(
                    f"Non-finite service metric: "
                    f"{field}={value}."
                )

        (
            self._rows_by_tti
            .setdefault(
                tti_index,
                [],
            )
            .append(
                row
            )
        )

        return row


    def pop_tti_rows(
        self,
        tti_index: int,
    ) -> list[
        dict[
            str,
            float | int,
        ]
    ]:

        rows = (
            self._rows_by_tti
            .pop(
                tti_index,
                [],
            )
        )

        if len(
            rows
        ) != self.num_cells:
            raise RuntimeError(
                "Expected one service-metric row "
                "per cell. "
                f"TTI={tti_index}, "
                f"expected={self.num_cells}, "
                f"got={len(rows)}."
            )

        return sorted(
            rows,
            key=lambda row: int(
                row[
                    "cell_index"
                ]
            ),
        )


def aggregate_service_rows(
    rows: list[
        dict[
            str,
            float | int,
        ]
    ],
) -> dict[
    str,
    float,
]:

    if not rows:
        raise ValueError(
            "Cannot aggregate empty service rows."
        )

    return {
        f"{field}_mean": (
            sum(
                float(
                    row[
                        field
                    ]
                )
                for row
                in rows
            )
            / len(
                rows
            )
        )
        for field
        in SERVICE_CELL_FIELDS
    }
