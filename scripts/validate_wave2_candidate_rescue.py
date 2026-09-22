from __future__ import annotations

import csv
import math
from pathlib import Path
import sys


NUM_TTIS = 30

NUM_CELLS = 21


RUN_ROOT = Path(
    "experiments/stress_tests/"
    "wave2_candidate_rescue"
)

PARALLEL_ROOT = Path(
    "experiments/stress_tests/"
    "parallel"
)


CONDITIONS = (
    "stale_candidates__stale_features",
    "fresh_candidates__stale_features",
    "stale_candidates__fresh_features",
    "fresh_candidates__fresh_features",
)


def latest_run_id() -> str:

    path = (
        RUN_ROOT
        / "latest_run_id.txt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}."
        )

    return (
        path
        .read_text()
        .strip()
    )


def read_csv(
    path: Path,
) -> list[
    dict[
        str,
        str,
    ]
]:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing output: {path}"
        )

    with path.open() as file:

        return list(
            csv.DictReader(
                file
            )
        )


def require_finite(
    rows: list[
        dict[
            str,
            str,
        ]
    ],
    fields: tuple[
        str,
        ...
    ],
    *,
    path: Path,
) -> None:

    for row_index, row in enumerate(
        rows
    ):

        for field in fields:

            value = float(
                row[
                    field
                ]
            )

            if not math.isfinite(
                value
            ):
                raise RuntimeError(
                    "Non-finite value: "
                    f"{path}, "
                    f"row={row_index}, "
                    f"field={field}, "
                    f"value={value}."
                )


def validate_tti_rows(
    rows: list[
        dict[
            str,
            str,
        ]
    ],
    *,
    path: Path,
) -> None:

    if len(
        rows
    ) != NUM_TTIS:
        raise RuntimeError(
            f"{path}: expected {NUM_TTIS} rows, "
            f"found {len(rows)}."
        )

    ttis = [
        int(
            row[
                "tti"
            ]
        )
        for row
        in rows
    ]

    expected = list(
        range(
            NUM_TTIS
        )
    )

    if ttis != expected:
        raise RuntimeError(
            f"{path}: TTI sequence is not "
            "0..29 exactly."
        )


def validate_candidate_rows(
    rows: list[
        dict[
            str,
            str,
        ]
    ],
    *,
    path: Path,
) -> None:

    expected_count = (
        NUM_TTIS
        * NUM_CELLS
    )

    if len(
        rows
    ) != expected_count:
        raise RuntimeError(
            f"{path}: expected "
            f"{expected_count} candidate rows, "
            f"found {len(rows)}."
        )

    seen: set[
        tuple[
            int,
            int,
        ]
    ] = set()

    for row in rows:

        key = (
            int(
                row[
                    "tti"
                ]
            ),
            int(
                row[
                    "cell_index"
                ]
            ),
        )

        if key in seen:
            raise RuntimeError(
                f"{path}: duplicate candidate row "
                f"{key}."
            )

        seen.add(
            key
        )

    expected = {
        (
            tti,
            cell,
        )
        for tti
        in range(
            NUM_TTIS
        )
        for cell
        in range(
            NUM_CELLS
        )
    }

    if seen != expected:
        missing = sorted(
            expected
            - seen
        )

        extra = sorted(
            seen
            - expected
        )

        raise RuntimeError(
            f"{path}: candidate coverage mismatch. "
            f"missing={missing[:10]}, "
            f"extra={extra[:10]}."
        )


def validate_condition(
    *,
    run_id: str,
    condition: str,
) -> None:

    tag = (
        f"{run_id}"
        f"__{condition}"
    )

    metrics_path = (
        PARALLEL_ROOT
        / f"{tag}_metrics.csv"
    )

    kpi_path = (
        PARALLEL_ROOT
        / f"{tag}_kpi.csv"
    )

    candidate_path = (
        PARALLEL_ROOT
        / f"{tag}_candidate.csv"
    )


    metrics = read_csv(
        metrics_path
    )

    kpi = read_csv(
        kpi_path
    )

    candidate = read_csv(
        candidate_path
    )


    validate_tti_rows(
        metrics,
        path=metrics_path,
    )

    validate_tti_rows(
        kpi,
        path=kpi_path,
    )

    validate_candidate_rows(
        candidate,
        path=candidate_path,
    )


    require_finite(
        metrics,
        (
            "elapsed_s",
            "mean_td_rate_mbps",
            "mean_wideband_cqi",
            "actor_norm",
            "critic_norm",
            "candidate_jaccard_mean",
            "candidate_fresh_recall_mean",
            "candidate_top1_retention_rate",
            "candidate_set_changed_fraction",
            (
                "candidate_fresh_truth_"
                "pf_retention_mean"
            ),
        ),
        path=metrics_path,
    )


    require_finite(
        kpi,
        (
            "mean_delivered_mbps",
            "mean_history_mbps",
            "p05_history_mbps",
            "median_history_mbps",
            "history_geomean_mbps",
        ),
        path=kpi_path,
    )


    require_finite(
        candidate,
        (
            "candidate_jaccard",
            "candidate_fresh_recall",
            "candidate_top1_retained",
            "candidate_rank_displacement",
            (
                "candidate_fresh_truth_"
                "pf_retention"
            ),
        ),
        path=candidate_path,
    )


    #
    # Evaluation must remain frozen.
    #
    if any(
        int(
            row[
                "ppo_updates_this_tti"
            ]
        ) != 0
        for row
        in metrics
    ):
        raise RuntimeError(
            f"{metrics_path}: PPO updated "
            "during frozen evaluation."
        )


    if any(
        int(
            row[
                "total_ppo_updates"
            ]
        ) != 0
        for row
        in metrics
    ):
        raise RuntimeError(
            f"{metrics_path}: total PPO update "
            "count is not zero."
        )


    actor_norms = [
        float(
            row[
                "actor_norm"
            ]
        )
        for row
        in metrics
    ]

    actor_drift = (
        max(
            actor_norms
        )
        - min(
            actor_norms
        )
    )

    if actor_drift > 1.0e-9:
        raise RuntimeError(
            f"{metrics_path}: actor norm changed "
            f"by {actor_drift}."
        )


    print(
        f"PASS  {condition}"
    )


def main() -> None:

    run_id = (
        sys.argv[1]
        if len(
            sys.argv
        ) > 1
        else latest_run_id()
    )

    print(
        f"Validating run: {run_id}"
    )

    print()

    for condition in CONDITIONS:

        validate_condition(
            run_id=run_id,
            condition=condition,
        )

    print()
    print(
        "ALL 4 WAVE-2 RESCUE RUNS PASSED"
    )


if __name__ == "__main__":
    main()
