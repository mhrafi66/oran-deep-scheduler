from __future__ import annotations

import csv
from pathlib import Path
import sys


ANALYSIS_START_TTI = 5


RUN_ROOT = Path(
    "experiments/stress_tests/"
    "wave2_candidate_rescue"
)

PARALLEL_ROOT = Path(
    "experiments/stress_tests/"
    "parallel"
)


CONDITIONS = (
    (
        "stale_candidates__stale_features",
        "stale cand / stale feat",
    ),
    (
        "fresh_candidates__stale_features",
        "fresh cand / stale feat",
    ),
    (
        "stale_candidates__fresh_features",
        "stale cand / fresh feat",
    ),
    (
        "fresh_candidates__fresh_features",
        "fresh cand / fresh feat",
    ),
)


def latest_run_id() -> str:

    return (
        RUN_ROOT
        .joinpath(
            "latest_run_id.txt"
        )
        .read_text()
        .strip()
    )


def load_csv(
    path: Path,
) -> list[
    dict[
        str,
        str,
    ]
]:

    with path.open() as file:

        return list(
            csv.DictReader(
                file
            )
        )


def analysis_rows(
    rows: list[
        dict[
            str,
            str,
        ]
    ],
) -> list[
    dict[
        str,
        str,
    ]
]:

    selected = [
        row
        for row
        in rows
        if int(
            row[
                "tti"
            ]
        ) >= ANALYSIS_START_TTI
    ]

    if not selected:
        raise RuntimeError(
            "No rows remain after the "
            "analysis-start filter."
        )

    return selected


def mean(
    rows: list[
        dict[
            str,
            str,
        ]
    ],
    field: str,
) -> float:

    return sum(
        float(
            row[
                field
            ]
        )
        for row
        in rows
    ) / len(
        rows
    )


def safe_recovery(
    *,
    value: float,
    native: float,
    full: float,
) -> float:

    denominator = (
        full
        - native
    )

    if abs(
        denominator
    ) < 1.0e-12:
        return 0.0

    return (
        (
            value
            - native
        )
        / denominator
    )


def summarize_condition(
    *,
    run_id: str,
    key: str,
    label: str,
) -> dict[
    str,
    float | str,
]:

    tag = (
        f"{run_id}"
        f"__{key}"
    )

    metrics = analysis_rows(
        load_csv(
            PARALLEL_ROOT
            / f"{tag}_metrics.csv"
        )
    )

    kpi = analysis_rows(
        load_csv(
            PARALLEL_ROOT
            / f"{tag}_kpi.csv"
        )
    )


    return {
        "condition": label,

        "mean_delivered_mbps": mean(
            kpi,
            "mean_delivered_mbps",
        ),

        "p05_history_mbps": mean(
            kpi,
            "p05_history_mbps",
        ),

        "median_history_mbps": mean(
            kpi,
            "median_history_mbps",
        ),

        "history_geomean_mbps": mean(
            kpi,
            "history_geomean_mbps",
        ),

        "candidate_jaccard": mean(
            metrics,
            "candidate_jaccard_mean",
        ),

        "candidate_recall": mean(
            metrics,
            "candidate_fresh_recall_mean",
        ),

        "top1_retention": mean(
            metrics,
            "candidate_top1_retention_rate",
        ),

        "set_changed_fraction": mean(
            metrics,
            "candidate_set_changed_fraction",
        ),

        "candidate_pf_retention": mean(
            metrics,
            (
                "candidate_fresh_truth_"
                "pf_retention_mean"
            ),
        ),

        "top1_pf_ratio": mean(
            metrics,
            (
                "candidate_stressed_top1_"
                "fresh_pf_ratio_mean"
            ),
        ),
    }


def main() -> None:

    run_id = (
        sys.argv[1]
        if len(
            sys.argv
        ) > 1
        else latest_run_id()
    )


    summaries = [
        summarize_condition(
            run_id=run_id,
            key=key,
            label=label,
        )
        for key, label
        in CONDITIONS
    ]


    native_geo = float(
        summaries[
            0
        ][
            "history_geomean_mbps"
        ]
    )

    full_geo = float(
        summaries[
            3
        ][
            "history_geomean_mbps"
        ]
    )


    print()
    print(
        "WAVE-2 CANDIDATE RESCUE"
    )

    print(
        "=" * 139
    )

    print(
        f"Run: {run_id}"
    )

    print(
        "Analysis window: "
        f"TTI {ANALYSIS_START_TTI}..29"
    )

    print()


    header = (
        f"{'condition':27s}"
        f"{'deliv':>9s}"
        f"{'P05':>9s}"
        f"{'median':>9s}"
        f"{'geo':>9s}"
        f"{'geo rescue':>12s}"
        f"{'Jaccard':>10s}"
        f"{'recall':>10s}"
        f"{'top1':>9s}"
        f"{'PF ret':>10s}"
        f"{'changed':>10s}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )


    for summary in summaries:

        geo = float(
            summary[
                "history_geomean_mbps"
            ]
        )

        recovery = safe_recovery(
            value=geo,
            native=native_geo,
            full=full_geo,
        )

        print(
            f"{str(summary['condition']):27s}"
            f"{float(summary['mean_delivered_mbps']):9.3f}"
            f"{float(summary['p05_history_mbps']):9.3f}"
            f"{float(summary['median_history_mbps']):9.3f}"
            f"{geo:9.3f}"
            f"{100.0 * recovery:11.1f}%"
            f"{float(summary['candidate_jaccard']):10.4f}"
            f"{float(summary['candidate_recall']):10.4f}"
            f"{float(summary['top1_retention']):9.4f}"
            f"{float(summary['candidate_pf_retention']):10.4f}"
            f"{float(summary['set_changed_fraction']):10.4f}"
        )


    candidate_rescue_geo = float(
        summaries[
            1
        ][
            "history_geomean_mbps"
        ]
    )

    feature_rescue_geo = float(
        summaries[
            2
        ][
            "history_geomean_mbps"
        ]
    )


    print()
    print(
        "Mechanism contrasts"
    )

    print(
        "-------------------"
    )

    print(
        "Candidate rescue effect:"
    )

    print(
        "  fresh candidates / stale features"
        " - native"
    )

    print(
        "  GeoMean delta = "
        f"{candidate_rescue_geo - native_geo:+.6f} Mbps"
    )

    print()

    print(
        "PPO-feature rescue effect:"
    )

    print(
        "  stale candidates / fresh features"
        " - native"
    )

    print(
        "  GeoMean delta = "
        f"{feature_rescue_geo - native_geo:+.6f} Mbps"
    )

    print()

    print(
        "Full rescue effect:"
    )

    print(
        "  fresh candidates / fresh features"
        " - native"
    )

    print(
        "  GeoMean delta = "
        f"{full_geo - native_geo:+.6f} Mbps"
    )

    print()


if __name__ == "__main__":
    main()
