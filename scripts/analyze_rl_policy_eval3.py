from __future__ import annotations

import csv
import math
import os
import statistics

from collections import defaultdict

from pathlib import Path


EVALUATIONS = (
    "representation_capacity",
    "neuron_ablation",
    "parameter_noise",
    "compression",
    "quantization",
    "normalization_contract",
)

EXPECTED_TASKS = (
    len(EVALUATIONS)
    * 5
)


def as_float(
    value,
):

    try:

        if value == "":
            return None

        number = float(
            value
        )

        return (
            number
            if math.isfinite(
                number
            )
            else None
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def avg(
    values,
):

    return (
        statistics.mean(
            values
        )
        if values
        else math.nan
    )


def std(
    values,
):

    return (
        statistics.stdev(
            values
        )
        if len(values) >= 2
        else math.nan
    )


def main():

    root = Path(
        os.getenv(
            "RL_EVAL3_OUTPUT_ROOT",

            str(
                Path.home()
                / "oran_experiments"
                / "rl_eval3"
            ),
        )
    ).expanduser()


    run_id = os.getenv(
        "RL_EVAL3_RUN_ID",
        "",
    ).strip()


    if not run_id:

        latest = (
            root
            / "latest_run_id.txt"
        )

        if not latest.exists():

            raise RuntimeError(
                "Set RL_EVAL3_RUN_ID "
                "or create latest_run_id.txt"
            )

        run_id = (
            latest
            .read_text()
            .strip()
        )


    run_dir = (
        root
        / run_id
    )


    files = sorted(
        run_dir.glob(
            "task_*_seed_*.csv"
        )
    )


    rows = []

    task_ids = set()


    for path in files:

        with path.open() as handle:

            file_rows = list(
                csv.DictReader(
                    handle
                )
            )


        if not file_rows:
            continue


        task_ids.add(
            int(
                file_rows[
                    0
                ][
                    "task_id"
                ]
            )
        )

        rows.extend(
            file_rows
        )


    missing = sorted(
        set(
            range(
                EXPECTED_TASKS
            )
        )
        -
        task_ids
    )


    print(
        "=" * 88
    )

    print(
        "RL POLICY EVALUATION — SET 3 AUDIT"
    )

    print(
        "=" * 88
    )

    print(
        f"Run ID:          {run_id}"
    )

    print(
        "Task CSVs:       "
        f"{len(task_ids)}/"
        f"{EXPECTED_TASKS}"
    )

    print(
        f"Total data rows: {len(rows)}"
    )

    print(
        "Missing task IDs:  "
        +
        (
            ",".join(
                map(
                    str,
                    missing,
                )
            )
            if missing
            else "none"
        )
    )


    if not rows:

        raise RuntimeError(
            "No Set-3 rows found"
        )


    # ========================================================
    # MERGED RAW TABLE
    # ========================================================

    keys = []

    seen = set()


    for row in rows:

        for key in row:

            if key not in seen:

                seen.add(
                    key
                )

                keys.append(
                    key
                )


    merged_path = (
        run_dir
        / "rl_eval3_all_rows.csv"
    )


    with merged_path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=keys,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # ========================================================
    # AGGREGATE EACH CONDITION ACROSS THE FIVE SEEDS
    # ========================================================

    groups = defaultdict(
        list
    )


    for row in rows:

        groups[
            (
                row[
                    "evaluation"
                ],

                row.get(
                    "condition",
                    "",
                ),
            )
        ].append(
            row
        )


    metadata = {
        "run_id",
        "task_id",
        "evaluation",
        "replicate",
        "seed",
        "checkpoint",
        "checkpoint_tti",
        "ppo_updates",
        "device",
        "condition",
    }


    summary = []


    for (
        evaluation,
        condition,
    ), group in sorted(
        groups.items()
    ):

        output = {
            "evaluation":
                evaluation,

            "condition":
                condition,

            "n_seed_rows":
                len(
                    group
                ),
        }


        numeric_keys = (
            set()
            .union(
                *(
                    set(
                        row
                    )
                    - metadata

                    for row
                    in group
                )
            )
        )


        for key in sorted(
            numeric_keys
        ):

            values = [
                number

                for row
                in group

                if (
                    number := as_float(
                        row.get(
                            key,
                            "",
                        )
                    )
                )
                is not None
            ]


            if values:

                output[
                    f"{key}_mean"
                ] = avg(
                    values
                )

                output[
                    f"{key}_sd"
                ] = std(
                    values
                )


        summary.append(
            output
        )


    summary_keys = []

    seen = set()


    for row in summary:

        for key in row:

            if key not in seen:

                seen.add(
                    key
                )

                summary_keys.append(
                    key
                )


    summary_path = (
        run_dir
        / "rl_eval3_summary_by_condition.csv"
    )


    with summary_path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=summary_keys,
        )

        writer.writeheader()

        writer.writerows(
            summary
        )


    # ========================================================
    # HUMAN-READABLE HIGHLIGHTS
    # ========================================================

    print()

    print(
        "=" * 88
    )

    print(
        "REPRESENTATION CAPACITY"
    )

    print(
        "=" * 88
    )


    for row in summary:

        if (
            row[
                "evaluation"
            ]
            ==
            "representation_capacity"
        ):

            print(
                f"{row['condition']:<18s} "

                "effective_rank="
                f"{row.get('effective_rank_mean', math.nan):.3f} "

                "stable_rank="
                f"{row.get('stable_rank_mean', math.nan):.3f} "

                "zero="
                f"{row.get('zero_activation_pct_mean', math.nan):.2f}%"
            )


    neuron = [
        row

        for row
        in summary

        if (
            row[
                "evaluation"
            ]
            ==
            "neuron_ablation"

            and

            "_neuron_"
            in row[
                "condition"
            ]
        )
    ]


    neuron.sort(
        key=lambda row:
            row.get(
                "rbg_flip_pct_mean",
                -math.inf,
            ),

        reverse=True,
    )


    print()

    print(
        "=" * 88
    )

    print(
        "TOP-10 SINGLE-NEURON SENSITIVITY"
    )

    print(
        "=" * 88
    )


    for row in neuron[
        :10
    ]:

        print(
            f"{row['condition']:<24s} "

            "RBG flip="
            f"{row.get('rbg_flip_pct_mean', math.nan):7.3f}% "

            "whole change="
            f"{row.get('whole_schedule_change_pct_mean', math.nan):7.3f}% "

            "TV="
            f"{row.get('mean_tv_mean', math.nan):.6f}"
        )


    print()

    print(
        "=" * 88
    )

    print(
        "PARAMETER NOISE"
    )

    print(
        "=" * 88
    )


    for row in summary:

        if (
            row[
                "evaluation"
            ]
            ==
            "parameter_noise"
        ):

            print(
                f"{row['condition']:<24s} "

                "RBG flip="
                f"{row.get('rbg_flip_pct_mean', math.nan):7.3f}% "

                "whole change="
                f"{row.get('whole_schedule_change_pct_mean', math.nan):7.3f}%"
            )


    print()

    print(
        "=" * 88
    )

    print(
        "COMPRESSION / QUANTIZATION"
    )

    print(
        "=" * 88
    )


    for row in summary:

        if (
            row[
                "evaluation"
            ]
            in {
                "compression",
                "quantization",
            }
        ):

            print(
                f"{row['evaluation']:<13s} "

                f"{row['condition']:<38s} "

                "RBG flip="
                f"{row.get('rbg_flip_pct_mean', math.nan):7.3f}% "

                "TV="
                f"{row.get('mean_tv_mean', math.nan):.6f}"
            )


    norm = [
        row

        for row
        in summary

        if (
            row[
                "evaluation"
            ]
            ==
            "normalization_contract"
        )
    ]


    norm.sort(
        key=lambda row:
            row.get(
                "rbg_flip_pct_mean",
                -math.inf,
            ),

        reverse=True,
    )


    print()

    print(
        "=" * 88
    )

    print(
        "TOP NORMALIZATION-CONTRACT SHIFTS"
    )

    print(
        "=" * 88
    )


    for row in norm[
        :15
    ]:

        print(
            f"{row['condition']:<42s} "

            "RBG flip="
            f"{row.get('rbg_flip_pct_mean', math.nan):7.3f}% "

            "whole change="
            f"{row.get('whole_schedule_change_pct_mean', math.nan):7.3f}%"
        )


    print()

    print(
        f"Merged rows:     {merged_path}"
    )

    print(
        f"Condition table: {summary_path}"
    )


    if (
        len(
            task_ids
        )
        ==
        EXPECTED_TASKS
    ):

        print(
            "RL_POLICY_EVAL3_DATASET_PASS"
        )

    else:

        print(
            "RL_POLICY_EVAL3_DATASET_INCOMPLETE"
        )


if __name__ == "__main__":
    main()
