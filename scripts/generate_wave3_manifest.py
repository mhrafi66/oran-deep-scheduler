from __future__ import annotations

import csv
import json

from dataclasses import dataclass
from pathlib import Path


OUTPUT = Path(
    "experiments/stress_tests/"
    "wave3_manifest.csv"
)


SCHEDULERS = (
    "ppo",
    "baseline",
    "pf_greedy",
)


#
# Three paired drops for overnight exploration.
#
# Tomorrow, promising effects can be expanded to
# 5-10+ seeds.
#
TOPOLOGY_SEEDS = (
    42,
    314,
    2718,
)


@dataclass(frozen=True)
class Condition:

    family: str

    name: str

    num_ttis: int = 100

    num_ut_per_sector: int = 20

    num_user_slots: int = 4

    csi_delay_ttis: int = 0

    csi_stale_mode: str = "none"

    td_rate_log_noise_std: float = 0.0

    cqi_noise_std: float = 0.0

    ftp_rate_scale: float = 1.0

    runtime_scenario_json: str = ""

    post_csi_interference_power_scale: float = 1.0

    post_csi_serving_power_scale: float = 1.0

    post_csi_failed_bs: str = ""


def scenario_json(
    phases,
) -> str:

    return json.dumps(
        phases,
        separators=(
            ",",
            ":",
        ),
    )


CONDITIONS = (

    # =========================================================
    # CONTROLS
    # =========================================================

    Condition(
        family="control",
        name="clean",
    ),


    # =========================================================
    # OBSERVATION / CSI
    # =========================================================

    Condition(
        family="csi",
        name="csi_all_d1",
        csi_delay_ttis=1,
        csi_stale_mode="all",
    ),

    Condition(
        family="td_noise",
        name="tdnoise_025",
        td_rate_log_noise_std=0.25,
    ),

    Condition(
        family="td_noise",
        name="tdnoise_050",
        td_rate_log_noise_std=0.50,
    ),

    Condition(
        family="td_noise",
        name="tdnoise_075",
        td_rate_log_noise_std=0.75,
    ),

    Condition(
        family="cqi_noise",
        name="cqinoise_1",
        cqi_noise_std=1.0,
    ),

    Condition(
        family="cqi_noise",
        name="cqinoise_2",
        cqi_noise_std=2.0,
    ),


    # =========================================================
    # TRAFFIC LOAD
    # =========================================================

    Condition(
        family="traffic_load",
        name="ftp_load_050",
        ftp_rate_scale=0.5,
    ),

    Condition(
        family="traffic_load",
        name="ftp_load_200",
        ftp_rate_scale=2.0,
    ),

    Condition(
        family="traffic_load",
        name="ftp_load_400",
        ftp_rate_scale=4.0,
    ),


    # =========================================================
    # DYNAMIC LOAD / RECOVERY
    # =========================================================

    Condition(
        family="dynamic_load",
        name="normal_overload_recovery",
        num_ttis=90,
        runtime_scenario_json=(
            scenario_json(
                [
                    {
                        "start_tti": 0,
                        "name": "normal",
                        "ftp_rate_scale": 1.0,
                    },

                    {
                        "start_tti": 30,
                        "name": "overload",
                        "ftp_rate_scale": 4.0,
                    },

                    {
                        "start_tti": 60,
                        "name": "recovery",
                        "ftp_rate_scale": 1.0,
                    },
                ]
            )
        ),
    ),


    # =========================================================
    # STATE LOSS / RESTART
    # =========================================================

    Condition(
        family="restart",
        name="pf_history_restart",
        num_ttis=90,
        runtime_scenario_json=(
            scenario_json(
                [
                    {
                        "start_tti": 0,
                        "name": "normal",
                    },

                    {
                        "start_tti": 30,
                        "name": "pf_reset",
                        "reset_pf_history": True,
                        "pf_history_bps": 1.0e6,
                    },

                    {
                        "start_tti": 60,
                        "name": "post_reset",
                    },
                ]
            )
        ),
    ),

    Condition(
        family="restart",
        name="queue_restart",
        num_ttis=90,
        runtime_scenario_json=(
            scenario_json(
                [
                    {
                        "start_tti": 0,
                        "name": "normal",
                    },

                    {
                        "start_tti": 30,
                        "name": "queue_reset",
                        "reset_ftp_buffers": True,
                        "ftp_buffer_bits": 0.0,
                    },

                    {
                        "start_tti": 60,
                        "name": "post_reset",
                    },
                ]
            )
        ),
    ),


    # =========================================================
    # UE DENSITY
    # =========================================================

    Condition(
        family="ue_density",
        name="ue_density_10",
        num_ut_per_sector=10,
    ),

    Condition(
        family="ue_density",
        name="ue_density_30",
        num_ut_per_sector=30,
    ),

    Condition(
        family="ue_density",
        name="ue_density_40",
        num_ut_per_sector=40,
    ),


    # =========================================================
    # SPATIAL LAYER BUDGET
    # =========================================================

    Condition(
        family="mu_layers",
        name="mu_slots_2",
        num_user_slots=2,
    ),

    Condition(
        family="mu_layers",
        name="mu_slots_6",
        num_user_slots=6,
    ),

    Condition(
        family="mu_layers",
        name="mu_slots_8",
        num_user_slots=8,
    ),


    # =========================================================
    # POST-CSI PHYSICAL INTERFERENCE SHOCK
    #
    # Scheduler sees pre-shock CSI.
    # PHY executes under the changed environment.
    # =========================================================

    Condition(
        family="interference_shock",
        name="interference_x2",
        post_csi_interference_power_scale=2.0,
    ),

    Condition(
        family="interference_shock",
        name="interference_x4",
        post_csi_interference_power_scale=4.0,
    ),

    Condition(
        family="interference_shock",
        name="interference_x10",
        post_csi_interference_power_scale=10.0,
    ),


    # =========================================================
    # SERVING-LINK EXECUTION SHOCK
    # =========================================================

    Condition(
        family="serving_drop",
        name="serving_power_050",
        post_csi_serving_power_scale=0.5,
    ),

    Condition(
        family="serving_drop",
        name="serving_power_025",
        post_csi_serving_power_scale=0.25,
    ),


    # =========================================================
    # SUDDEN BS FAILURE
    #
    # BS 0 is removed at execution time.
    # Its own served cell loses desired signal;
    # other cells lose that interferer.
    # =========================================================

    Condition(
        family="bs_failure",
        name="bs0_failure",
        post_csi_failed_bs="0",
    ),


    # =========================================================
    # COMPOUND FAILURE
    # =========================================================

    Condition(
        family="compound",
        name="overload_plus_csi",
        num_ttis=90,
        csi_delay_ttis=1,
        csi_stale_mode="all",

        runtime_scenario_json=(
            scenario_json(
                [
                    {
                        "start_tti": 0,
                        "name": "normal",
                    },

                    {
                        "start_tti": 30,
                        "name": "overload",
                        "ftp_rate_scale": 4.0,
                    },

                    {
                        "start_tti": 60,
                        "name": "recovery",
                        "ftp_rate_scale": 1.0,
                    },
                ]
            )
        ),
    ),
)


FIELDS = (
    "task_id",
    "family",
    "condition",
    "scheduler_mode",
    "replicate",
    "topology_seed",
    "run_seed",
    "num_eval_ttis",
    "num_ut_per_sector",
    "num_user_slots",
    "csi_delay_ttis",
    "csi_stale_mode",
    "td_rate_log_noise_std",
    "cqi_noise_std",
    "ftp_rate_scale",
    "runtime_scenario_json",
    "post_csi_interference_power_scale",
    "post_csi_serving_power_scale",
    "post_csi_failed_bs",
)


def main() -> None:

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    task_id = 0

    for condition in CONDITIONS:

        for replicate, topology_seed in enumerate(
            TOPOLOGY_SEEDS
        ):

            #
            # IMPORTANT:
            # all 3 schedulers within the same
            # condition/replicate use identical seeds.
            #
            paired_run_seed = (
                100_000
                + 10_000 * replicate
                + 100 * len(
                    rows
                )
            )

            for scheduler in SCHEDULERS:

                rows.append(
                    {
                        "task_id": task_id,

                        "family": (
                            condition.family
                        ),

                        "condition": (
                            condition.name
                        ),

                        "scheduler_mode": (
                            scheduler
                        ),

                        "replicate": replicate,

                        "topology_seed": (
                            topology_seed
                        ),

                        "run_seed": (
                            paired_run_seed
                        ),

                        "num_eval_ttis": (
                            condition.num_ttis
                        ),

                        "num_ut_per_sector": (
                            condition
                            .num_ut_per_sector
                        ),

                        "num_user_slots": (
                            condition
                            .num_user_slots
                        ),

                        "csi_delay_ttis": (
                            condition
                            .csi_delay_ttis
                        ),

                        "csi_stale_mode": (
                            condition
                            .csi_stale_mode
                        ),

                        "td_rate_log_noise_std": (
                            condition
                            .td_rate_log_noise_std
                        ),

                        "cqi_noise_std": (
                            condition
                            .cqi_noise_std
                        ),

                        "ftp_rate_scale": (
                            condition
                            .ftp_rate_scale
                        ),

                        "runtime_scenario_json": (
                            condition
                            .runtime_scenario_json
                        ),

                        (
                            "post_csi_"
                            "interference_power_scale"
                        ): (
                            condition
                            .post_csi_interference_power_scale
                        ),

                        (
                            "post_csi_"
                            "serving_power_scale"
                        ): (
                            condition
                            .post_csi_serving_power_scale
                        ),

                        "post_csi_failed_bs": (
                            condition
                            .post_csi_failed_bs
                        ),
                    }
                )

                task_id += 1

    with OUTPUT.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"Manifest: {OUTPUT}"
    )

    print(
        f"Conditions: {len(CONDITIONS)}"
    )

    print(
        f"Schedulers: {len(SCHEDULERS)}"
    )

    print(
        f"Replicates: {len(TOPOLOGY_SEEDS)}"
    )

    print(
        f"Total jobs: {len(rows)}"
    )


if __name__ == "__main__":
    main()
