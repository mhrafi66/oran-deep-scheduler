from __future__ import annotations

import csv
from pathlib import Path


OUT = Path(
    "experiments/stress_tests/"
    "robustness_v2/manifest.tsv"
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


BASE = {
    "run_seed": 1234,
    "topology_seed": 42,

    "num_ut": 20,
    "user_slots": 4,

    "fb": 0.50,

    "ftp_rate": 1.0,
    "ftp_size": 1.0,

    "delay": 0,
    "stale_mode": "all",

    "cqi_bias": 0.0,
    "cqi_noise": 0.0,
    "cqi_quant": 0.0,
    "flat_cqi": 0,

    "rank_flip": 0.0,
    "rank_force": 0,

    "prec_ue": 0,
    "prec_rbg": 0,

    "td_noise": 0.0,
    "td_dropout": 0.0,
    "td_shuffle": 0,
}


experiments = []


def add(
    tag,
    **changes,
):
    config = dict(
        BASE
    )

    config.update(
        changes
    )

    config["tag"] = (
        "v2_" + tag
    )

    experiments.append(
        config
    )


# ============================================================
# CLEAN CONTROLS / RANDOM DROPS
# ============================================================

for seed in (
    42,
    43,
    44,
    45,
    46,
):
    add(
        f"clean_topo{seed}",
        topology_seed=seed,
    )


# ============================================================
# CSI STALENESS DECOMPOSITION
# ============================================================

for mode in (
    "all",
    "ppo_radio",
    "td_rate_only",
    "cqi_only",
    "rank_only",
    "precoder_only",
):
    add(
        f"stale_{mode}_d1",
        delay=1,
        stale_mode=mode,
    )


# ============================================================
# CQI BIAS
# ============================================================

for value in (
    -4,
    -2,
    -1,
    1,
    2,
    4,
):
    sign = (
        f"p{value}"
        if value > 0
        else f"m{abs(value)}"
    )

    add(
        f"cqi_bias_{sign}",
        cqi_bias=value,
    )


# ============================================================
# CQI RANDOM ERROR
# ============================================================

for value in (
    0.5,
    1.0,
    2.0,
    4.0,
):
    add(
        (
            "cqi_noise_"
            + str(value).replace(
                ".",
                "p",
            )
        ),
        cqi_noise=value,
    )


# ============================================================
# CQI INFORMATION RESOLUTION
# ============================================================

for step in (
    2,
    4,
    8,
):
    add(
        f"cqi_quant_{step}",
        cqi_quant=step,
    )

add(
    "cqi_flat_subband",
    flat_cqi=1,
)


# ============================================================
# RI / RANK ERRORS
# ============================================================

for probability in (
    0.10,
    0.25,
    0.50,
    1.00,
):
    add(
        (
            "rank_flip_"
            + str(probability).replace(
                ".",
                "p",
            )
        ),
        rank_flip=probability,
    )

add(
    "rank_force_1",
    rank_force=1,
)

add(
    "rank_force_2",
    rank_force=2,
)


# ============================================================
# PMI / PRECODER REPORT ERRORS
# ============================================================

add(
    "precoder_ue_shuffle",
    prec_ue=1,
)

add(
    "precoder_rbg_shuffle",
    prec_rbg=1,
)

add(
    "precoder_both_shuffle",
    prec_ue=1,
    prec_rbg=1,
)


# ============================================================
# PF-TDS RATE / CANDIDATE-FRONT-END ROBUSTNESS
# ============================================================

for sigma in (
    0.25,
    0.50,
    1.00,
    2.00,
):
    add(
        (
            "td_rate_noise_"
            + str(sigma).replace(
                ".",
                "p",
            )
        ),
        td_noise=sigma,
    )

for probability in (
    0.10,
    0.25,
    0.50,
):
    add(
        (
            "td_rate_dropout_"
            + str(probability).replace(
                ".",
                "p",
            )
        ),
        td_dropout=probability,
    )

add(
    "td_rate_shuffle",
    td_shuffle=1,
)


# ============================================================
# UE POPULATION SHIFT
# ============================================================

for num_ut in (
    10,
    15,
    25,
    30,
    40,
):
    add(
        f"ue_density_{num_ut}",
        num_ut=num_ut,
    )


# ============================================================
# MU-MIMO DEPTH SHIFT
# ============================================================

for layers in (
    1,
    2,
    6,
    8,
):
    add(
        f"user_slots_{layers}",
        user_slots=layers,
    )


# ============================================================
# TRAFFIC-MIXTURE SHIFT
# ============================================================

for fb in (
    0.00,
    0.25,
    0.75,
    1.00,
):
    add(
        (
            "traffic_fb_"
            + f"{int(round(100 * fb)):03d}"
        ),
        fb=fb,
    )


# ============================================================
# FTP3 LOAD SHIFT — mixed 50/50
# ============================================================

for scale in (
    0.25,
    0.50,
    2.00,
    4.00,
):
    add(
        (
            "mixed_ftp_rate_"
            + str(scale).replace(
                ".",
                "p",
            )
        ),
        ftp_rate=scale,
    )


# ============================================================
# FTP3 LOAD SHIFT — all FTP3
# ============================================================

for scale in (
    0.25,
    0.50,
    1.00,
    2.00,
    4.00,
):
    add(
        (
            "allftp_rate_"
            + str(scale).replace(
                ".",
                "p",
            )
        ),
        fb=0.0,
        ftp_rate=scale,
    )


# ============================================================
# FTP3 BURSTINESS
#
# Approximate same mean offered bit rate:
#
# packet size x rate scale ~= 1
# ============================================================

for size, rate in (
    (0.50, 2.00),
    (2.00, 0.50),
    (4.00, 0.25),
):
    add(
        (
            "allftp_burst_"
            + str(size).replace(
                ".",
                "p",
            )
        ),
        fb=0.0,
        ftp_size=size,
        ftp_rate=rate,
    )


# ============================================================
# COMBINED SENSOR FAILURES
# ============================================================

add(
    "combined_csi_moderate",
    cqi_noise=1.0,
    rank_flip=0.10,
    prec_rbg=1,
)

add(
    "combined_csi_severe",
    cqi_noise=2.0,
    rank_flip=0.25,
    prec_ue=1,
    td_noise=0.50,
)


# ============================================================
# COMPOUND OOD CONDITIONS
# ============================================================

add(
    "ood_density30_layers8",
    num_ut=30,
    user_slots=8,
)

add(
    "ood_density30_cqi2",
    num_ut=30,
    cqi_noise=2.0,
)

add(
    "ood_fb100_cqi2",
    fb=1.0,
    cqi_noise=2.0,
)

add(
    "ood_ftp4x_cqi2",
    fb=0.0,
    ftp_rate=4.0,
    cqi_noise=2.0,
)

add(
    "ood_layers8_cqi2",
    user_slots=8,
    cqi_noise=2.0,
)

add(
    "ood_layers8_precoder",
    user_slots=8,
    prec_ue=1,
)

add(
    "ood_density30_tdnoise1",
    num_ut=30,
    td_noise=1.0,
)


FIELDNAMES = [
    "tag",
    "run_seed",
    "topology_seed",
    "num_ut",
    "user_slots",
    "fb",
    "ftp_rate",
    "ftp_size",
    "delay",
    "stale_mode",
    "cqi_bias",
    "cqi_noise",
    "cqi_quant",
    "flat_cqi",
    "rank_flip",
    "rank_force",
    "prec_ue",
    "prec_rbg",
    "td_noise",
    "td_dropout",
    "td_shuffle",
]


with OUT.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=FIELDNAMES,
        delimiter="\t",
        lineterminator="\n",
    )

    writer.writeheader()

    writer.writerows(
        experiments
    )


print(
    f"Wrote {len(experiments)} experiments:"
)

print(
    OUT
)
