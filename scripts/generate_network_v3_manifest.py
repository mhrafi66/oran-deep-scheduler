from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

ROOT = Path("experiments/stress_tests/network_v3")
RUN_ID = "netv3_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = ROOT / RUN_ID
MANIFEST = RUN_DIR / "manifest.tsv"
LATEST = ROOT / "latest_run_id.txt"

RUN_DIR.mkdir(parents=True, exist_ok=False)
ROOT.mkdir(parents=True, exist_ok=True)

# Five independent topology/traffic realizations for exploratory networking sweeps.
# Important findings can later be rerun with 10+ seeds for paper-quality CIs.
REPLICATES = tuple(range(5))
TOPOLOGY_SEEDS = (42, 43, 44, 45, 46)
RUN_SEEDS = (1234, 2234, 3234, 4234, 5234)
CORRUPTION_SEEDS = (918273, 918274, 918275, 918276, 918277)

BASE = {
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

rows: list[dict[str, object]] = []


def safe_token(value: object) -> str:
    return (
        str(value)
        .replace("-", "m")
        .replace(".", "p")
        .replace("+", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def add_condition(
    family: str,
    condition: str,
    *,
    replicates: tuple[int, ...] = REPLICATES,
    **changes: object,
) -> None:
    for rep in replicates:
        config = dict(BASE)
        config.update(changes)
        tag = (
            f"{RUN_ID}__{family}__{condition}__r{rep}"
        )
        rows.append(
            {
                "tag": tag,
                "family": family,
                "condition": condition,
                "replicate": rep,
                "run_seed": RUN_SEEDS[rep],
                "topology_seed": TOPOLOGY_SEEDS[rep],
                "corruption_seed": CORRUPTION_SEEDS[rep],
                **config,
            }
        )


# ------------------------------------------------------------------
# 0. CLEAN CONTROLS
# ------------------------------------------------------------------
add_condition("clean", "nominal")

# ------------------------------------------------------------------
# 1. CSI AGE / MISMATCH
# NOTE: current temporal channel is independent across TTIs. Interpret
# d>0 as mismatch severity, NOT calibrated mobility latency.
# ------------------------------------------------------------------
for delay in (1, 2, 4, 8, 16):
    add_condition("csi_delay", f"d{delay}", delay=delay)

for mode in (
    "all",
    "ppo_radio",
    "td_rate_only",
    "cqi_only",
    "rank_only",
    "precoder_only",
    "none",
):
    add_condition(
        "csi_decomposition",
        f"{mode}_d1",
        delay=1,
        stale_mode=mode,
    )

# ------------------------------------------------------------------
# 2. CQI QUALITY: BIAS, RANDOM ERROR, INFORMATION RESOLUTION
# ------------------------------------------------------------------
for value in (-6, -4, -2, -1, 1, 2, 4, 6):
    add_condition(
        "cqi_bias",
        f"bias_{safe_token(value)}",
        cqi_bias=float(value),
    )

for value in (0.25, 0.5, 1.0, 2.0, 4.0, 6.0):
    add_condition(
        "cqi_noise",
        f"std_{safe_token(value)}",
        cqi_noise=float(value),
    )

for step in (2, 4, 8):
    add_condition(
        "cqi_resolution",
        f"quant_{step}",
        cqi_quant=float(step),
    )
add_condition("cqi_resolution", "flat_subband", flat_cqi=1)

# ------------------------------------------------------------------
# 3. RI / RANK REPORT ERRORS
# ------------------------------------------------------------------
for probability in (0.10, 0.25, 0.50, 0.75, 1.00):
    add_condition(
        "rank_error",
        f"flip_{safe_token(probability)}",
        rank_flip=probability,
    )
add_condition("rank_error", "force_rank1", rank_force=1)
add_condition("rank_error", "force_rank2", rank_force=2)

# ------------------------------------------------------------------
# 4. PMI / PRECODER-DIRECTION CORRUPTION
# ------------------------------------------------------------------
add_condition("precoder_error", "ue_shuffle", prec_ue=1)
add_condition("precoder_error", "rbg_shuffle", prec_rbg=1)
add_condition(
    "precoder_error",
    "ue_and_rbg_shuffle",
    prec_ue=1,
    prec_rbg=1,
)

# ------------------------------------------------------------------
# 5. PF-TDS / FRONTEND RATE UNCERTAINTY
# ------------------------------------------------------------------
for sigma in (0.10, 0.25, 0.50, 0.75, 1.00, 2.00):
    add_condition(
        "td_rate_noise",
        f"logstd_{safe_token(sigma)}",
        td_noise=sigma,
    )

for probability in (0.05, 0.10, 0.25, 0.50, 0.75):
    add_condition(
        "td_rate_dropout",
        f"drop_{safe_token(probability)}",
        td_dropout=probability,
    )

add_condition("td_rate_shuffle", "shuffle", td_shuffle=1)

# ------------------------------------------------------------------
# 6. UE LOAD / POPULATION SHIFT
# ------------------------------------------------------------------
for num_ut in (5, 10, 15, 25, 30, 40):
    add_condition(
        "ue_density",
        f"ut_per_sector_{num_ut}",
        num_ut=num_ut,
    )

# ------------------------------------------------------------------
# 7. MU-MIMO DEPTH SHIFT
# ------------------------------------------------------------------
for layers in (1, 2, 3, 5, 6, 8):
    add_condition(
        "mu_layers",
        f"layers_{layers}",
        user_slots=layers,
    )

# ------------------------------------------------------------------
# 8. TRAFFIC-MIXTURE SHIFT
# ------------------------------------------------------------------
for fb in (0.0, 0.25, 0.75, 1.0):
    add_condition(
        "traffic_mix",
        f"fb_{int(round(100*fb)):03d}",
        fb=fb,
    )

# ------------------------------------------------------------------
# 9. OFFERED-LOAD SHIFT WITH MIXED TRAINING-TYPE TRAFFIC
# ------------------------------------------------------------------
for scale in (0.10, 0.25, 0.50, 2.00, 4.00, 8.00):
    add_condition(
        "mixed_ftp_load",
        f"rate_x{safe_token(scale)}",
        ftp_rate=scale,
    )

# ------------------------------------------------------------------
# 10. ALL-FTP OFFERED-LOAD SHIFT
# ------------------------------------------------------------------
for scale in (0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00):
    add_condition(
        "allftp_load",
        f"rate_x{safe_token(scale)}",
        fb=0.0,
        ftp_rate=scale,
    )

# ------------------------------------------------------------------
# 11. PACKET-SIZE SHIFT AT FIXED ARRIVAL RATE
# ------------------------------------------------------------------
for scale in (0.25, 0.50, 2.00, 4.00):
    add_condition(
        "allftp_packet_size",
        f"size_x{safe_token(scale)}",
        fb=0.0,
        ftp_size=scale,
    )

# ------------------------------------------------------------------
# 12. BURSTINESS AT APPROXIMATELY CONSTANT MEAN OFFERED BIT RATE
# packet_size_scale * arrival_rate_scale ~= 1
# ------------------------------------------------------------------
for size, rate in (
    (0.25, 4.00),
    (0.50, 2.00),
    (2.00, 0.50),
    (4.00, 0.25),
):
    add_condition(
        "allftp_burstiness",
        f"size_x{safe_token(size)}_rate_x{safe_token(rate)}",
        fb=0.0,
        ftp_size=size,
        ftp_rate=rate,
    )

# ------------------------------------------------------------------
# 13. COMBINED SENSOR / FRONTEND FAILURES
# ------------------------------------------------------------------
add_condition(
    "compound_sensor",
    "mild",
    cqi_noise=0.5,
    rank_flip=0.10,
    td_noise=0.10,
)
add_condition(
    "compound_sensor",
    "moderate",
    cqi_noise=1.0,
    rank_flip=0.25,
    prec_rbg=1,
    td_noise=0.25,
)
add_condition(
    "compound_sensor",
    "severe",
    cqi_noise=2.0,
    rank_flip=0.50,
    prec_ue=1,
    td_noise=0.50,
)
add_condition(
    "compound_sensor",
    "extreme",
    cqi_noise=4.0,
    rank_flip=1.00,
    prec_ue=1,
    prec_rbg=1,
    td_noise=1.00,
    td_dropout=0.25,
)

# ------------------------------------------------------------------
# 14. TWO-DIMENSIONAL FAILURE SURFACES
# These are especially useful for heatmaps / failure-frontier plots.
# ------------------------------------------------------------------
for num_ut in (10, 20, 30, 40):
    for sigma in (0.25, 0.50, 1.00):
        add_condition(
            "surface_density_tdnoise",
            f"ut{num_ut}_td{safe_token(sigma)}",
            num_ut=num_ut,
            td_noise=sigma,
        )

for fb in (0.0, 0.5, 1.0):
    for noise in (1.0, 2.0, 4.0):
        add_condition(
            "surface_traffic_cqi",
            f"fb{int(100*fb):03d}_cqi{safe_token(noise)}",
            fb=fb,
            cqi_noise=noise,
        )

for num_ut in (10, 20, 30, 40):
    for layers in (2, 4, 8):
        add_condition(
            "surface_density_layers",
            f"ut{num_ut}_layers{layers}",
            num_ut=num_ut,
            user_slots=layers,
        )

for layers in (2, 4, 6, 8):
    add_condition(
        "surface_layers_precoder",
        f"layers{layers}_ueshuffle",
        user_slots=layers,
        prec_ue=1,
    )
    add_condition(
        "surface_layers_precoder",
        f"layers{layers}_rbgshuffle",
        user_slots=layers,
        prec_rbg=1,
    )

for num_ut in (10, 20, 30, 40):
    for cqi_noise in (1.0, 2.0):
        add_condition(
            "surface_density_cqi",
            f"ut{num_ut}_cqi{safe_token(cqi_noise)}",
            num_ut=num_ut,
            cqi_noise=cqi_noise,
        )

FIELDNAMES = [
    "tag",
    "family",
    "condition",
    "replicate",
    "run_seed",
    "topology_seed",
    "corruption_seed",
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

with MANIFEST.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=FIELDNAMES,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

LATEST.write_text(RUN_ID + "\n")

families = sorted({str(row["family"]) for row in rows})
print(f"Run ID:      {RUN_ID}")
print(f"Experiments: {len(rows)}")
print(f"Families:    {len(families)}")
print(f"Manifest:    {MANIFEST}")
print("Families:")
for family in families:
    count = sum(1 for row in rows if row["family"] == family)
    print(f"  {family:<28} {count:>4}")
