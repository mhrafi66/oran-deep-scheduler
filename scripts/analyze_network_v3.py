from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("experiments/stress_tests/network_v3")
RUN_ID = (ROOT / "latest_run_id.txt").read_text().strip()
RUN_DIR = ROOT / RUN_ID
MANIFEST = RUN_DIR / "manifest.tsv"
PARALLEL = Path("experiments/stress_tests/parallel")
START_TTI = 20
END_TTI = 99

with MANIFEST.open() as f:
    manifest = list(csv.DictReader(f, delimiter="\t"))


def fmean(rows, field):
    return statistics.fmean(float(r[field]) for r in rows)


def p95(values):
    xs = sorted(values)
    if not xs:
        return float("nan")
    index = min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)
    return xs[index]


per_run: list[dict[str, object]] = []

for meta in manifest:
    tag = meta["tag"]
    with (PARALLEL / f"{tag}_kpi.csv").open() as f:
        all_kpi = list(csv.DictReader(f))
    with (PARALLEL / f"{tag}_metrics.csv").open() as f:
        all_metrics = list(csv.DictReader(f))

    kpi = [
        r for r in all_kpi
        if START_TTI <= int(r["tti"]) <= END_TTI
    ]
    metrics = [
        r for r in all_metrics
        if START_TTI <= int(r["tti"]) <= END_TTI
    ]

    elapsed = [float(r["elapsed_s"]) for r in metrics]
    gpu = [float(r["peak_gpu_mib"]) for r in metrics]

    per_run.append(
        {
            **meta,
            "delivered_mbps": fmean(kpi, "mean_delivered_mbps"),
            "p05_mbps": fmean(kpi, "p05_history_mbps"),
            "median_mbps": fmean(kpi, "median_history_mbps"),
            "geomean_mbps": fmean(kpi, "paper_floor_history_geomean_mbps"),
            "mean_tti_s": statistics.fmean(elapsed),
            "p95_tti_s": p95(elapsed),
            "max_peak_gpu_mib": max(gpu),
        }
    )

per_run_path = RUN_DIR / "per_run_summary.csv"
fields = list(per_run[0].keys())
with per_run_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(per_run)

# Clean control indexed by replicate, so stressed conditions are compared
# against the corresponding random topology/drop rather than a different one.
clean_by_rep = {
    int(r["replicate"]): r
    for r in per_run
    if r["family"] == "clean" and r["condition"] == "nominal"
}

paired: list[dict[str, object]] = []
for r in per_run:
    clean = clean_by_rep[int(r["replicate"])]
    paired.append(
        {
            "tag": r["tag"],
            "family": r["family"],
            "condition": r["condition"],
            "replicate": r["replicate"],
            "geomean_mbps": r["geomean_mbps"],
            "geomean_retention_pct": 100.0 * float(r["geomean_mbps"]) / float(clean["geomean_mbps"]),
            "delivered_retention_pct": 100.0 * float(r["delivered_mbps"]) / float(clean["delivered_mbps"]),
            "p05_retention_pct": 100.0 * float(r["p05_mbps"]) / float(clean["p05_mbps"]),
        }
    )

paired_path = RUN_DIR / "paired_vs_clean.csv"
with paired_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
    writer.writeheader()
    writer.writerows(paired)

# Aggregate over the five random drops.
groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
for r in per_run:
    groups[(str(r["family"]), str(r["condition"]))].append(r)

paired_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
for r in paired:
    paired_groups[(str(r["family"]), str(r["condition"]))].append(r)

aggregate: list[dict[str, object]] = []

def mean_sd(rows, field):
    xs = [float(r[field]) for r in rows]
    mean = statistics.fmean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mean, sd

for key in sorted(groups):
    rows = groups[key]
    prows = paired_groups[key]
    dm, ds = mean_sd(rows, "delivered_mbps")
    pm, ps = mean_sd(rows, "p05_mbps")
    mm, ms = mean_sd(rows, "median_mbps")
    gm, gs = mean_sd(rows, "geomean_mbps")
    rm, rs = mean_sd(prows, "geomean_retention_pct")
    tm, ts = mean_sd(rows, "mean_tti_s")
    aggregate.append(
        {
            "family": key[0],
            "condition": key[1],
            "n": len(rows),
            "delivered_mean_mbps": dm,
            "delivered_sd_mbps": ds,
            "p05_mean_mbps": pm,
            "p05_sd_mbps": ps,
            "median_mean_mbps": mm,
            "median_sd_mbps": ms,
            "geomean_mean_mbps": gm,
            "geomean_sd_mbps": gs,
            "geomean_retention_mean_pct": rm,
            "geomean_retention_sd_pct": rs,
            "mean_tti_s": tm,
            "mean_tti_sd_s": ts,
        }
    )

aggregate_path = RUN_DIR / "family_summary.csv"
with aggregate_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
    writer.writeheader()
    writer.writerows(aggregate)

# Human-readable Markdown report.
report_path = RUN_DIR / "REPORT.md"
by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
for row in aggregate:
    by_family[str(row["family"])].append(row)

with report_path.open("w") as out:
    out.write(f"# O-RAN Network Stress Report — {RUN_ID}\n\n")
    out.write(f"Evaluation window: TTI {START_TTI}..{END_TTI}.\n\n")
    out.write(f"Total runs: {len(per_run)}.\n\n")
    out.write("Each condition is averaged over five independent topology/traffic replicates. ")
    out.write("Retention is paired against the clean run with the same replicate.\n\n")
    for family in sorted(by_family):
        out.write(f"## {family}\n\n")
        out.write("| Condition | N | Delivered Mbps | P05 Mbps | Median Mbps | GeoMean Mbps | Geo retention | Mean TTI s |\n")
        out.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in by_family[family]:
            out.write(
                f"| {r['condition']} | {r['n']} | "
                f"{float(r['delivered_mean_mbps']):.3f} ± {float(r['delivered_sd_mbps']):.3f} | "
                f"{float(r['p05_mean_mbps']):.3f} ± {float(r['p05_sd_mbps']):.3f} | "
                f"{float(r['median_mean_mbps']):.3f} ± {float(r['median_sd_mbps']):.3f} | "
                f"{float(r['geomean_mean_mbps']):.3f} ± {float(r['geomean_sd_mbps']):.3f} | "
                f"{float(r['geomean_retention_mean_pct']):.2f}% ± {float(r['geomean_retention_sd_pct']):.2f}% | "
                f"{float(r['mean_tti_s']):.2f} |\n"
            )
        out.write("\n")

print(f"Run:             {RUN_ID}")
print(f"Per-run summary: {per_run_path}")
print(f"Paired summary:  {paired_path}")
print(f"Family summary:  {aggregate_path}")
print(f"Markdown report: {report_path}")
