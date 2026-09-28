from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path("experiments/stress_tests/network_v3")
RUN_ID = (ROOT / "latest_run_id.txt").read_text().strip()
RUN_DIR = ROOT / RUN_ID
MANIFEST = RUN_DIR / "manifest.tsv"
PARALLEL = Path("experiments/stress_tests/parallel")
EXPECTED_TTIS = list(range(100))

with MANIFEST.open() as f:
    manifest = list(csv.DictReader(f, delimiter="\t"))

failures: list[str] = []
passed = 0
missing = 0

print(f"Validating run: {RUN_ID}")
print("=" * 100)

for row in manifest:
    tag = row["tag"]
    kpi_path = PARALLEL / f"{tag}_kpi.csv"
    metrics_path = PARALLEL / f"{tag}_metrics.csv"

    reasons: list[str] = []
    if not kpi_path.exists() or not metrics_path.exists():
        missing += 1
        reasons.append("missing output")
    else:
        try:
            with kpi_path.open() as f:
                kpi = list(csv.DictReader(f))
            with metrics_path.open() as f:
                metrics = list(csv.DictReader(f))

            if len(kpi) != 100:
                reasons.append(f"KPI rows={len(kpi)}")
            if len(metrics) != 100:
                reasons.append(f"metrics rows={len(metrics)}")

            kpi_ttis = [int(r["tti"]) for r in kpi]
            metric_ttis = [int(r["tti"]) for r in metrics]
            if kpi_ttis != EXPECTED_TTIS:
                reasons.append("KPI TTIs != 0..99")
            if metric_ttis != EXPECTED_TTIS:
                reasons.append("metrics TTIs != 0..99")

            numeric_kpi = (
                "mean_delivered_mbps",
                "p05_history_mbps",
                "median_history_mbps",
                "paper_floor_history_geomean_mbps",
            )
            for name in numeric_kpi:
                for r in kpi:
                    if not math.isfinite(float(r[name])):
                        reasons.append(f"nonfinite {name}")
                        break

            updates = [int(r["total_ppo_updates"]) for r in metrics]
            if any(x != 0 for x in updates):
                reasons.append("PPO learning occurred")

            actor_norms = [float(r["actor_norm"]) for r in metrics]
            if any(not math.isfinite(x) for x in actor_norms):
                reasons.append("nonfinite actor norm")
            elif max(actor_norms) - min(actor_norms) > 1e-10:
                reasons.append("actor norm changed")

        except Exception as exc:
            reasons.append(f"parse error: {exc}")

    if reasons:
        failures.append(tag + ": " + "; ".join(reasons))
        print(f"FAIL {tag}: {'; '.join(reasons)}")
    else:
        passed += 1

print("=" * 100)
print(f"Manifest rows: {len(manifest)}")
print(f"Passed:        {passed}")
print(f"Failed:        {len(failures)}")
print(f"Missing:       {missing}")

if failures:
    fail_path = RUN_DIR / "validation_failures.txt"
    fail_path.write_text("\n".join(failures) + "\n")
    print(f"Failure list:  {fail_path}")
    raise SystemExit(1)

print("ALL NETWORK V3 RUNS PASSED")
