import csv
import json
import statistics as st
from pathlib import Path

base = Path("results/benchmark_v2_plans/campaign31")
plan = base / "mac_m4_pilot.csv"
status_dir = base / "execution33/mac_m4/pilot"
result_root = Path("results/benchmark_v2_vision/mac_m4")

with plan.open(newline="") as f:
    planned = {r["run_id"]: r for r in csv.DictReader(f)}

report = []
failures = []

for status_path in sorted(status_dir.glob("*.json")):
    status = json.loads(status_path.read_text())
    plan_id = status["plan_run_id"]

    if plan_id not in planned:
        raise RuntimeError(f"Unknown plan ID: {plan_id}")

    p = planned[plan_id]

    if status["status"] != "ok":
        failures.append(plan_id)
        continue

    worker_path = result_root / "jobs" / (
        status["runner_run_id"] + ".json"
    )

    if not worker_path.is_file():
        raise RuntimeError(f"Missing worker result: {worker_path}")

    measurement = json.loads(worker_path.read_text())

    if measurement["status"] != "ok":
        raise RuntimeError(f"Unexpected worker status: {worker_path}")

    raw_path = result_root / measurement["raw_latency_file"]

    with raw_path.open(newline="") as f:
        raw = list(csv.DictReader(f))

    values = [float(r["latency_ms"]) for r in raw]

    if len(values) != int(p["steps"]) * int(p["repeats"]):
        raise RuntimeError(f"Sample count mismatch: {plan_id}")

    if abs(st.mean(values) - measurement["latency_mean_ms"]) > 0.001:
        raise RuntimeError(f"Latency mismatch: {plan_id}")

    report.append({
        "plan_run_id": plan_id,
        "worker_run_id": status["runner_run_id"],
        "model": p["model"],
        "profile": p["profile"],
        "placement": p["category"],
        "mapping": p["mapping"],
        "threads": p["threads"],
        "process_run": p["process_run"],
        "latency_mean_ms": measurement["latency_mean_ms"],
        "latency_p50_ms": measurement["latency_p50_ms"],
        "latency_p95_ms": measurement["latency_p95_ms"],
        "throughput_img_s": measurement["throughput_img_s"],
        "sample_count": len(values),
        "wall_time_sec": status["wall_time_sec"],
        "numerical_pass": measurement["numerical_pass"],
        "weights": measurement["weights"],
        "accelerator": measurement["accelerator"],
        "raw_latency_file": str(raw_path),
    })

out = base / "mac34_pilot_report.csv"

if report:
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)

print("=== MAC-34 CAMPAIGN REPORT ===")
print("Completed:", len(report))
print("Failed:", len(failures))
print("Remaining pilot jobs:", len(planned) - len(report) - len(failures))

for r in sorted(report, key=lambda x: float(x["latency_mean_ms"])):
    print(
        f"{r['model']:12s} "
        f"{r['placement']:10s} "
        f"T={r['threads']:2s} "
        f"latency={float(r['latency_mean_ms']):7.3f}ms "
        f"wall={float(r['wall_time_sec']):5.2f}s"
    )

if report:
    print(
        "\nTotal measured wall time:",
        round(sum(float(r["wall_time_sec"]) for r in report), 2),
        "seconds"
    )

print("Saved:", out)
