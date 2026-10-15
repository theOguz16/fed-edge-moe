import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PLAN_DIR = BASE / "results/benchmark_v2_plans/campaign31"
RUNNER = BASE / "benchmarks/vision_partition_runner29.py"
VERSION = "vision-partition-v2-mac29"

p = argparse.ArgumentParser()
p.add_argument("--machine", default="mac_m4")
p.add_argument("--tier", default="pilot")
p.add_argument("--model", default="resnet50")
p.add_argument("--profile", default="stage6")
p.add_argument("--categories", default="full_cpu,full_gpu")
p.add_argument("--max-jobs", type=int, default=4)
p.add_argument("--timeout", type=int, default=1200)
p.add_argument("--execute", action="store_true")
p.add_argument("--retry-failed", action="store_true")
p.add_argument("--allow-formal", action="store_true")
a = p.parse_args()

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)

def runner_id(job):
    payload = json.dumps(
        job, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(
        payload.encode()
    ).hexdigest()[:20]

def make_job(row):
    return {
        "protocol": VERSION,
        "machine": row["machine"],
        "model": row["model"],
        "profile": row["profile"],
        "manifest_sha256": row["manifest_sha256"],
        "mapping_name":
            f"{row['tier']}_{row['candidate_id']}",
        "mapping": row["mapping"],
        "accelerator": row["accelerator"],
        "threads": int(row["threads"]),
        "size": int(row["size"]),
        "batch": int(row["batch"]),
        "warmup": int(row["warmup"]),
        "steps": int(row["steps"]),
        "repeats": int(row["repeats"]),
        "process_run": int(row["process_run"]),
        "seed": 42,
        "weights": row["weights"],
        "precision": row["precision"],
        "measurement_mode":
            "sequential_cpu_input_cpu_output",
    }

def check_environment():
    if a.machine == "mac_m4":
        if platform.system() != "Darwin":
            raise RuntimeError("mac_m4 requires macOS")

        status = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout

        if "AC Power" not in status:
            raise RuntimeError(
                "Mac is not connected to AC power. "
                "Plug in the charger before execution."
            )

    free_gb = shutil.disk_usage(BASE).free / 1e9

    if free_gb < 10:
        raise RuntimeError(
            f"Insufficient free disk space: {free_gb:.1f} GB"
        )

    print(f"Free disk: {free_gb:.2f} GB")

plan_file = PLAN_DIR / f"{a.machine}_{a.tier}.csv"

if not plan_file.is_file():
    raise SystemExit(f"Plan not found: {plan_file}")

rows = read_csv(plan_file)
categories = set(a.categories.split(","))

selected = [
    r for r in rows
    if (a.model == "all" or r["model"] == a.model)
    and (a.profile == "all" or r["profile"] == a.profile)
    and (
        a.categories == "all"
        or r["category"] in categories
    )
]

if a.max_jobs < 1:
    raise SystemExit("--max-jobs must be positive")

selected = selected[:a.max_jobs]

output_root = (
    BASE / "results/benchmark_v2_vision" / a.machine
)
campaign_root = (
    PLAN_DIR / "execution33" / a.machine / a.tier
)

print("=== MAC-33 CAMPAIGN BRIDGE ===")
print("Machine:", a.machine)
print("Tier:", a.tier)
print("Model:", a.model)
print("Profile:", a.profile)
print("Selected jobs:", len(selected))
print("Mode:", "EXECUTE" if a.execute else "DRY RUN")

for i, row in enumerate(selected, 1):
    job = make_job(row)
    manifest_path = (
        BASE / "configs/partition_manifests_v2"
        / f"{job['model']}__{job['profile']}.json"
    )

    digest = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()

    if digest != job["manifest_sha256"]:
        raise RuntimeError(
            f"Manifest hash mismatch: {manifest_path}"
        )

    print(
        f"{i:2d}. {row['model']:15s} "
        f"{row['profile']:14s} "
        f"{row['category']:11s} "
        f"T={row['threads']:2s} "
        f"run={row['process_run']} "
        f"id={row['run_id']}",
        flush=True,
    )

if not a.execute:
    print("\nDRY RUN: no inference executed.")
    sys.exit(0)

if a.tier != "pilot" and not a.allow_formal:
    raise SystemExit(
        "Formal campaign locked. "
        "Use --allow-formal only after pilot review."
    )

if a.tier == "pilot" and a.max_jobs > 12:
    raise SystemExit(
        "Pilot safety limit: --max-jobs must be <= 12"
    )

check_environment()

campaign_root.mkdir(parents=True, exist_ok=True)
completed_count = 0
failed_count = 0

for index, row in enumerate(selected, 1):
    plan_id = row["run_id"]
    status_path = campaign_root / f"{plan_id}.json"

    job = make_job(row)
    rid = runner_id(job)

    worker_result = output_root / "jobs" / f"{rid}.json"

    if worker_result.exists():
        try:
            existing = json.loads(worker_result.read_text())
            if existing.get("status") == "ok":
                print(
                    f"[{index}/{len(selected)}] "
                    f"SKIP completed {plan_id}",
                    flush=True,
                )
                completed_count += 1
                continue
        except (OSError, ValueError):
            pass

    if status_path.exists() and not a.retry_failed:
        try:
            previous = json.loads(status_path.read_text())
            if previous.get("status") == "failed":
                print(f"SKIP failed {plan_id}")
                continue
        except (OSError, ValueError):
            pass

    print(
        f"[{index}/{len(selected)}] RUN {plan_id}",
        flush=True,
    )

    start = time.perf_counter()
    error = None

    try:
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--mode", "worker",
                "--out", str(output_root),
                "--job-json", json.dumps(job),
            ],
            cwd=str(BASE),
            env=env,
            text=True,
            capture_output=True,
            timeout=a.timeout,
        )

        if result.stdout.strip():
            print(result.stdout.strip(), flush=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Worker exit {result.returncode}: "
                + result.stderr[-1500:]
            )

        if not worker_result.is_file():
            raise RuntimeError(
                "Worker returned success without result JSON"
            )

        measurement = json.loads(worker_result.read_text())

        if measurement.get("status") != "ok":
            raise RuntimeError(
                "Worker result status is not ok"
            )

        status = "ok"
        completed_count += 1

    except Exception as exc:
        status = "failed"
        error = str(exc)
        failed_count += 1
        print("FAILED:", error, flush=True)

    elapsed = time.perf_counter() - start

    atomic_json(status_path, {
        "plan_run_id": plan_id,
        "runner_run_id": rid,
        "status": status,
        "wall_time_sec": elapsed,
        "worker_result": str(worker_result),
        "error": error,
    })

    print(f"Wall time: {elapsed:.2f}s", flush=True)

    # Stop on unexpected failure during pilot.
    if status != "ok":
        break

print("\n=== MAC-33 EXECUTION RESULT ===")
print("Completed or previously completed:", completed_count)
print("New failures:", failed_count)
print("Campaign status directory:", campaign_root)
print("Worker measurements:", output_root / "jobs")
