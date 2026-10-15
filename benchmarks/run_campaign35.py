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
PLANS = BASE / "results/benchmark_v2_plans/campaign31"
RUNNER = BASE / "benchmarks/vision_partition_runner29.py"
PROTOCOL = "vision-partition-v2-mac29"

p = argparse.ArgumentParser()
p.add_argument("--machine", choices=["mac_m4", "msi_3050"],
               default="mac_m4")
p.add_argument("--tier",
               choices=["pilot", "core", "extended_additions"],
               default="pilot")
p.add_argument("--model", default="resnet50")
p.add_argument("--profile", default="stage6")
p.add_argument("--categories", default="all")
p.add_argument("--max-new-jobs", type=int, default=2)
p.add_argument("--timeout", type=int, default=1200)
p.add_argument("--execute", action="store_true")
p.add_argument("--retry-failed", action="store_true")
p.add_argument("--allow-formal", action="store_true")
a = p.parse_args()

if a.max_new_jobs < 1:
    sys.exit("--max-new-jobs must be positive")

plan_path = PLANS / f"{a.machine}_{a.tier}.csv"
output = BASE / "results/benchmark_v2_vision" / a.machine
statuses35 = PLANS / "execution35" / a.machine / a.tier
statuses33 = PLANS / "execution33" / a.machine / a.tier

with plan_path.open(newline="") as f:
    plan = list(csv.DictReader(f))

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)

def atomic_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def make_job(r):
    # Must remain compatible with MAC-33 and MAC-29 worker IDs.
    return {
        "protocol": PROTOCOL,
        "machine": r["machine"],
        "model": r["model"],
        "profile": r["profile"],
        "manifest_sha256": r["manifest_sha256"],
        "mapping_name":
            f"{r['tier']}_{r['candidate_id']}",
        "mapping": r["mapping"],
        "accelerator": r["accelerator"],
        "threads": int(r["threads"]),
        "size": int(r["size"]),
        "batch": int(r["batch"]),
        "warmup": int(r["warmup"]),
        "steps": int(r["steps"]),
        "repeats": int(r["repeats"]),
        "process_run": int(r["process_run"]),
        "seed": 42,
        "weights": r["weights"],
        "precision": r["precision"],
        "measurement_mode":
            "sequential_cpu_input_cpu_output",
    }

def worker_id(job):
    text = json.dumps(
        job, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode()).hexdigest()[:20]

def worker_path(job):
    return output / "jobs" / f"{worker_id(job)}.json"

def completed(job):
    result = read_json(worker_path(job))
    if not isinstance(result, dict):
        return False

    required = (
        result.get("status") == "ok"
        and result.get("run_id") == worker_id(job)
        and result.get("numerical_pass") is True
        and result.get("mapping") == job["mapping"]
        and result.get("threads") == job["threads"]
        and result.get("manifest_sha256")
            == job["manifest_sha256"]
        and result.get("process_run") == job["process_run"]
        and result.get("sample_count")
            == job["steps"] * job["repeats"]
    )

    if not required:
        return False

    raw_name = result.get("raw_latency_file")
    return bool(raw_name and (output / raw_name).is_file())

def failed_previously(plan_id):
    for directory in (statuses35, statuses33):
        status = read_json(directory / f"{plan_id}.json")
        if status and status.get("status") == "failed":
            return True
    return False

def check_manifest(r):
    path = (
        BASE / "configs/partition_manifests_v2"
        / f"{r['model']}__{r['profile']}.json"
    )
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()

    if digest != r["manifest_sha256"]:
        raise RuntimeError(f"Manifest hash mismatch: {path}")

    manifest = json.loads(raw)
    mapping = r["mapping"]

    if (len(mapping) != manifest["group_count"]
            or set(mapping) - {"C", "M"}):
        raise RuntimeError(f"Invalid mapping: {r['run_id']}")

    if r["machine"] != a.machine or r["tier"] != a.tier:
        raise RuntimeError("Campaign row mismatch")

def safety_check():
    free_gb = shutil.disk_usage(BASE).free / 1e9
    print(f"Disk free: {free_gb:.2f} GB", flush=True)

    if free_gb < 10:
        raise RuntimeError("Less than 10 GB disk free")

    if a.machine == "mac_m4":
        if platform.system() != "Darwin":
            raise RuntimeError("Expected macOS")

        state = subprocess.check_output(
            ["pmset", "-g", "batt"], text=True, timeout=10
        )

        if "AC Power" not in state:
            raise RuntimeError(
                "Mac is running on battery. Connect AC power."
            )

    if a.machine == "msi_3050":
        if platform.system() == "Darwin":
            raise RuntimeError(
                "MSI campaign must run on the MSI machine"
            )

def rebuild_reports():
    results = []

    for path in sorted((output / "jobs").glob("*.json")):
        value = read_json(path)
        if isinstance(value, dict):
            results.append(value)

    if results:
        preferred = [
            "run_id", "status", "machine", "model",
            "profile", "mapping_name", "mapping",
            "accelerator", "threads", "process_run",
            "size", "batch", "latency_mean_ms",
            "latency_p50_ms", "latency_p95_ms",
            "throughput_img_s", "sample_count",
            "numerical_pass", "logical_transfer_count",
            "logical_transfer_bytes", "energy_j",
            "energy_boundary", "raw_latency_file",
        ]

        all_fields = set().union(*(r.keys() for r in results))
        columns = (
            [x for x in preferred if x in all_fields]
            + sorted(all_fields - set(preferred))
        )

        atomic_csv(output / "summary.csv", results, columns)

    status_by_id = {}

    for directory in (statuses33, statuses35):
        for path in sorted(directory.glob("*.json")):
            value = read_json(path)
            if value and "plan_run_id" in value:
                status_by_id[value["plan_run_id"]] = value

    status_rows = list(status_by_id.values())

    if status_rows:
        columns = sorted(set().union(
            *(r.keys() for r in status_rows)
        ))
        atomic_csv(
            PLANS / f"{a.machine}_{a.tier}_status35.csv",
            status_rows,
            columns
        )

categories = {
    x.strip() for x in a.categories.split(",") if x.strip()
}

selected = [
    r for r in plan
    if (a.model == "all" or r["model"] == a.model)
    and (a.profile == "all" or r["profile"] == a.profile)
    and (
        a.categories == "all"
        or r["category"] in categories
    )
]

if len({r["run_id"] for r in selected}) != len(selected):
    raise RuntimeError("Duplicate plan run IDs")

done = []
failed = []
pending = []

for row in selected:
    check_manifest(row)
    job = make_job(row)

    if completed(job):
        done.append(row)
    elif failed_previously(row["run_id"]) and not a.retry_failed:
        failed.append(row)
    else:
        pending.append(row)

next_jobs = pending[:a.max_new_jobs]

print("=== MAC-35 RESUMABLE CAMPAIGN ===")
print("Machine:", a.machine)
print("Tier:", a.tier)
print("Matching plan jobs:", len(selected))
print("Previously completed:", len(done))
print("Previously failed (held):", len(failed))
print("Pending:", len(pending))
print("Next new jobs:", len(next_jobs))
print("Mode:", "EXECUTE" if a.execute else "DRY RUN")

for i, row in enumerate(next_jobs, 1):
    print(
        f"{i:2d}. {row['model']:15s} "
        f"{row['profile']:14s} "
        f"{row['category']:11s} "
        f"T={row['threads']:2s} "
        f"id={row['run_id']}",
        flush=True
    )

if not a.execute:
    print("\nNo inference executed.")
    sys.exit(0)

if a.tier != "pilot" and not a.allow_formal:
    sys.exit("Formal campaign locked until pilot review")

if a.tier == "pilot" and a.max_new_jobs > 12:
    sys.exit("Pilot limit: at most 12 new jobs per invocation")

safety_check()
statuses35.mkdir(parents=True, exist_ok=True)

for i, row in enumerate(next_jobs, 1):
    job = make_job(row)
    plan_id = row["run_id"]
    wid = worker_id(job)

    print(
        f"\n[{i}/{len(next_jobs)}] RUN "
        f"{row['category']} T={row['threads']}",
        flush=True
    )

    start = time.perf_counter()
    state = "failed"
    error = None

    try:
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

        proc = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--mode", "worker",
                "--out", str(output),
                "--job-json", json.dumps(job),
            ],
            cwd=str(BASE),
            env=env,
            capture_output=True,
            text=True,
            timeout=a.timeout,
        )

        if proc.stdout.strip():
            print(proc.stdout.strip(), flush=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Worker exit {proc.returncode}: "
                + proc.stderr[-2000:]
            )

        if not completed(job):
            raise RuntimeError(
                "Worker result or raw samples failed validation"
            )

        state = "ok"

    except Exception as exc:
        error = str(exc)
        print("FAILED:", error, flush=True)

    wall = time.perf_counter() - start

    atomic_json(
        statuses35 / f"{plan_id}.json",
        {
            "plan_run_id": plan_id,
            "runner_run_id": wid,
            "status": state,
            "wall_time_sec": wall,
            "worker_result": str(worker_path(job)),
            "error": error,
        }
    )

    rebuild_reports()
    print(f"Wall time: {wall:.2f} seconds", flush=True)

    if state != "ok":
        print("Stopped after failure.")
        break

rebuild_reports()
print("\nUpdated:", output / "summary.csv")
print("Updated:",
      PLANS / f"{a.machine}_{a.tier}_status35.csv")
