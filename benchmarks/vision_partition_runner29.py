import argparse
import csv
import hashlib
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
VERSION = "vision-partition-v2-mac29"

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["plan", "run", "worker"], default="plan")
p.add_argument("--machine", default="mac_m4")
p.add_argument("--model", default="resnet50")
p.add_argument("--profile", default="stage6")
p.add_argument("--accelerator", choices=["mps", "cuda"], default="mps")
p.add_argument("--patterns", default="C,M,CM,MC,MCM,CMC")
p.add_argument("--mapping", default="")
p.add_argument("--threads", default="1,4")
p.add_argument("--size", type=int, default=224)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--warmup", type=int, default=4)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--repeats", type=int, default=2)
p.add_argument("--process-runs", type=int, default=1)
p.add_argument("--max-new-jobs", type=int, default=0)
p.add_argument("--timeout", type=int, default=1200)
p.add_argument("--retry-failed", action="store_true")
p.add_argument("--out", default="")
p.add_argument("--job-json", default="")
a = p.parse_args()

root = Path(a.out or (
    BASE / "results" / "benchmark_v2_vision" / a.machine
)).resolve()

jobs_dir = root / "jobs"
raw_dir = root / "raw_latency"

def manifest_file(model, profile):
    return (
        BASE / "configs" / "partition_manifests_v2"
        / f"{model}__{profile}.json"
    )

def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)

def job_id(job):
    s = json.dumps(job, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:20]

def percentile(values, q):
    s = sorted(values)
    pos = (len(s) - 1) * q
    low = int(pos)
    high = min(low + 1, len(s) - 1)
    return s[low] + (s[high] - s[low]) * (pos - low)

def make_mapping(pattern, n):
    half = n // 2
    left, right = n // 3, 2 * n // 3

    presets = {
        "C": "C" * n,
        "M": "M" * n,
        "CM": "C" * half + "M" * (n - half),
        "MC": "M" * half + "C" * (n - half),
        "MCM": "M" * left + "C" * (right - left)
               + "M" * (n - right),
        "CMC": "C" * left + "M" * (right - left)
               + "C" * (n - right),
    }

    if pattern not in presets:
        raise ValueError(f"Unknown pattern: {pattern}")

    return presets[pattern]

def validate_mapping(mapping, n):
    if len(mapping) != n or set(mapping) - {"C", "M"}:
        raise ValueError(
            f"Expected {n} characters containing only C or M"
        )

def build_jobs():
    path = manifest_file(a.model, a.profile)
    content = path.read_bytes()
    manifest = json.loads(content)

    if manifest["model_id"] != a.model:
        raise ValueError("Manifest/model mismatch")

    n = manifest["group_count"]
    if n != len(manifest["groups"]):
        raise ValueError("Invalid group count")

    if a.mapping:
        mappings = [("custom", a.mapping)]
    else:
        names = [
            x.strip().upper()
            for x in a.patterns.split(",") if x.strip()
        ]
        mappings = [(name, make_mapping(name, n)) for name in names]

    for _, mapping in mappings:
        validate_mapping(mapping, n)

    threads = [
        int(x.strip())
        for x in a.threads.split(",") if x.strip()
    ]

    if not threads or any(t < 1 for t in threads):
        raise ValueError("Invalid thread set")

    if min(a.batch, a.size, a.steps, a.repeats,
           a.process_runs) < 1 or a.warmup < 0:
        raise ValueError("Invalid workload settings")

    jobs = []
    for name, mapping in mappings:
        for thread in threads:
            for process_run in range(a.process_runs):
                jobs.append({
                    "protocol": VERSION,
                    "machine": a.machine,
                    "model": a.model,
                    "profile": a.profile,
                    "manifest_sha256":
                        hashlib.sha256(content).hexdigest(),
                    "mapping_name": name,
                    "mapping": mapping,
                    "accelerator": a.accelerator,
                    "threads": thread,
                    "size": a.size,
                    "batch": a.batch,
                    "warmup": a.warmup,
                    "steps": a.steps,
                    "repeats": a.repeats,
                    "process_run": process_run,
                    "seed": 42,
                    "weights": "random_none",
                    "precision": "fp32",
                    "measurement_mode":
                        "sequential_cpu_input_cpu_output",
                })
    return jobs

def peak_rss_mb():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1048576 if sys.platform == "darwin"
                        else 1024)
    except (ImportError, OSError):
        return None

def worker(job):
    import torch
    from torchvision import models

    torch.manual_seed(job["seed"])
    torch.set_num_threads(job["threads"])

    path = manifest_file(job["model"], job["profile"])
    content = path.read_bytes()

    if hashlib.sha256(content).hexdigest() != job["manifest_sha256"]:
        raise RuntimeError("Manifest changed after planning")

    manifest = json.loads(content)
    groups = manifest["groups"]
    mapping = job["mapping"]
    validate_mapping(mapping, len(groups))

    backend = job["accelerator"]
    uses_accelerator = "M" in mapping

    if uses_accelerator and backend == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")

    if uses_accelerator and backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")

    device = torch.device(backend)
    model = getattr(models, job["model"])(
        weights=None
    ).eval()

    x_cpu = torch.randn(
        job["batch"], 3, job["size"], job["size"]
    )

    with torch.inference_mode():
        reference = model(x_cpu).clone()

    for group, letter in zip(groups, mapping):
        target = device if letter == "M" else torch.device("cpu")
        for module_path in group["modules"]:
            model.get_submodule(module_path).to(target).eval()

    def sync():
        if not uses_accelerator:
            return
        if backend == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize()

    def forward():
        z = x_cpu
        transfers = 0
        transfer_bytes = 0

        for group, letter in zip(groups, mapping):
            target = device if letter == "M" else torch.device("cpu")

            if z.device.type != target.type:
                transfer_bytes += z.numel() * z.element_size()
                z = z.to(target)
                transfers += 1

            for module_path in group["modules"]:
                module = model.get_submodule(module_path)

                if job["model"] == "resnet50" and module_path == "fc":
                    z = torch.flatten(z, 1)

                z = module(z)

        if z.device.type != "cpu":
            transfer_bytes += z.numel() * z.element_size()
            z = z.to("cpu")
            transfers += 1

        return z, transfers, transfer_bytes

    with torch.inference_mode():
        actual, transfer_count, transfer_bytes = forward()
        sync()

        shape_ok = actual.shape == reference.shape
        max_diff = (
            (actual - reference).abs().max().item()
            if shape_ok else None
        )
        numerical_pass = (
            shape_ok
            and bool(torch.isfinite(actual).all())
            and torch.allclose(
                actual, reference, atol=1e-3, rtol=1e-3
            )
        )

        if not numerical_pass:
            raise RuntimeError(
                f"Numerical mismatch: max_diff={max_diff}"
            )

        for _ in range(job["warmup"]):
            forward()
        sync()

        if uses_accelerator and backend == "cuda":
            torch.cuda.reset_peak_memory_stats()

        samples = []

        for repeat in range(job["repeats"]):
            for step in range(job["steps"]):
                sync()
                start = time.perf_counter()
                forward()
                sync()

                duration_ms = (
                    time.perf_counter() - start
                ) * 1000

                samples.append({
                    "repeat": repeat,
                    "step": step,
                    "latency_ms": duration_ms
                })

    values = [r["latency_ms"] for r in samples]
    mean_ms = st.mean(values)

    accel_memory = None
    memory_scope = "not_available"

    if uses_accelerator and backend == "cuda":
        accel_memory = (
            torch.cuda.max_memory_allocated() / 1048576
        )
        memory_scope = "cuda_peak_torch_allocated"

    elif uses_accelerator and backend == "mps":
        if hasattr(torch.mps, "current_allocated_memory"):
            accel_memory = (
                torch.mps.current_allocated_memory() / 1048576
            )
            memory_scope = "mps_current_not_peak"

    ident = job_id(job)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{ident}.csv"
    raw_tmp = raw_path.with_suffix(".tmp")

    with raw_tmp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["repeat", "step", "latency_ms"]
        )
        writer.writeheader()
        writer.writerows(samples)

    raw_tmp.replace(raw_path)

    result = {
        **job,
        "run_id": ident,
        "status": "ok",
        "group_count": len(groups),
        "internal_device_switches": sum(
            mapping[i] != mapping[i - 1]
            for i in range(1, len(mapping))
        ),
        "logical_transfer_count": transfer_count,
        "logical_transfer_bytes": transfer_bytes,
        "numerical_pass": numerical_pass,
        "max_abs_diff": max_diff,
        "sample_count": len(values),
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": percentile(values, 0.50),
        "latency_p95_ms": percentile(values, 0.95),
        "throughput_img_s":
            1000 * job["batch"] / mean_ms,
        "cpu_process_peak_rss_mb": peak_rss_mb(),
        "accelerator_memory_mb": accel_memory,
        "accelerator_memory_scope": memory_scope,
        "energy_j": None,
        "power_w": None,
        "energy_boundary": "not_measured",
        "quality_measured": False,
        "raw_latency_file": str(raw_path.relative_to(root)),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "python_version": platform.python_version(),
        "system": platform.platform(),
        "timestamp_utc":
            __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
    }

    atomic_json(jobs_dir / f"{ident}.json", result)

    print(
        f"PASS {job['model']} {job['profile']} "
        f"{job['mapping_name']} T={job['threads']} "
        f"mean={mean_ms:.3f}ms "
        f"P95={result['latency_p95_ms']:.3f}ms "
        f"transfers={transfer_count}",
        flush=True
    )

def write_summary():
    fields = [
        "run_id", "status", "machine", "model",
        "profile", "manifest_sha256", "mapping_name",
        "mapping", "accelerator", "threads", "process_run",
        "size", "batch", "weights", "precision",
        "measurement_mode", "group_count",
        "internal_device_switches",
        "logical_transfer_count", "logical_transfer_bytes",
        "numerical_pass", "max_abs_diff",
        "sample_count", "latency_mean_ms",
        "latency_p50_ms", "latency_p95_ms",
        "throughput_img_s", "cpu_process_peak_rss_mb",
        "accelerator_memory_mb", "accelerator_memory_scope",
        "energy_j", "power_w", "energy_boundary",
        "quality_measured", "torch_version",
        "timestamp_utc", "error",
    ]

    rows = []
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            pass

    root.mkdir(parents=True, exist_ok=True)
    dest = root / "summary.csv"
    tmp = dest.with_suffix(".tmp")

    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    tmp.replace(dest)

if a.mode == "worker":
    worker(json.loads(a.job_json))
    sys.exit(0)

jobs = build_jobs()

print("=== MAC-29 VISION PARTITION FRAMEWORK ===")
print("Machine:", a.machine)
print("Model:", a.model, "| Profile:", a.profile)
print("Planned independent processes:", len(jobs))

if a.mode == "plan":
    for job in jobs:
        print(
            f"{job_id(job)} "
            f"{job['mapping_name']:6s} "
            f"T={job['threads']:2d} "
            f"process={job['process_run']} "
            f"mapping={job['mapping']}"
        )
    sys.exit(0)

jobs_dir.mkdir(parents=True, exist_ok=True)
new_jobs = 0

for index, job in enumerate(jobs, 1):
    ident = job_id(job)
    dest = jobs_dir / f"{ident}.json"

    if dest.exists():
        try:
            previous = json.loads(dest.read_text())
        except (OSError, ValueError):
            previous = {}

        if previous.get("status") == "ok" or (
            previous and not a.retry_failed
        ):
            print(f"[{index}/{len(jobs)}] SKIP {ident}", flush=True)
            continue

    if a.max_new_jobs and new_jobs >= a.max_new_jobs:
        break

    new_jobs += 1
    print(
        f"[{index}/{len(jobs)}] RUN "
        f"{job['mapping_name']} T={job['threads']}",
        flush=True
    )

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode", "worker",
                "--out", str(root),
                "--job-json", json.dumps(job),
            ],
            cwd=str(BASE),
            capture_output=True,
            text=True,
            timeout=a.timeout,
            env=env
        )

        if completed.stdout.strip():
            print(completed.stdout.strip(), flush=True)

        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr[-2000:]
                or f"Worker exit code {completed.returncode}"
            )

    except Exception as exc:
        print("FAILED:", ident, exc, flush=True)
        atomic_json(dest, {
            **job,
            "run_id": ident,
            "status": "failed",
            "error": str(exc),
        })

    write_summary()

write_summary()
print("\nSaved:", root / "summary.csv")
