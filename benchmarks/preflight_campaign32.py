import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument(
    "--machine",
    choices=["mac_m4", "msi_3050"],
    default="mac_m4"
)
a = p.parse_args()

base = Path.cwd()
plans = base / "results/benchmark_v2_plans/campaign31"
manifests = base / "configs/partition_manifests_v2"

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def ram_gb():
    try:
        if platform.system() == "Darwin":
            result = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            )
            return round(int(result.strip()) / 1e9, 2)
        if platform.system() == "Linux":
            return round(
                os.sysconf("SC_PAGE_SIZE")
                * os.sysconf("SC_PHYS_PAGES") / 1e9, 2
            )
    except Exception:
        pass
    return None

print("=== MAC-32 CAMPAIGN PREFLIGHT ===")
print("Machine:", a.machine)
print("OS:", platform.platform())
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)

disk = shutil.disk_usage(base)
print("Physical RAM (GB):", ram_gb())
print("Disk free (GB):", round(disk.free / 1e9, 2))

environment = {
    "machine": a.machine,
    "os": platform.platform(),
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "ram_gb": ram_gb(),
    "disk_free_gb": round(disk.free / 1e9, 2),
}

print("\nPYTORCH ENVIRONMENT")
try:
    import torch
    import torchvision

    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)

    environment["torch"] = torch.__version__
    environment["torchvision"] = torchvision.__version__

    if a.machine == "mac_m4":
        available = torch.backends.mps.is_available()
        print("MPS available:", available)
        environment["accelerator_available"] = available
    else:
        available = torch.cuda.is_available()
        print("CUDA available:", available)
        environment["accelerator_available"] = available

        if available:
            props = torch.cuda.get_device_properties(0)
            print("GPU:", props.name)
            print("GPU memory (GB):",
                  round(props.total_memory / 1e9, 2))
            environment["gpu"] = props.name
            environment["gpu_memory_gb"] = round(
                props.total_memory / 1e9, 2
            )

except Exception as exc:
    print("Framework check FAILED:", repr(exc))
    environment["framework_error"] = str(exc)

print("Transformers installed:",
      importlib.util.find_spec("transformers") is not None)

if a.machine == "mac_m4":
    try:
        power = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5
        )
        print("\nMAC POWER STATUS")
        print(power.stdout.strip())
        environment["power_status"] = power.stdout.strip()
    except Exception as exc:
        print("Power state unavailable:", exc)

print("\nCAMPAIGN AUDIT")

tiers = ["pilot", "core", "extended_additions"]
seen_ids = set()
total_jobs = 0
total_timed = 0
errors = []
summary = {}

manifest_cache = {}

for tier in tiers:
    path = plans / f"{a.machine}_{tier}.csv"

    if not path.exists():
        errors.append(f"Missing plan: {path}")
        continue

    rows = read_csv(path)
    total_jobs += len(rows)

    model_counts = Counter()
    timed = 0

    for r in rows:
        model_counts[r["model"]] += 1

        rid = r["run_id"]
        if rid in seen_ids:
            errors.append(f"Duplicate run ID: {rid}")
        seen_ids.add(rid)

        timed += int(r["steps"]) * int(r["repeats"])

        key = (r["model"], r["profile"])

        if key not in manifest_cache:
            mp = manifests / (
                f"{r['model']}__{r['profile']}.json"
            )

            if not mp.exists():
                errors.append(f"Missing manifest: {mp}")
                continue

            content = mp.read_bytes()
            manifest_cache[key] = (
                json.loads(content),
                hashlib.sha256(content).hexdigest()
            )

        manifest, digest = manifest_cache[key]

        if r["manifest_sha256"] != digest:
            errors.append(f"Manifest hash mismatch: {key}")

        mapping = r["mapping"]

        if (len(mapping) != manifest["group_count"]
                or set(mapping) - {"C", "M"}):
            errors.append(f"Invalid mapping: {rid}")
            continue

        switches = sum(
            mapping[i] != mapping[i - 1]
            for i in range(1, len(mapping))
        )
        if switches > 2:
            errors.append(f"Too many switches: {rid}")

    total_timed += timed
    summary[tier] = {
        "processes": len(rows),
        "timed_inferences": timed,
        "model_counts": dict(model_counts),
    }

    print(
        f"{tier:20s} "
        f"jobs={len(rows):6d} "
        f"timed_calls={timed:9d}"
    )

print("\nTOTAL")
print("Independent process jobs:", total_jobs)
print("Timed inference calls:", total_timed)
print("Distinct manifest profiles:", len(manifest_cache))
print("Validation issues:", len(errors))

for error in errors[:12]:
    print("ERROR:", error)

out = plans / f"preflight_{a.machine}.json"
out.write_text(json.dumps({
    "environment": environment,
    "campaign": summary,
    "total_jobs": total_jobs,
    "total_timed_calls": total_timed,
    "validation_errors": errors,
}, indent=2) + "\n")

print("\nSaved:", out)
print("No benchmark was executed.")

if errors:
    sys.exit(1)
