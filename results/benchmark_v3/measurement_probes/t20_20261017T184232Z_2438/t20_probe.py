import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()

DB = Path(
    "results/benchmark_v3/dataset_extensions/"
    "t19_3_20261017T183826Z_2438/"
    "dataset_v3_extended.sqlite"
).resolve()

def run_command(args, timeout=8):
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return {
            "returncode": p.returncode,
            "output": (p.stdout + "\n" + p.stderr)[:16000]
        }
    except Exception as exc:
        return {
            "error": type(exc).__name__ + ": " + str(exc)
        }

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()

assert DB.is_file(), f"Extended database missing: {DB}"

conn = sqlite3.connect(
    f"file:{DB}?mode=ro",
    uri=True
)

try:
    integrity = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    assert integrity == "ok"

    states = dict(conn.execute(
        "SELECT state, COUNT(*) FROM tasks GROUP BY state"
    ).fetchall())

    assert states == {
        "completed": 6,
        "planned": 327330,
    }

    registry_counts = {}
    for table in (
        "research_models",
        "experiment_dimensions",
        "measurement_registry",
        "figure_registry"
    ):
        registry_counts[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

    assert registry_counts == {
        "research_models": 6,
        "experiment_dimensions": 12,
        "measurement_registry": 28,
        "figure_registry": 8,
    }

finally:
    conn.close()

packages = {}
for name in ("torch", "torchvision", "psutil", "numpy"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

mps = {
    "torch_imported": False,
    "is_available": None,
    "is_built": None,
}

try:
    import torch

    mps = {
        "torch_imported": True,
        "is_available": torch.backends.mps.is_available(),
        "is_built": torch.backends.mps.is_built(),
    }
except Exception as exc:
    mps["error"] = repr(exc)

tools = {
    name: shutil.which(name)
    for name in (
        "powermetrics",
        "caffeinate",
        "taskpolicy",
        "sysctl",
        "sudo"
    )
}

power_help = {
    "available": bool(tools["powermetrics"]),
    "sampler_options_mentioned": [],
    "actual_power_measurement_validated": False,
    "administrator_permission_tested": False,
}

if tools["powermetrics"]:
    result = run_command(
        [tools["powermetrics"], "--help"]
    )

    power_help["help_returncode"] = result.get("returncode")
    power_help["help_error"] = result.get("error")

    help_text = result.get("output", "")
    keywords = (
        "cpu_power",
        "gpu_power",
        "power",
        "sampler",
        "interval",
    )

    relevant = [
        line.strip()
        for line in help_text.splitlines()
        if any(
            word in line.lower()
            for word in keywords
        )
    ]

    power_help["sampler_options_mentioned"] = relevant[:18]

system = {
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": sys.version.split()[0],
    "cpu_logical": os.cpu_count(),
    "physical_cpu_probe": run_command(
        ["sysctl", "-n", "hw.physicalcpu"]
    ) if tools["sysctl"] else None,
}

disk = shutil.disk_usage(OUT)

report = {
    "protocol": "T20_capability_probe_v1",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "source_database": str(DB),
    "source_database_sha256_at_probe": sha(DB),
    "source_states": states,
    "registry_counts": registry_counts,
    "system": system,
    "python_packages": packages,
    "mps": mps,
    "available_tools": tools,
    "powermetrics": power_help,
    "free_disk_gib": round(
        disk.free / (1024 ** 3), 2
    ),
    "new_inference_measurements": 0,
    "new_power_measurements": 0,
    "sudo_invoked": False,
    "existing_database_modified": False,
}

path = OUT / "capability_report.json"
temp = OUT / "capability_report.json.tmp"

with temp.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())

os.replace(temp, path)

assert json.loads(path.read_text()) == report

print("=== T20 CAPABILITY PROBE ===")
print("EXTENDED SQLITE INTEGRITY: PASS")
print("SOURCE TASK STATES:", states)
print("REGISTRY:", registry_counts)
print("SYSTEM:", system["platform"])
print("TORCH:", packages["torch"])
print("TORCHVISION:", packages["torchvision"])
print("PSUTIL:", packages["psutil"])
print("MPS AVAILABLE:", mps["is_available"])
print("POWERMETRICS FOUND:", power_help["available"])
print(
    "POWERMETRICS HELP RETURN CODE:",
    power_help.get("help_returncode")
)
print("POWER SAMPLER HELP LINES:")
for line in power_help["sampler_options_mentioned"]:
    print("  ", line)
print(
    "POWER MEASUREMENT VALIDATED: NO "
    "(capability check only)"
)
print("FREE DISK GIB:", report["free_disk_gib"])
print("NEW BENCHMARK RUNS: 0")
print("EXISTING DATABASE MODIFIED: NO")
print()
print("T20 CAPABILITY PROBE: PASS")
print("REPORT:", path)
