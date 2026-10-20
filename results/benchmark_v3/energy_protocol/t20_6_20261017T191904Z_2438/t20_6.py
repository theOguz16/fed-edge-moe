import hashlib
import importlib.util
import json
import math
import os
import re
import select
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("results/benchmark_v3").resolve()

OLD = BASE / (
    "energy_pilot/"
    "t20_4_20261017T190349Z_2438"
)

WORKER_SOURCE = OLD / "t20_4.py"
DB = BASE / (
    "dataset_extensions/"
    "t19_3_20261017T183826Z_2438/"
    "dataset_v3_extended.sqlite"
)

LOAD_SAMPLES = 14
IDLE_SAMPLES = 4
WORKER_SECONDS = 38

CONFIGS = [
    "full_cpu",
    "full_mps",
    "mps_cpu_cut5",
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(part)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(
            value, f, indent=2,
            ensure_ascii=False
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)

def verify_sources():
    assert DB.is_file()
    assert WORKER_SOURCE.is_file()

    checks = json.loads(
        (OLD / "checksums.json").read_text()
    )
    assert sha(WORKER_SOURCE) == checks["t20_4.py"]

    conn = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True
    )
    try:
        assert conn.execute(
            "PRAGMA quick_check"
        ).fetchone()[0] == "ok"

        states = dict(conn.execute(
            "SELECT state,COUNT(*) "
            "FROM tasks GROUP BY state"
        ).fetchall())

        assert states == {
            "completed": 6,
            "planned": 327330,
        }
    finally:
        conn.close()

def worker(name, result_path):
    spec = importlib.util.spec_from_file_location(
        "t20_4_verified_worker",
        str(WORKER_SOURCE)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.DURATION_SECONDS = WORKER_SECONDS
    module.worker(name, result_path)

def parse_power(raw):
    pattern = re.compile(
        r"^\s*(CPU Power|GPU Power|ANE Power|"
        r"Combined Power \(CPU \+ GPU \+ ANE\))"
        r":\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b"
    )

    names = {
        "CPU Power": "cpu",
        "GPU Power": "gpu",
        "ANE Power": "ane",
        "Combined Power (CPU + GPU + ANE)": "combined",
    }

    samples = []
    current = {}

    for line in raw.splitlines():
        match = pattern.match(line)
        if match is None:
            continue

        label, number, unit = match.groups()
        watts = float(number)
        if unit == "mW":
            watts /= 1000

        current[names[label]] = watts

        if names[label] == "combined":
            if all(
                key in current
                for key in ("cpu", "gpu", "ane", "combined")
            ):
                samples.append({
                    key: current[key]
                    for key in (
                        "cpu", "gpu", "ane", "combined"
                    )
                })
            current = {}

    return samples

def sample_power(pm, count, prefix):
    stdout_path = Path(str(prefix) + ".txt")
    stderr_path = Path(str(prefix) + ".stderr.txt")

    start_ns = time.monotonic_ns()

    with stdout_path.open("w") as out, \
         stderr_path.open("w") as err:
        proc = subprocess.run(
            [
                "sudo", "-n", pm,
                "-i", "1000",
                "-n", str(count),
                "-s", "cpu_power,gpu_power"
            ],
            stdout=out,
            stderr=err,
            timeout=count + 15,
            check=False
        )

    end_ns = time.monotonic_ns()

    if proc.returncode != 0:
        raise RuntimeError(
            "powermetrics failed: "
            + stderr_path.read_text(
                errors="replace"
            )[:2000]
        )

    samples = parse_power(
        stdout_path.read_text(errors="replace")
    )

    assert len(samples) >= count - 1, (
        "Insufficient power frames: "
        + str(len(samples))
    )

    return {
        "samples": samples,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "raw_path": stdout_path,
        "stderr_path": stderr_path,
        "raw_sha256": sha(stdout_path),
    }

def wait_for(proc, expected, timeout=120):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Worker exited before {expected}: "
                f"exit={proc.returncode}"
            )

        ready, _, _ = select.select(
            [proc.stdout], [], [], 0.5
        )

        if ready:
            line = proc.stdout.readline().strip()
            if line == expected:
                return

    raise TimeoutError(
        f"Worker did not report {expected}"
    )

def run_one(out, name, pm):
    folder = out / name
    folder.mkdir(parents=True, exist_ok=False)

    print(f"\n=== {name} ===", flush=True)
    print("Measuring idle power before workload...", flush=True)

    before = sample_power(
        pm, IDLE_SAMPLES,
        folder / "idle_before"
    )

    worker_path = folder / "worker_result.json"
    worker_error = folder / "worker_stderr.txt"

    with worker_error.open("w") as err:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "worker",
                name,
                str(worker_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            bufsize=1,
            env={
                **os.environ,
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            },
        )

        try:
            wait_for(proc, "READY", timeout=120)

            proc.stdin.write("GO\n")
            proc.stdin.flush()

            wait_for(proc, "STARTED", timeout=15)

            print(
                "Measuring power under sustained load...",
                flush=True
            )

            active = sample_power(
                pm, LOAD_SAMPLES,
                folder / "active_load"
            )

            output, _ = proc.communicate(timeout=65)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Worker failed: {proc.returncode}"
                )

            assert "FINISHED" in output
            assert worker_path.is_file()

        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise

    print("Measuring idle power after workload...", flush=True)

    after = sample_power(
        pm, IDLE_SAMPLES,
        folder / "idle_after"
    )

    data = json.loads(worker_path.read_text())

    assert data["numerical_pass"] is True
    assert data["top1_agreement"] is True
    assert data["mapping"] == name
    assert data["total_completed"] > 100

    # The complete active sensor invocation must
    # fit within the inference worker's active interval.
    assert (
        data["active_start_monotonic_ns"]
        <= active["start_ns"]
    )
    assert (
        data["active_end_monotonic_ns"]
        >= active["end_ns"]
    )

    # Discard first/last two sampled periods.
    selected = active["samples"][2:-2]
    assert len(selected) >= 9

    central_start_ns = (
        active["start_ns"] + 2_000_000_000
    )
    central_end_ns = (
        active["end_ns"] - 2_000_000_000
    )

    assert central_end_ns > central_start_ns

    completions = sum(
        central_start_ns <= stamp <= central_end_ns
        for stamp in data["completion_timestamps_ns"]
    )

    seconds = (
        central_end_ns - central_start_ns
    ) / 1e9

    assert completions > 0
    throughput = completions / seconds

    components = ("cpu", "gpu", "ane", "combined")

    active_watts = {
        key: statistics.mean(
            sample[key] for sample in selected
        )
        for key in components
    }

    # Trim the first and last idle sample.
    idle_samples = (
        before["samples"][1:-1]
        + after["samples"][1:-1]
    )
    assert len(idle_samples) >= 2

    idle_watts = {
        key: statistics.mean(
            sample[key] for sample in idle_samples
        )
        for key in components
    }

    gross_energy = (
        active_watts["combined"] / throughput
    )

    net_energy = (
        active_watts["combined"]
        - idle_watts["combined"]
    ) / throughput

    assert math.isfinite(gross_energy)
    assert math.isfinite(net_energy)
    assert gross_energy > 0

    summary = {
        "protocol": "T20_6_load_and_idle_power_pilot",
        "mapping": name,
        "model": "resnet50",
        "cpu_threads": 4,
        "batch_size": 1,
        "precision": "fp32",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "independent_processes_for_this_mapping": 1,
        "power_scope":
            "powermetrics_estimated_cpu_gpu_ane",
        "power_is_whole_system": False,
        "measurement_alignment":
            "entire_sampler_invocation_inside_active_load",
        "per_sample_timestamp_alignment": False,
        "active_power_samples_used": len(selected),
        "idle_power_samples_used": len(idle_samples),
        "average_active_power_watts": active_watts,
        "local_idle_power_watts": idle_watts,
        "throughput_images_per_second": throughput,
        "completed_in_central_window": completions,
        "central_window_seconds": seconds,
        "gross_joules_per_image": gross_energy,
        "idle_adjusted_joules_per_image": net_energy,
        "idle_adjustment_is_estimate": True,
        "worker_sha256": sha(worker_path),
        "power_raw_sha256": {
            "idle_before": before["raw_sha256"],
            "active_load": active["raw_sha256"],
            "idle_after": after["raw_sha256"],
        },
        "source_worker_sha256": sha(WORKER_SOURCE),
        "sqlite_modified": False,
    }

    atomic_json(folder / "summary.json", summary)

    print(
        f"{name}: "
        f"{throughput:.2f} images/s | "
        f"active={active_watts['combined']:.3f} W | "
        f"idle={idle_watts['combined']:.3f} W | "
        f"gross={gross_energy:.4f} J/image | "
        f"idle-adjusted={net_energy:.4f} J/image",
        flush=True
    )

    return summary

def main(out):
    verify_sources()

    assert shutil.disk_usage(out).free >= (
        5 * 1024**3
    ), "Insufficient disk space"

    pm = shutil.which("powermetrics")
    assert pm is not None

    print("=== T20.6 POWER PROTOCOL PILOT ===")
    print("DATABASE: READ-ONLY")
    print("CONFIGURATIONS: 3")
    print("INDEPENDENT WORKER PROCESSES: 3")
    print("ACTIVE WORKER DURATION: 38 seconds")
    print("LOCAL IDLE BASELINES: BEFORE AND AFTER")
    print()

    summaries = []

    for name in CONFIGS:
        summaries.append(
            run_one(out, name, pm)
        )

    atomic_json(out / "pilot_summary.json", {
        "protocol": "T20_6_load_and_idle_power_pilot",
        "results": summaries,
        "statistical_status": "pilot_only",
        "power_sample_timestamps_available": False,
        "environmental_confounding_excluded": False,
        "new_sqlite_tasks": 0,
        "source_database_modified": False,
    })

    files = sorted(
        path for path in out.rglob("*")
        if path.is_file()
        and path.name != "checksums.json"
    )

    checks = {
        str(path.relative_to(out)): sha(path)
        for path in files
    }

    atomic_json(out / "checksums.json", checks)

    for relative, expected in checks.items():
        assert sha(out / relative) == expected

    print("\n=== T20.6 FINAL VALIDATION ===")
    print("REAL WORKER PROCESSES: 3/3")
    print("NUMERICAL VALIDATION: PASS")
    print("IDLE + ACTIVE POWER CAPTURE: PASS")
    print("ACTIVE LOAD INTERVAL COVERAGE: PASS")
    print("SOURCE WORKER CHECKSUM: PASS")
    print("ARTIFACT CHECKSUMS: PASS")
    print("EXISTING SQLITE MODIFIED: NO")
    print("T20.6 POWER PROTOCOL PILOT: PASS")
    print("RESULT DIRECTORY:", out)

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "worker":
        worker(sys.argv[2], Path(sys.argv[3]).resolve())
    else:
        assert len(sys.argv) == 2
        main(Path(sys.argv[1]).resolve())
