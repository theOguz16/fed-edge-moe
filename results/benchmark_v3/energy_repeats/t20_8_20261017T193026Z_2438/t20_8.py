import csv
import hashlib
import importlib.util
import itertools
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
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()

OLD = BASE / (
    "energy_pilot/t20_4_20261017T190349Z_2438"
)
SOURCE = OLD / "t20_4.py"
LOCAL_WORKER = ROOT / "verified_t20_4_worker.py"

DB = BASE / (
    "dataset_extensions/"
    "t19_3_20261017T183826Z_2438/"
    "dataset_v3_extended.sqlite"
)

NAMES = ("full_cpu", "full_mps", "mps_cpu_cut5")
ORDERS = list(itertools.permutations(NAMES))

WORKER_SECONDS = 32
IDLE_SAMPLES = 4
ACTIVE_SAMPLES = 12

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=2, ensure_ascii=False
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

def verify_environment():
    assert SOURCE.is_file()
    assert DB.is_file()

    checks = json.loads(
        (OLD / "checksums.json").read_text()
    )
    assert sha(SOURCE) == checks["t20_4.py"]

    if LOCAL_WORKER.exists():
        assert sha(LOCAL_WORKER) == sha(SOURCE)
    else:
        tmp = ROOT / "verified_t20_4_worker.py.tmp"
        with SOURCE.open("rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, LOCAL_WORKER)
        assert sha(LOCAL_WORKER) == sha(SOURCE)

    conn = sqlite3.connect(
        f"file:{DB}?mode=ro", uri=True
    )
    try:
        assert conn.execute(
            "PRAGMA quick_check"
        ).fetchone()[0] == "ok"

        assert dict(conn.execute(
            "SELECT state,COUNT(*) FROM tasks "
            "GROUP BY state"
        ).fetchall()) == {
            "completed": 6,
            "planned": 327330,
        }
    finally:
        conn.close()

    assert shutil.disk_usage(ROOT).free > 5 * 1024**3

def initialize_plan():
    path = ROOT / "experiment_plan.json"

    expected = {
        "protocol": "T20_8_balanced_energy_repeats_v1",
        "model": "resnet50",
        "hardware": "apple_m4",
        "cpu_threads": 4,
        "precision": "float32",
        "batch_size": 1,
        "worker_seconds": WORKER_SECONDS,
        "idle_samples": IDLE_SAMPLES,
        "active_samples": ACTIVE_SAMPLES,
        "orders": [list(order) for order in ORDERS],
        "worker_source_sha256": sha(LOCAL_WORKER),
        "new_sqlite_measurements": 0,
    }

    if path.exists():
        assert json.loads(path.read_text()) == expected
    else:
        save(path, expected)

def renew_permission():
    proc = subprocess.run(
        ["sudo", "-n", "-v"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Sudo yetkisi sona erdi. "
            "Tamamlanan koşular korunuyor. "
            "Terminalde sudo -v çalıştırıp "
            "kampanyaya devam et."
        )

def parse_power(raw):
    pattern = re.compile(
        r"^\s*(CPU Power|GPU Power|ANE Power|"
        r"Combined Power \(CPU \+ GPU \+ ANE\))"
        r":\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b"
    )

    keys = {
        "CPU Power": "cpu",
        "GPU Power": "gpu",
        "ANE Power": "ane",
        "Combined Power (CPU + GPU + ANE)": "combined",
    }

    frames = []
    current = {}

    for line in raw.splitlines():
        match = pattern.match(line)
        if not match:
            continue

        label, number, unit = match.groups()
        value = float(number)
        if unit == "mW":
            value /= 1000

        key = keys[label]
        current[key] = value

        if key == "combined":
            if all(
                k in current
                for k in ("cpu", "gpu", "ane", "combined")
            ):
                frames.append({
                    k: current[k]
                    for k in (
                        "cpu", "gpu", "ane", "combined"
                    )
                })
            current = {}

    return frames

def sample_power(pm, n, prefix):
    renew_permission()

    stdout = Path(str(prefix) + ".txt")
    stderr = Path(str(prefix) + ".stderr.txt")

    started = time.monotonic_ns()

    with stdout.open("w") as out, stderr.open("w") as err:
        result = subprocess.run(
            [
                "sudo", "-n", pm,
                "-i", "1000",
                "-n", str(n),
                "-s", "cpu_power,gpu_power",
            ],
            stdout=out,
            stderr=err,
            timeout=n + 15,
        )

    ended = time.monotonic_ns()

    if result.returncode != 0:
        raise RuntimeError(
            stderr.read_text(errors="replace")[:2000]
        )

    frames = parse_power(
        stdout.read_text(errors="replace")
    )

    assert len(frames) >= n - 1, (
        "Insufficient power samples: "
        + str(len(frames))
    )

    return {
        "frames": frames,
        "started_ns": started,
        "ended_ns": ended,
        "stdout": stdout,
        "stderr": stderr,
    }

def wait_signal(proc, expected, timeout=120):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Worker exited before {expected}: "
                f"{proc.returncode}"
            )

        ready, _, _ = select.select(
            [proc.stdout], [], [], 0.5
        )
        if ready:
            line = proc.stdout.readline().strip()
            if line == expected:
                return

    raise TimeoutError(expected)

def worker(name, result_path):
    spec = importlib.util.spec_from_file_location(
        "t20_8_verified_worker",
        str(LOCAL_WORKER)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.DURATION_SECONDS = WORKER_SECONDS
    module.worker(name, result_path)

def avg(frames, key):
    return statistics.mean(
        frame[key] for frame in frames
    )

def run_one(round_number, name, pm):
    folder = ROOT / f"round_{round_number:02d}" / name
    folder.mkdir(parents=True, exist_ok=True)

    completed = folder / "summary.json"

    if completed.exists():
        summary = json.loads(completed.read_text())
        assert summary["mapping"] == name
        assert summary["round"] == round_number

        for relative, expected in summary["artifact_sha256"].items():
            assert sha(ROOT / relative) == expected

        print(
            f"SKIP VERIFIED: round={round_number} "
            f"{name}",
            flush=True
        )
        return summary

    attempt = folder / (
        "attempt_" + uuid.uuid4().hex
    )
    attempt.mkdir()

    print(
        f"\nRUN round={round_number}/6 "
        f"mapping={name}",
        flush=True
    )

    before = sample_power(
        pm, IDLE_SAMPLES,
        attempt / "idle_before"
    )

    result_path = attempt / "worker_result.json"
    error_path = attempt / "worker_stderr.txt"

    with error_path.open("w") as err:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "worker",
                name,
                str(result_path),
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
            wait_signal(proc, "READY")

            proc.stdin.write("GO\n")
            proc.stdin.flush()
            wait_signal(proc, "STARTED", timeout=15)

            active = sample_power(
                pm, ACTIVE_SAMPLES,
                attempt / "active"
            )

            stdout_tail, _ = proc.communicate(timeout=60)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Worker failed: {proc.returncode}"
                )

            assert "FINISHED" in stdout_tail
            assert result_path.is_file()

        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise

    time.sleep(2)

    after = sample_power(
        pm, IDLE_SAMPLES,
        attempt / "idle_after"
    )

    result = json.loads(result_path.read_text())

    assert result["numerical_pass"] is True
    assert result["top1_agreement"] is True
    assert result["mapping"] == name

    assert (
        result["active_start_monotonic_ns"]
        <= active["started_ns"]
    )
    assert (
        result["active_end_monotonic_ns"]
        >= active["ended_ns"]
    )

    # Reject edge windows; per-frame time stamps
    # are not available in this exploratory protocol.
    frames = active["frames"][2:-2]
    assert len(frames) >= 8

    start_ns = active["started_ns"] + 2_000_000_000
    end_ns = active["ended_ns"] - 2_000_000_000
    seconds = (end_ns - start_ns) / 1e9
    assert seconds > 0

    completed_images = sum(
        start_ns <= stamp <= end_ns
        for stamp in result["completion_timestamps_ns"]
    )
    assert completed_images > 0

    throughput = completed_images / seconds

    idle_before = avg(
        before["frames"][1:-1], "combined"
    )
    idle_after = avg(
        after["frames"][1:-1], "combined"
    )

    idle = statistics.mean(
        [
            frame["combined"]
            for frame in (
                before["frames"][1:-1]
                + after["frames"][1:-1]
            )
        ]
    )

    active_power = {
        key: avg(frames, key)
        for key in ("cpu", "gpu", "ane", "combined")
    }

    gross = active_power["combined"] / throughput
    adjusted = (
        active_power["combined"] - idle
    ) / throughput

    assert math.isfinite(gross) and gross > 0
    assert math.isfinite(adjusted)

    artifacts = [
        result_path,
        error_path,
        before["stdout"],
        before["stderr"],
        active["stdout"],
        active["stderr"],
        after["stdout"],
        after["stderr"],
    ]

    summary = {
        "protocol": "T20_8_balanced_energy_repeats_v1",
        "round": round_number,
        "mapping": name,
        "attempt": attempt.name,
        "measured_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "completed_images": completed_images,
        "window_seconds": seconds,
        "throughput_images_per_second": throughput,
        "cpu_active_watts": active_power["cpu"],
        "gpu_active_watts": active_power["gpu"],
        "combined_active_watts": active_power["combined"],
        "idle_before_watts": idle_before,
        "idle_after_watts": idle_after,
        "idle_drift_watts": idle_after - idle_before,
        "mean_idle_watts": idle,
        "gross_joules_per_image": gross,
        "idle_adjusted_joules_per_image": adjusted,
        "numerical_pass": True,
        "power_boundary":
            "powermetrics_estimated_cpu_gpu_ane",
        "power_is_whole_system": False,
        "per_sample_timestamp_alignment": False,
        "statistical_status": "exploratory_repeated",
        "artifact_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in artifacts
        },
    }

    # Commit marker: this is written only after every
    # validation and artifact check has completed.
    save(completed, summary)

    print(
        f"COMMITTED {name}: "
        f"{throughput:.2f} img/s | "
        f"{active_power['combined']:.3f} W | "
        f"gross={gross:.5f} J/img | "
        f"adjusted={adjusted:.5f} J/img | "
        f"idle_drift={idle_after-idle_before:+.4f} W",
        flush=True
    )

    return summary

def audit_manifest():
    path = ROOT / "checksums.json"
    if not path.exists():
        return

    checks = json.loads(path.read_text())
    for relative, expected in checks.items():
        assert sha(ROOT / relative) == expected, (
            "Artifact integrity failure: " + relative
        )

def update_checksums():
    files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p.name != "checksums.json"
        and not p.name.endswith(".tmp")
    )

    save(
        ROOT / "checksums.json",
        {
            str(p.relative_to(ROOT)): sha(p)
            for p in files
        }
    )

def write_reports():
    rows = []

    for r, order in enumerate(ORDERS, 1):
        for name in order:
            path = (
                ROOT / f"round_{r:02d}"
                / name / "summary.json"
            )

            if path.exists():
                data = json.loads(path.read_text())
                rows.append({
                    key: data[key]
                    for key in (
                        "round",
                        "mapping",
                        "throughput_images_per_second",
                        "combined_active_watts",
                        "idle_before_watts",
                        "idle_after_watts",
                        "idle_drift_watts",
                        "gross_joules_per_image",
                        "idle_adjusted_joules_per_image",
                    )
                })

    csv_path = ROOT / "measurements.csv"
    if rows:
        temp = csv_path.with_name(
            csv_path.name + ".tmp"
        )
        with temp.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0])
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, csv_path)

    print(
        "\nCOMPLETED:",
        len(rows), "/18",
        flush=True
    )

    if len(rows) == 18:
        for name in NAMES:
            group = [
                r for r in rows
                if r["mapping"] == name
            ]
            energies = [
                r["gross_joules_per_image"]
                for r in group
            ]
            print(
                f"{name}: gross median="
                f"{statistics.median(energies):.5f} "
                f"range=[{min(energies):.5f},"
                f"{max(energies):.5f}] J/image"
            )

        print("T20.8 BALANCED POWER REPEATS: PASS")

    update_checksums()

def run():
    verify_environment()
    initialize_plan()
    audit_manifest()

    pm = shutil.which("powermetrics")
    assert pm

    print("=== T20.8 BALANCED POWER REPEATS ===")
    print("18 independent worker processes")
    print("6 rounds x 3 configurations")
    print("SQLite: READ-ONLY")
    print("CTRL+C: stop; completed runs will be retained")
    print()

    try:
        for round_number, order in enumerate(ORDERS, 1):
            for name in order:
                if shutil.disk_usage(ROOT).free < (
                    5 * 1024**3
                ):
                    raise RuntimeError(
                        "Disk guard: less than 5 GiB free"
                    )

                run_one(round_number, name, pm)
                write_reports()

    except KeyboardInterrupt:
        print(
            "\nSTOPPED BY USER — completed "
            "results remain saved.",
            flush=True
        )
        write_reports()
        return

    write_reports()
    print("RESULT DIRECTORY:", ROOT)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        assert len(sys.argv) == 4
        worker(
            sys.argv[2],
            Path(sys.argv[3]).resolve()
        )

    elif len(sys.argv) == 2 and sys.argv[1] == "run":
        run()

    else:
        raise SystemExit(
            "Usage: python t20_8.py run"
        )
