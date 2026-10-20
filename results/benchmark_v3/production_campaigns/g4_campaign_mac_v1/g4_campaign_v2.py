import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"
G4 = BASE / "production_campaigns/g4_campaign_mac_v1"
CORE = G4 / "g4_pipeline_core_v2.py"
PERF_WORKER = G4 / "g4_perf_worker_v2.py"
NUM_WORKER = G4 / "g4_numeric.py"
NUM_RUN = G4 / "numerical_v1"
NUM_DB = NUM_RUN / "numerical.sqlite"
RUN = G4 / "g4_long_campaign_v2"
DB = RUN / "performance.sqlite"

PROTOCOL = "G4_LONG_PERFORMANCE_V2"
MIN_WINDOW_SECONDS = 16
REPEATS = (1, 2)
DIRECTIONS = ("cpu_mps", "mps_cpu")

CUTS = {
    "resnet50": tuple(range(1, 6)),
    "convnext_base": tuple(range(1, 41)),
    "convnext_large": tuple(range(1, 41)),
    "distilgpt2": tuple(range(1, 6)),
    "qwen2_5_1_5b": tuple(range(1, 28)),
    "qwen3_1_7b": tuple(range(1, 28)),
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def atomic_json(path, data):
    assert not path.exists(), path
    tmp = path.with_suffix(".tmp")
    assert not tmp.exists(), tmp
    with tmp.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def disk_guard():
    free = shutil.disk_usage(ROOT).free / 1024**3
    assert free >= 5, f"DISK_GUARD: {free:.2f} GiB"

def verify_sources():
    manifest = json.loads(
        (G4 / "source_manifest.json").read_text()
    )
    for item in manifest["sources"]:
        path = G4 / item["snapshot"].split(
            "g4_campaign_mac_v1/", 1
        )[-1]
        assert path.is_file()
        assert sha(path) == item["sha256"]

def build_perf_worker():
    assert CORE.is_file(), CORE
    code = CORE.read_text()

    old_protocol = (
        'PROTOCOL = "G4_PIPELINE_CORE_ACCEPTANCE_V1"'
    )
    new_protocol = f'PROTOCOL = "{PROTOCOL}"'
    assert code.count(old_protocol) == 1
    code = code.replace(old_protocol, new_protocol, 1)

    old = (
        "        rounds = [round_trip(mode) for _ in range(2)]"
    )
    assert code.count(old) == 1

    replacement = '''
        observer_stop = threading.Event()
        observer_errors = []
        peaks = {
            "rss": process.memory_info().rss,
            "mps_current": torch.mps.current_allocated_memory(),
            "mps_driver": torch.mps.driver_allocated_memory(),
        }

        def observe_memory():
            while not observer_stop.is_set():
                try:
                    peaks["rss"] = max(
                        peaks["rss"], process.memory_info().rss
                    )
                except BaseException as exc:
                    observer_errors.append(repr(exc))
                    observer_stop.set()
                observer_stop.wait(0.05)

        observer = threading.Thread(
            target=observe_memory, daemon=True
        )
        rounds = []
        observer.start()
        window_start_ns = time.monotonic_ns()

        try:
            while (
                len(rounds) < 3
                or (
                    time.monotonic_ns() - window_start_ns
                ) < 16000000000
            ):
                rounds.append(round_trip(mode))
                assert len(rounds) < 100000
        finally:
            window_end_ns = time.monotonic_ns()
            observer_stop.set()
            observer.join(timeout=5)

        assert not observer_errors, observer_errors
        assert not observer.is_alive()
        peaks["mps_current"] = max(
            peaks["mps_current"],
            torch.mps.current_allocated_memory()
        )
        peaks["mps_driver"] = max(
            peaks["mps_driver"],
            torch.mps.driver_allocated_memory()
        )
'''

    code = code.replace(old, replacement.rstrip("\n"), 1)

    old = "        results[mode] = {\n"
    assert code.count(old) == 1

    replacement = '''        results[mode] = {
            "window_start_monotonic_ns": window_start_ns,
            "window_end_monotonic_ns": window_end_ns,
            "rss_peak_sampled_bytes": peaks["rss"],
            "mps_current_prepost_snapshot_max_bytes":
                peaks["mps_current"],
            "mps_driver_prepost_snapshot_max_bytes":
                peaks["mps_driver"],
            "memory_sampling_interval_s": 0.05,
            "mps_peak_measured": False,
'''

    code = code.replace(old, replacement, 1)
    compile(code, str(PERF_WORKER), "exec")
    encoded = code.encode()

    if PERF_WORKER.exists():
        assert PERF_WORKER.read_bytes() == encoded, (
            "Derived worker changed: refusing overwrite"
        )
    else:
        with PERF_WORKER.open("xb") as f:
            f.write(encoded)

def planned_cases():
    output = []
    for model, cuts in CUTS.items():
        for repeat in REPEATS:
            for cut in cuts:
                for direction in DIRECTIONS:
                    case = {
                        "model": model,
                        "cut": cut,
                        "direction": direction,
                        "precision": "fp32",
                        "microbatch": 2,
                        "qos": "default",
                        "repeat": repeat,
                    }
                    key = hashlib.sha256(
                        json.dumps(
                            [PROTOCOL, case],
                            sort_keys=True
                        ).encode()
                    ).hexdigest()[:24]
                    output.append((key, case))
    assert len(output) == 576
    return output

def connect():
    con = sqlite3.connect(DB, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    return con

def prepare():
    assert ROOT.name == "fed-edge-moe", (
        "Run from the fed-edge-moe project root"
    )
    disk_guard()
    verify_sources()
    build_perf_worker()

    assert NUM_DB.is_file()
    with sqlite3.connect(
        f"file:{NUM_DB}?mode=ro", uri=True
    ) as num:
        assert num.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
        assert num.execute(
            "SELECT COUNT(*) FROM cases"
        ).fetchone()[0] == 576

    RUN.mkdir(exist_ok=True)
    con = connect()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS perf_cases (
                case_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL UNIQUE,
                model TEXT NOT NULL,
                cut INTEGER NOT NULL,
                direction TEXT NOT NULL,
                repeat INTEGER NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                result_path TEXT,
                result_sha256 TEXT,
                last_error TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS sensor_sessions (
                session_id TEXT PRIMARY KEY,
                raw_path TEXT NOT NULL,
                raw_sha256 TEXT,
                sample_pairs INTEGER,
                sensor_exit_code INTEGER
            )
        """)

        sources = {
            "protocol": PROTOCOL,
            "core_sha256": sha(CORE),
            "perf_worker_sha256": sha(PERF_WORKER),
            "numeric_worker_sha256": sha(NUM_WORKER),
            "runner_sha256": sha(Path(__file__).resolve()),
            "source_manifest_sha256":
                sha(G4 / "source_manifest.json"),
            "hardware_id": "mac_m4",
            "precision": "fp32",
            "microbatch": "2",
            "qos": "default",
            "timed_window_min_seconds":
                str(MIN_WINDOW_SECONDS),
            "power_scope": "estimated_cpu_gpu_soc_not_wall",
        }

        for key, value in sources.items():
            previous = con.execute(
                "SELECT value FROM metadata WHERE key=?",
                (key,)
            ).fetchone()
            if previous:
                assert previous[0] == value, (
                    "CAMPAIGN_SOURCE_DRIFT", key
                )
            else:
                con.execute(
                    "INSERT INTO metadata VALUES (?, ?)",
                    (key, value)
                )

        expected = planned_cases()
        existing = {
            r[0] for r in con.execute(
                "SELECT case_id FROM perf_cases"
            )
        }
        expected_ids = {case_id for case_id, _ in expected}
        assert existing <= expected_ids, (
            "Unknown case IDs in database"
        )

        for ordinal, (case_id, case) in enumerate(expected):
            con.execute("""
                INSERT OR IGNORE INTO perf_cases (
                    case_id, ordinal, model, cut,
                    direction, repeat, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """, (
                case_id, ordinal, case["model"],
                case["cut"], case["direction"],
                case["repeat"],
            ))

        assert con.execute(
            "SELECT COUNT(*) FROM perf_cases"
        ).fetchone()[0] == 576

        con.commit()
        assert con.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
    finally:
        con.close()

def status():
    if DB.exists():
        with sqlite3.connect(
            f"file:{DB}?mode=ro", uri=True
        ) as con:
            print("\nPERFORMANCE CASES:")
            for state, count in con.execute("""
                SELECT status, COUNT(*) FROM perf_cases
                GROUP BY status ORDER BY status
            """):
                print(" ", state, count)
    else:
        print("PERFORMANCE: not planned")

    if NUM_DB.exists():
        with sqlite3.connect(
            f"file:{NUM_DB}?mode=ro", uri=True
        ) as con:
            print("NUMERICAL CASES:")
            for state, count in con.execute("""
                SELECT status, COUNT(*) FROM cases
                GROUP BY status ORDER BY status
            """):
                print(" ", state, count)

class PowerSensor:
    def __init__(self):
        self.session_id = uuid.uuid4().hex
        folder = RUN / "power_sessions"
        folder.mkdir(exist_ok=True)

        self.path = folder / (
            "powermetrics_" + self.session_id + ".txt"
        )
        self.file = self.path.open("x")
        self.samples = []
        self.pending = None
        self.errors = []

        self.proc = subprocess.Popen(
            [
                "sudo", "-n", "/usr/bin/powermetrics",
                "--samplers", "cpu_power,gpu_power",
                "-i", "1000",
                "-b", "1",
                "-n", "-1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # Intentionally inherits the authenticated
            # terminal session; no sudo per case.
            start_new_session=False,
        )

        self.thread = threading.Thread(
            target=self.read_loop, daemon=True
        )
        self.thread.start()

    def read_loop(self):
        try:
            for line in self.proc.stdout:
                self.file.write(line)
                self.file.flush()

                cpu = re.search(
                    r"CPU Power:\s*([\d.]+)\s*mW",
                    line, re.I
                )
                gpu = re.search(
                    r"GPU Power:\s*([\d.]+)\s*mW",
                    line, re.I
                )

                if cpu:
                    if (
                        self.pending is not None
                        and self.pending.get("gpu_w") is not None
                    ):
                        self.samples.append(self.pending)

                    self.pending = {
                        "received_monotonic_ns":
                            time.monotonic_ns(),
                        "cpu_w": float(cpu.group(1)) / 1000,
                    }

                elif gpu and self.pending is not None:
                    if "gpu_w" not in self.pending:
                        self.pending["gpu_w"] = (
                            float(gpu.group(1)) / 1000
                        )

            if (
                self.pending is not None
                and self.pending.get("gpu_w") is not None
            ):
                self.samples.append(self.pending)
        except BaseException as exc:
            self.errors.append(repr(exc))

    def wait_ready(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if len(self.samples) >= 2:
                break
            if self.proc.poll() is not None:
                break
            time.sleep(0.15)

        assert not self.errors, self.errors
        assert len(self.samples) >= 2, (
            "POWER_SENSOR_START_FAILED",
            self.path.read_text(errors="replace")[-1500:],
        )

    def aligned_samples(self, start_ns, end_ns):
        # Exclude approximately 1 s around each boundary
        # because powermetrics reports sampled intervals.
        lower = start_ns + 1000000000
        upper = end_ns - 1000000000

        selected = [
            dict(sample)
            for sample in list(self.samples)
            if lower <= sample["received_monotonic_ns"]
            <= upper
        ]

        assert len(selected) >= 5, (
            "INSUFFICIENT_ALIGNED_POWER_SAMPLES",
            len(selected),
        )

        for item in selected:
            assert all(
                math.isfinite(v) and v >= 0
                for v in (item["cpu_w"], item["gpu_w"])
            )
        return selected

    def ensure_alive(self):
        assert not self.errors, self.errors
        assert self.proc.poll() is None, (
            "POWER_SENSOR_EXITED", self.proc.returncode
        )

    def close(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()

        self.thread.join(timeout=10)
        self.file.close()
        assert not self.thread.is_alive()
        assert not self.errors, self.errors

def stop_worker(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

def run_child(args, folder, timeout=1800):
    stdout = folder / "stdout.txt"
    stderr = folder / "stderr.txt"

    env = {
        **os.environ,
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    with stdout.open("x") as out, \
         stderr.open("x") as err:
        proc = subprocess.Popen(
            [sys.executable, "-u", *map(str, args)],
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_worker(proc)
            return 124
        except KeyboardInterrupt:
            stop_worker(proc)
            raise

def record_power(worker_result, sensor):
    modes = worker_result["modes"]

    for mode in ("sequential", "pipeline"):
        item = modes[mode]
        start = item["window_start_monotonic_ns"]
        end = item["window_end_monotonic_ns"]
        assert end > start

        samples = sensor.aligned_samples(start, end)
        window_s = (end - start) / 1e9
        assert window_s >= MIN_WINDOW_SECONDS

        cpu_w = sum(
            x["cpu_w"] for x in samples
        ) / len(samples)
        gpu_w = sum(
            x["gpu_w"] for x in samples
        ) / len(samples)

        # Align throughput denominator with the
        # measurement interval used for power samples.
        throughput = item["total_work_items"] / window_s
        assert throughput > 0

        item["compute_round_throughput_per_second"] = (
            item["throughput_per_second"]
        )
        item["throughput_per_second"] = throughput
        item["power_window_s"] = window_s
        item["cpu_power_w"] = cpu_w
        item["gpu_power_w"] = gpu_w
        item["power_sample_count"] = len(samples)
        item["estimated_soc_energy_per_work_item_j"] = (
            (cpu_w + gpu_w) / throughput
        )
        item["power_measured"] = True
        item["energy_measured"] = True

        evidence = json.dumps(
            samples,
            sort_keys=True,
            separators=(",", ":")
        ).encode()
        item["power_sample_sha256"] = (
            hashlib.sha256(evidence).hexdigest()
        )
        item["power_samples"] = samples

    worker_result["pipeline_speedup"] = (
        modes["pipeline"]["throughput_per_second"]
        / modes["sequential"]["throughput_per_second"]
    )
    worker_result["pipeline_energy_ratio"] = (
        modes["pipeline"][
            "estimated_soc_energy_per_work_item_j"
        ]
        / modes["sequential"][
            "estimated_soc_energy_per_work_item_j"
        ]
    )
    worker_result["power_measured"] = True
    worker_result["energy_measured"] = True
    worker_result["power_scope"] = (
        "estimated_soc_cpu_plus_gpu_not_wall"
    )
    worker_result["power_sample_alignment"] = (
        "host_monotonic_receive_time_with_1s_edge_trim"
    )
    worker_result["power_window_exactly_aligned"] = False
    worker_result["physical_transfer_bytes_measured"] = False
    worker_result["gpu_kernel_overlap_proven"] = False
    worker_result["measurement_includes_output_validation"] = True
    worker_result["power_sensor_session_id"] = sensor.session_id
    worker_result["power_raw_session_path"] = str(
        sensor.path.relative_to(ROOT)
    )
    worker_result["publication_ready"] = False
    return worker_result

def run_performance(con, sensor, smoke=False):
    if smoke:
        rows = con.execute("""
            SELECT * FROM perf_cases
            WHERE model='resnet50'
              AND cut=2
              AND direction='mps_cpu'
              AND repeat=1
              AND status='pending'
        """).fetchall()
    else:
        rows = con.execute("""
            SELECT * FROM perf_cases
            WHERE status='pending'
            ORDER BY ordinal
        """).fetchall()

    print("PERFORMANCE SELECTED:", len(rows), flush=True)
    failures_in_row = 0

    for index, row in enumerate(rows, 1):
        disk_guard()
        sensor.ensure_alive()

        case = {
            "model": row["model"],
            "cut": row["cut"],
            "direction": row["direction"],
            "precision": "fp32",
            "microbatch": 2,
            "qos": "default",
            "repeat": row["repeat"],
        }

        attempt = row["attempts"] + 1
        folder = (
            RUN / "performance_runs"
            / row["case_id"] / f"attempt_{attempt:03d}"
        )
        folder.mkdir(parents=True, exist_ok=False)

        config = folder / "case.json"
        worker_result = folder / "worker_result.json"
        result = folder / "result.json"
        atomic_json(config, case)

        con.execute("""
            UPDATE perf_cases
            SET status='running', attempts=?
            WHERE case_id=? AND status='pending'
        """, (attempt, row["case_id"]))
        con.commit()

        print(
            f"[PERF {index}/{len(rows)}]",
            row["model"], "cut", row["cut"],
            row["direction"], "repeat", row["repeat"],
            flush=True,
        )

        rc = run_child([
            PERF_WORKER, "worker",
            config, worker_result,
        ], folder)

        try:
            assert rc == 0 and worker_result.is_file(), (
                "WORKER_EXIT", rc,
                (folder / "stderr.txt").read_text(
                    errors="replace"
                )[-1200:],
            )

            record = json.loads(worker_result.read_text())
            assert record["protocol"] == PROTOCOL
            assert record["case"]["model"] == row["model"]
            assert record["case"]["cut"] == row["cut"]
            assert record["case"]["direction"] == row["direction"]
            assert record["case"]["repeat"] == row["repeat"]
            assert set(record["modes"]) == {
                "sequential", "pipeline"
            }

            record = record_power(record, sensor)
            atomic_json(result, record)

            con.execute("""
                UPDATE perf_cases
                SET status='completed',
                    result_path=?,
                    result_sha256=?,
                    last_error=NULL
                WHERE case_id=?
            """, (
                str(result.relative_to(ROOT)),
                sha(result),
                row["case_id"],
            ))
            con.commit()

            failures_in_row = 0
            print(
                "COMMITTED speedup=",
                round(record["pipeline_speedup"], 4),
                "energy_ratio=",
                round(record["pipeline_energy_ratio"], 4),
                flush=True,
            )

        except (AssertionError, ValueError, KeyError) as exc:
            con.execute("""
                UPDATE perf_cases
                SET status='execution_failed',
                    last_error=?
                WHERE case_id=?
            """, (repr(exc)[-2000:], row["case_id"]))
            con.commit()

            failures_in_row += 1
            print("CASE FAILED:", repr(exc)[-600:], flush=True)

            if failures_in_row >= 3:
                raise RuntimeError(
                    "Three consecutive performance failures; "
                    "campaign stopped to protect data"
                )

def validate_numeric_result(row, path):
    obj = json.loads(path.read_text())
    assert obj["protocol"] == "G4_NUMERICAL_SCREEN_V1"
    assert obj["model"] == row["model"]
    assert obj["cut_after_stage"] == row["cut_after_stage"]
    assert obj["precision"] == row["precision"]
    assert obj["direction"] == row["direction"]
    assert obj["logical_activation_bytes"] > 0

    verdict = obj["numerical_status"]
    assert verdict in (
        "numerical_pass",
        "numerical_mismatch",
        "numerical_invalid",
    )
    return verdict

def run_numerical():
    con = sqlite3.connect(NUM_DB)
    con.row_factory = sqlite3.Row
    failures_in_row = 0

    try:
        rows = con.execute("""
            SELECT * FROM cases
            WHERE status='pending'
            ORDER BY model, direction, precision,
                     cut_after_stage
        """).fetchall()

        print("NUMERICAL SELECTED:", len(rows), flush=True)

        for index, row in enumerate(rows, 1):
            disk_guard()
            attempt = row["attempts"] + 1
            folder = (
                NUM_RUN / "runs" / row["case_id"]
                / f"attempt_{attempt:03d}"
            )
            folder.mkdir(parents=True, exist_ok=False)

            result = folder / "result.json"

            con.execute("""
                UPDATE cases
                SET attempts=?
                WHERE case_id=? AND status='pending'
            """, (attempt, row["case_id"]))
            con.commit()

            print(
                f"[NUM {index}/{len(rows)}]",
                row["model"], "cut", row["cut_after_stage"],
                row["precision"], row["direction"],
                flush=True,
            )

            rc = run_child([
                NUM_WORKER, "worker",
                row["model"],
                row["cut_after_stage"],
                row["precision"],
                row["direction"],
                result,
            ], folder, timeout=1200)

            try:
                assert rc == 0 and result.is_file(), (
                    "NUMERIC_WORKER_EXIT", rc,
                    (folder / "stderr.txt").read_text(
                        errors="replace"
                    )[-1200:],
                )
                verdict = validate_numeric_result(row, result)
                con.execute("""
                    UPDATE cases
                    SET status=?, result_path=?,
                        result_sha256=?, note=NULL
                    WHERE case_id=?
                """, (
                    verdict,
                    str(result.relative_to(ROOT)),
                    sha(result),
                    row["case_id"],
                ))
                con.commit()

                failures_in_row = 0
                print("COMMITTED:", verdict, flush=True)

            except (AssertionError, ValueError, KeyError) as exc:
                con.execute("""
                    UPDATE cases
                    SET status='execution_failed', note=?
                    WHERE case_id=?
                """, (repr(exc)[-2000:], row["case_id"]))
                con.commit()
                failures_in_row += 1

                print(
                    "NUMERIC EXECUTION FAILURE:",
                    repr(exc)[-600:],
                    flush=True,
                )

                if failures_in_row >= 3:
                    raise RuntimeError(
                        "Three consecutive numeric worker failures"
                    )
    finally:
        con.close()

def run_campaign(smoke=False):
    prepare()
    lock_path = RUN / ".campaign.lock"

    with lock_path.open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )

        con = connect()
        try:
            # A process interruption leaves attempted files
            # intact, and the case becomes resumable.
            con.execute("""
                UPDATE perf_cases
                SET status='pending'
                WHERE status='running'
            """)
            con.commit()

            if smoke:
                pending = con.execute("""
                    SELECT COUNT(*) FROM perf_cases
                    WHERE model='resnet50' AND cut=2
                    AND direction='mps_cpu' AND repeat=1
                    AND status='pending'
                """).fetchone()[0]
            else:
                pending = con.execute("""
                    SELECT COUNT(*) FROM perf_cases
                    WHERE status='pending'
                """).fetchone()[0]

            if pending:
                sudo = subprocess.run(
                    ["sudo", "-n", "true"],
                    capture_output=True, text=True,
                    timeout=10
                )
                assert sudo.returncode == 0, (
                    "Run sudo -v before starting"
                )

                sensor = PowerSensor()
                try:
                    sensor.wait_ready()

                    con.execute("""
                        INSERT INTO sensor_sessions (
                            session_id,raw_path
                        ) VALUES (?,?)
                    """, (
                        sensor.session_id,
                        str(sensor.path.relative_to(ROOT)),
                    ))
                    con.commit()

                    run_performance(con, sensor, smoke=smoke)
                finally:
                    sensor.close()
                    con.execute("""
                        UPDATE sensor_sessions
                        SET raw_sha256=?,
                            sample_pairs=?,
                            sensor_exit_code=?
                        WHERE session_id=?
                    """, (
                        sha(sensor.path),
                        len(sensor.samples),
                        sensor.proc.returncode,
                        sensor.session_id,
                    ))
                    con.commit()

                print(
                    "POWER SESSION FINISHED:",
                    sensor.session_id,
                    "PAIRED SAMPLES:", len(sensor.samples),
                    flush=True,
                )

        finally:
            con.close()

        if not smoke:
            # No sudo required during the numerical sweep.
            run_numerical()

    status()

def retry_failed():
    prepare()
    with (RUN / ".campaign.lock").open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )
        con = connect()
        try:
            count = con.execute("""
                UPDATE perf_cases
                SET status='pending'
                WHERE status='execution_failed'
            """).rowcount
            con.commit()
            print("PERFORMANCE REQUEUED:", count)
        finally:
            con.close()

        with sqlite3.connect(NUM_DB) as num:
            count = num.execute("""
                UPDATE cases
                SET status='pending'
                WHERE status='execution_failed'
                AND result_path IS NULL
            """).rowcount
            print("NUMERICAL REQUEUED:", count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("plan", "smoke", "run",
                 "status", "retry-failed")
    )
    args = parser.parse_args()

    if args.command == "plan":
        prepare()
        print("G4 LONG CAMPAIGN PLAN: PASS")
        status()
    elif args.command == "status":
        status()
    elif args.command == "smoke":
        run_campaign(smoke=True)
        print("G4 LONG POWER INTEGRATION SMOKE: FINISHED")
    elif args.command == "run":
        run_campaign(smoke=False)
        print("G4 LONG CAMPAIGN RUN RETURNED")
    else:
        retry_failed()
        status()

if __name__ == "__main__":
    main()
