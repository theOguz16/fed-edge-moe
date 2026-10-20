import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "dataset_v3_t23_mac.sqlite"
SOURCE = ROOT / "t24r.py"
WORKER = ROOT / "t24_1_measure_worker.py"
OUTPUT = ROOT / "t24_1_remeasurements"
PROTOCOL = "T24_1_REMEASURE_NUMERICALLY_FLAGGED_V1"

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def connect():
    assert DB.is_file()
    db = sqlite3.connect(
        str(DB),
        timeout=60,
        isolation_level=None
    )
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    return db

def disk_guard():
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < 4:
        raise RuntimeError(
            f"Disk protection: {free:.2f} GiB available"
        )
    return free

def prepare_worker():
    assert SOURCE.is_file()
    source = SOURCE.read_text(encoding="utf-8")

    old = '''    if not allclose or not top1:
        raise RuntimeError(
            "NUMERICAL_MISMATCH "
            f"allclose={allclose} top1={top1} "
            f"max_error={diff}"
        )
'''

    new = '''    if not allclose or not top1:
        print(
            "NUMERICAL_QUALITY_FLAG "
            f"allclose={allclose} top1={top1} "
            f"max_error={diff}",
            flush=True
        )
        # Continue measuring performance and power.
        # The result retains numerical_allclose=False.
'''

    assert source.count(old) == 1, (
        "Original numerical validation block changed. "
        "Refusing unsafe patch."
    )

    patched = source.replace(old, new, 1)
    compile(patched, str(WORKER), "exec")
    encoded = patched.encode("utf-8")

    if WORKER.exists():
        assert WORKER.read_bytes() == encoded, (
            "Existing measurement worker differs"
        )
    else:
        with WORKER.open("xb") as f:
            f.write(encoded)

    return {
        "source_sha256": sha(SOURCE),
        "worker_sha256": sha(WORKER),
    }

def setup(db, provenance):
    assert db.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    assert db.execute(
        "SELECT COUNT(*) FROM tasks"
    ).fetchone()[0] == 327336

    assert db.execute(
        "SELECT COUNT(*) FROM t23_measurements"
    ).fetchone()[0] == 90

    assert db.execute(
        "SELECT COUNT(*) FROM t24r_measurements"
    ).fetchone()[0] == 203

    original = list(db.execute("""
        SELECT case_id,model,precision,status,
               workload_id,mapping,qos
        FROM t24r_cases
        WHERE status IN (
            'numerical_mismatch',
            'failed_runtime'
        )
        ORDER BY ordinal
    """))

    assert len(original) == 13
    assert sum(
        r["status"] == "numerical_mismatch"
        for r in original
    ) == 12
    assert sum(
        r["status"] == "failed_runtime"
        for r in original
    ) == 1

    for item in original:
        if item["status"] == "numerical_mismatch":
            assert item["model"] == "resnet50"
            assert item["precision"] == "bf16"
            assert item["workload_id"] == "batch8_micro2"

    db.execute("""
        CREATE TABLE IF NOT EXISTS t24_1_attempts (
            attempt_token TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            started_utc TEXT NOT NULL,
            finished_utc TEXT,
            status TEXT NOT NULL,
            exit_code INTEGER,
            log_path TEXT,
            error TEXT,
            worker_sha256 TEXT NOT NULL,
            FOREIGN KEY(case_id)
                REFERENCES t24r_cases(case_id)
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS t24_1_results (
            case_id TEXT PRIMARY KEY,
            attempt_token TEXT NOT NULL UNIQUE,
            original_failure_status TEXT NOT NULL,
            numerical_quality_status TEXT NOT NULL,
            numerical_allclose INTEGER NOT NULL,
            top1_agreement INTEGER NOT NULL,
            max_abs_logit_error REAL NOT NULL,
            throughput_items_s REAL NOT NULL,
            batch_latency_median_ms REAL NOT NULL,
            batch_latency_p95_ms REAL NOT NULL,
            cpu_power_w REAL NOT NULL,
            gpu_power_w REAL NOT NULL,
            gross_energy_j_item REAL NOT NULL,
            power_sample_count INTEGER NOT NULL,
            result_path TEXT NOT NULL,
            result_sha256 TEXT NOT NULL,
            power_raw_sha256 TEXT NOT NULL,
            worker_sha256 TEXT NOT NULL,
            completed_utc TEXT NOT NULL,
            FOREIGN KEY(case_id)
                REFERENCES t24r_cases(case_id),
            FOREIGN KEY(attempt_token)
                REFERENCES t24_1_attempts(attempt_token)
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS t24_1_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    expected = {
        "protocol": PROTOCOL,
        "original_source_sha256":
            provenance["source_sha256"],
        "worker_sha256":
            provenance["worker_sha256"],
        "expected_original_failures": "13",
        "power_required": "true",
        "historical_results_modified": "false",
    }

    for key, value in expected.items():
        existing = db.execute(
            "SELECT value FROM t24_1_metadata WHERE key=?",
            (key,)
        ).fetchone()

        if existing is not None:
            assert existing[0] == value, (
                "Recovery protocol changed: " + key
            )
        else:
            db.execute(
                "INSERT INTO t24_1_metadata VALUES (?,?)",
                (key, value)
            )

    # Recover attempts abandoned by an interrupted parent.
    db.execute("""
        UPDATE t24_1_attempts
        SET status='interrupted',
            finished_utc=?
        WHERE status='running'
    """, (now(),))

    for row in db.execute("""
        SELECT result_path,result_sha256
        FROM t24_1_results
    """):
        assert Path(row[0]).is_file()
        assert sha(row[0]) == row[1]

    print("T24.1 EXISTING DATA VERIFIED")
    print("OLD VALID T24R MEASUREMENTS: 203")
    print("OLD T23 MEASUREMENTS: 90")
    print("REMEASUREMENT TARGETS: 13")

def next_case(db):
    row = db.execute("""
        SELECT c.*
        FROM t24r_cases c
        LEFT JOIN t24_1_results r
          ON c.case_id=r.case_id
        WHERE c.status IN (
            'numerical_mismatch',
            'failed_runtime'
        )
          AND r.case_id IS NULL
        ORDER BY c.ordinal
        LIMIT 1
    """).fetchone()

    return dict(row) if row else None

def create_attempt(db, case, token, folder, worker_sha):
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("""
            INSERT INTO t24_1_attempts (
                attempt_token,case_id,started_utc,
                status,log_path,worker_sha256
            )
            VALUES (?,?,?,'running',?,?)
        """, (
            token,
            case["case_id"],
            now(),
            str(folder),
            worker_sha,
        ))
        db.commit()
    except BaseException:
        db.rollback()
        raise

def finish_attempt(db, token, status, code, error):
    db.execute("BEGIN IMMEDIATE")
    try:
        result = db.execute("""
            UPDATE t24_1_attempts
            SET status=?,
                finished_utc=?,
                exit_code=?,
                error=?
            WHERE attempt_token=?
              AND status='running'
        """, (
            status, now(), code,
            error[-5000:], token
        ))
        assert result.rowcount == 1
        db.commit()
    except BaseException:
        db.rollback()
        raise

def commit_result(db, case, token, result_path, worker_sha):
    result = json.loads(result_path.read_text())
    assert result["case_id"] == case["case_id"]
    assert result["attempt_token"] == token

    power_file = result_path.with_name(
        "powermetrics_raw.txt"
    )

    assert power_file.is_file()
    assert result["power_raw_sha256"] == sha(power_file)

    assert result["power_sample_count"] >= 2
    assert result["power_status"].startswith(
        "measured_gross_estimate"
    )
    assert result["qos_main_thread_verified"] is True

    required = [
        "throughput_items_s",
        "batch_latency_median_ms",
        "batch_latency_p95_ms",
        "cpu_power_w",
        "gpu_power_w",
        "gross_energy_j_item",
        "measurement_seconds",
    ]

    for key in required:
        value = result[key]
        assert value is not None
        assert math.isfinite(value)

    assert result["throughput_items_s"] > 0
    assert result["gross_energy_j_item"] >= 0
    assert result["cpu_power_w"] >= 0
    assert result["gpu_power_w"] >= 0
    assert result["measurement_seconds"] > 0

    expected_energy = (
        result["cpu_power_w"] + result["gpu_power_w"]
    ) / result["throughput_items_s"]

    assert math.isclose(
        expected_energy,
        result["gross_energy_j_item"],
        rel_tol=1e-8,
        abs_tol=1e-12,
    )

    numerical = bool(result["numerical_allclose"])
    top1 = bool(result["top1_agreement"])

    quality = (
        "numerical_pass"
        if numerical and top1
        else "numerical_mismatch_measured"
    )

    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("""
            INSERT INTO t24_1_results VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            case["case_id"],
            token,
            case["status"],
            quality,
            int(numerical),
            int(top1),
            result["max_abs_logit_error"],
            result["throughput_items_s"],
            result["batch_latency_median_ms"],
            result["batch_latency_p95_ms"],
            result["cpu_power_w"],
            result["gpu_power_w"],
            result["gross_energy_j_item"],
            result["power_sample_count"],
            str(result_path.resolve()),
            sha(result_path),
            sha(power_file),
            worker_sha,
            now(),
        ))

        changed = db.execute("""
            UPDATE t24_1_attempts
            SET status='measurement_completed',
                finished_utc=?,
                exit_code=0
            WHERE attempt_token=?
              AND status='running'
        """, (now(), token))

        assert changed.rowcount == 1
        db.commit()
    except BaseException:
        db.rollback()
        raise

    return result, quality

def report(db):
    total = db.execute(
        "SELECT COUNT(*) FROM t24_1_results"
    ).fetchone()[0]

    quality = list(db.execute("""
        SELECT numerical_quality_status,COUNT(*)
        FROM t24_1_results
        GROUP BY numerical_quality_status
    """))

    remaining = 13 - total

    print("\n=== T24.1 RECOVERY STATUS ===")
    print("MEASURED WITH POWER:", total, "/13")
    print("QUALITY CLASSIFICATIONS:", dict(quality))
    print("REMAINING:", remaining)
    print("HISTORICAL T24R VALID:", 203)
    print("HISTORICAL T23:", 90)
    print("SOURCE FAILURE RECORDS: PRESERVED")
    print("DATABASE:", DB)
    print(
        "CAMPAIGN:",
        "FINISHED" if remaining == 0
        else "INCOMPLETE_RESUMABLE"
    )

def run(limit):
    with (ROOT / ".campaign.lock").open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )

        disk_guard()
        provenance = prepare_worker()
        db = connect()

        try:
            setup(db, provenance)

            for index in range(limit):
                disk_guard()
                case = next_case(db)

                if case is None:
                    print("ALL 13 REMEASUREMENTS FINISHED")
                    break

                token = uuid.uuid4().hex
                folder = (
                    OUTPUT / case["case_id"] /
                    ("attempt_" + token)
                )
                folder.mkdir(parents=True)

                casefile = folder / "case.json"
                resultfile = folder / "result.json"
                stdoutfile = folder / "stdout.txt"
                stderrfile = folder / "stderr.txt"

                case["attempt_token"] = token
                write_json(casefile, case)
                create_attempt(
                    db, case, token, folder,
                    provenance["worker_sha256"]
                )

                print(
                    f"\n[{index+1}/{limit}] RUN",
                    case["model"],
                    case["precision"],
                    case["qos"],
                    case["mapping"][:12],
                    case["workload_id"],
                    flush=True
                )

                interrupted = False
                timed_out = False

                with stdoutfile.open("w") as out, \
                     stderrfile.open("w") as err:
                    proc = subprocess.Popen(
                        [
                            sys.executable, "-u",
                            str(WORKER),
                            "worker",
                            str(casefile),
                            str(resultfile),
                        ],
                        stdout=out,
                        stderr=err,
                        start_new_session=False,
                        env={
                            **os.environ,
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                        }
                    )

                    try:
                        code = proc.wait(timeout=600)
                    except KeyboardInterrupt:
                        interrupted = True
                        proc.send_signal(signal.SIGINT)
                        try:
                            code = proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            code = proc.wait()
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        proc.kill()
                        code = proc.wait()

                error = stderrfile.read_text(
                    errors="replace"
                )

                if interrupted:
                    finish_attempt(
                        db, token, "interrupted",
                        code, error
                    )
                    print(
                        "INTERRUPTED — SAFE TO RESUME",
                        flush=True
                    )
                    break

                if timed_out or code != 0:
                    status = (
                        "power_missing"
                        if "POWER_MISSING" in error
                        else (
                            "timeout"
                            if timed_out
                            else "runtime_failed"
                        )
                    )

                    finish_attempt(
                        db, token, status, code, error
                    )

                    print(
                        "FAILED; CAMPAIGN PAUSED:",
                        status,
                        "exit=", code,
                        flush=True
                    )
                    print(error[-1800:], flush=True)
                    break

                if not resultfile.is_file():
                    finish_attempt(
                        db, token, "missing_result",
                        code, "No result JSON"
                    )
                    print(
                        "MISSING RESULT — CAMPAIGN PAUSED"
                    )
                    break

                try:
                    result, quality = commit_result(
                        db, case, token,
                        resultfile,
                        provenance["worker_sha256"]
                    )
                except Exception as exc:
                    finish_attempt(
                        db, token,
                        "validation_failed",
                        code, repr(exc)
                    )
                    print(
                        "RESULT VALIDATION FAILED:",
                        repr(exc),
                        flush=True
                    )
                    break

                print(
                    "COMMITTED",
                    f"quality={quality}",
                    f"throughput={result['throughput_items_s']:.3f}",
                    f"power={result['soc_cpu_gpu_power_w']:.3f} W",
                    f"energy={result['gross_energy_j_item']:.6f} J/item",
                    flush=True
                )

            report(db)
        finally:
            db.close()

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        raise SystemExit(
            "Usage: python t24_1.py run 13"
        )
    run(int(sys.argv[2]))
