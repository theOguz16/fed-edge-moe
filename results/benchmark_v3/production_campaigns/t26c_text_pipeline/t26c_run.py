import fcntl
import hashlib
import json
import math
import os
import re
import signal
import sqlite3
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT.parent / (
    "t23_mac_v1/dataset_v3_t23_mac.sqlite"
)
WORKER = ROOT / "t26c_distil_worker.py"
LOCK = DB.parent / ".campaign.lock"
PROTOCOL = "T26C_DISTILGPT2_CPU_MPS_FP32_PRODUCTION_V1"
REPEATS = (1, 2)

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def connect():
    db = sqlite3.connect(
        str(DB), timeout=60, isolation_level=None
    )
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db

def identity(repeat):
    payload = {
        "protocol": PROTOCOL,
        "model": "distilgpt2",
        "precision": "fp32",
        "direction": "cpu_mps",
        "cut": 3,
        "context": 128,
        "batch": 4,
        "micro": 1,
        "qos": "default",
        "repeat": repeat,
    }
    return "t26c_" + hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()

def initialize(db):
    assert WORKER.is_file()
    assert db.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    db.execute("""
        CREATE TABLE IF NOT EXISTS t26c_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS t26c_cases (
            case_id TEXT PRIMARY KEY,
            repeat INTEGER NOT NULL UNIQUE,
            status TEXT NOT NULL,
            current_attempt TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS t26c_attempts (
            attempt_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            started_utc TEXT NOT NULL,
            finished_utc TEXT,
            status TEXT NOT NULL,
            error TEXT,
            folder TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS t26c_results (
            case_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            numerical_status TEXT NOT NULL,
            throughput_tokens_s REAL NOT NULL,
            median_wall_ms REAL NOT NULL,
            cpu_power_w REAL NOT NULL,
            gpu_power_w REAL NOT NULL,
            energy_j_token REAL NOT NULL,
            power_samples INTEGER NOT NULL,
            window_seconds REAL NOT NULL,
            result_path TEXT NOT NULL,
            result_sha256 TEXT NOT NULL,
            power_sha256 TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            PRIMARY KEY(case_id, mode)
        )
    """)

    metadata = {
        "protocol": PROTOCOL,
        "worker_sha256": sha(WORKER),
        "energy_scope": "estimated_cpu_gpu_soc_not_wall",
        "throughput_scope": "wikitext2_prefill_tokens",
        "kernel_overlap_proven": "false",
    }
    for key, value in metadata.items():
        current = db.execute(
            "SELECT value FROM t26c_metadata WHERE key=?",
            (key,)
        ).fetchone()
        if current is None:
            db.execute(
                "INSERT INTO t26c_metadata VALUES (?,?)",
                (key, value)
            )
        else:
            assert current[0] == value, (
                "T26C source/protocol changed", key
            )

    for repeat in REPEATS:
        db.execute("""
            INSERT OR IGNORE INTO t26c_cases
            VALUES (?,?,'planned',NULL)
        """, (identity(repeat), repeat))

    assert db.execute(
        "SELECT COUNT(*) FROM t26c_cases"
    ).fetchone()[0] == 2

    # Parent holds the exclusive campaign lock.
    # An interrupted parent cannot leave a permanently
    # unclaimable case; previous attempts remain intact.
    db.execute("""
        UPDATE t26c_attempts
        SET status='interrupted', finished_utc=?
        WHERE status='running'
    """, (utc(),))
    db.execute("""
        UPDATE t26c_cases
        SET status='planned', current_attempt=NULL
        WHERE status='running'
    """)

def power_pairs(raw):
    cpu = list(re.finditer(
        r"CPU Power:\s*([\d.]+)\s*mW", raw, re.I
    ))
    samples = []
    for i, match in enumerate(cpu):
        stop = (
            cpu[i+1].start()
            if i+1 < len(cpu) else len(raw)
        )
        gpu = re.search(
            r"GPU Power:\s*([\d.]+)\s*mW",
            raw[match.end():stop], re.I
        )
        if gpu:
            samples.append((
                float(match.group(1)) / 1000,
                float(gpu.group(1)) / 1000,
            ))
    return samples

def validate(folder, case_id):
    path = folder / "result.json"
    assert path.is_file(), path
    obj = json.loads(path.read_text())
    assert obj["protocol"] == (
        "T26C_DISTILGPT2_PIPELINE_PILOT_V1"
    )
    assert obj["model"] == "distilgpt2"
    assert obj["precision"] == "fp32"
    assert obj["mapping"] == "cpu_to_mps"
    assert obj["cut_after_block"] == 3
    assert obj["context_tokens"] == 128
    assert obj["batch_size"] == 4
    assert obj["microbatch_size"] == 1
    assert obj["main_qos_class"] == 0x15
    assert obj["pipeline_thread_qos_verified"]
    assert not obj["gpu_kernel_overlap_proven"]

    evidence = []
    for mode in ("sequential", "pipeline"):
        m = obj["measurements"][mode]
        assert m["numerical_status"] == "numerical_pass"
        assert m["rounds"] == 10
        assert m["power_window_s"] >= 10
        assert m["power_sample_count"] >= 2

        power_path = folder / (
            f"powermetrics_{mode}_power_pilot.txt"
        )
        assert sha(power_path) == m["power_file_sha256"]

        values = power_pairs(
            power_path.read_text(errors="replace")
        )
        assert len(values) == m["power_sample_count"]
        cpu = statistics.mean(x[0] for x in values)
        gpu = statistics.mean(x[1] for x in values)

        assert math.isclose(
            cpu, m["cpu_power_w"], rel_tol=1e-9
        )
        assert math.isclose(
            gpu, m["gpu_power_w"], rel_tol=1e-9
        )
        assert math.isclose(
            (cpu + gpu) / m["sustained_prefill_tokens_s"],
            m["gross_energy_j_token"],
            rel_tol=1e-9
        )

        evidence.append((
            case_id,
            mode,
            m["numerical_status"],
            m["sustained_prefill_tokens_s"],
            m["median_wall_ms"],
            cpu,
            gpu,
            m["gross_energy_j_token"],
            m["power_sample_count"],
            m["power_window_s"],
            str(path.resolve()),
            sha(path),
            sha(power_path),
        ))
    return evidence

def stop_group(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

def run(limit):
    with LOCK.open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )
        db = connect()
        try:
            initialize(db)
            print("T26C PROTOCOL VERIFIED", flush=True)

            for _ in range(limit):
                case = db.execute("""
                    SELECT * FROM t26c_cases
                    WHERE status='planned'
                    ORDER BY repeat LIMIT 1
                """).fetchone()

                if case is None:
                    break

                token = uuid.uuid4().hex
                folder = ROOT / "runs" / case["case_id"] / token
                folder.mkdir(parents=True, exist_ok=False)

                db.execute("BEGIN IMMEDIATE")
                try:
                    db.execute("""
                        INSERT INTO t26c_attempts
                        (attempt_id,case_id,started_utc,status,folder)
                        VALUES (?,?,?,'running',?)
                    """, (
                        token, case["case_id"], utc(), str(folder)
                    ))
                    changed = db.execute("""
                        UPDATE t26c_cases
                        SET status='running',current_attempt=?
                        WHERE case_id=? AND status='planned'
                    """, (token, case["case_id"]))
                    assert changed.rowcount == 1
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise

                print(
                    f"T26C REPEAT {case['repeat']}/2",
                    flush=True
                )

                interrupted = False
                timed_out = False
                with (
                    (folder / "stdout.txt").open("w") as out,
                    (folder / "stderr.txt").open("w") as err,
                ):
                    proc = subprocess.Popen(
                        [
                            sys.executable, "-u", str(WORKER)
                        ],
                        stdout=out,
                        stderr=err,
                        start_new_session=False,
                        process_group=0,
                        env={
                            **os.environ,
                            "T26C_RUN_DIR": str(folder),
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                            "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                        },
                    )
                    try:
                        code = proc.wait(timeout=600)
                    except KeyboardInterrupt:
                        interrupted = True
                        stop_group(proc)
                        code = proc.returncode
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        stop_group(proc)
                        code = proc.returncode

                error = (folder / "stderr.txt").read_text(
                    errors="replace"
                )

                evidence = None
                if code == 0 and not timed_out and not interrupted:
                    try:
                        evidence = validate(
                            folder, case["case_id"]
                        )
                    except Exception as exc:
                        error += "\nVALIDATION: " + repr(exc)

                if evidence is None:
                    status = (
                        "interrupted" if interrupted
                        else "failed"
                    )
                    db.execute("BEGIN IMMEDIATE")
                    try:
                        db.execute("""
                            UPDATE t26c_attempts
                            SET status=?,finished_utc=?,error=?
                            WHERE attempt_id=?
                        """, (status, utc(), error[-4000:], token))
                        db.execute("""
                            UPDATE t26c_cases
                            SET status=?,current_attempt=NULL
                            WHERE case_id=?
                        """, (
                            "planned" if interrupted else "failed",
                            case["case_id"]
                        ))
                        db.commit()
                    except BaseException:
                        db.rollback()
                        raise

                    print(
                        "T26C PAUSED:", status,
                        "exit=", code,
                        error[-1500:],
                        flush=True
                    )
                    break

                db.execute("BEGIN IMMEDIATE")
                try:
                    for values in evidence:
                        db.execute("""
                            INSERT INTO t26c_results
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, values + (token,))
                    db.execute("""
                        UPDATE t26c_cases
                        SET status='completed',current_attempt=NULL
                        WHERE case_id=?
                    """, (case["case_id"],))
                    db.execute("""
                        UPDATE t26c_attempts
                        SET status='completed',finished_utc=?
                        WHERE attempt_id=?
                    """, (utc(), token))
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise

                seq, pipe = evidence
                speedup = pipe[3] / seq[3]
                energy_ratio = pipe[7] / seq[7]
                print(
                    "COMMITTED",
                    f"repeat={case['repeat']}",
                    f"speedup={speedup:.3f}x",
                    f"energy_ratio={energy_ratio:.3f}x",
                    flush=True
                )

            globals()["status"](db)
        finally:
            db.close()

def status(db):
    print("\n=== T26C DISTILGPT2 PRODUCTION ===")
    for row in db.execute("""
        SELECT status,COUNT(*)
        FROM t26c_cases GROUP BY status
    """):
        print(row[0], row[1])
    print(
        "MODE MEASUREMENTS:",
        db.execute(
            "SELECT COUNT(*) FROM t26c_results"
        ).fetchone()[0], "/4"
    )
    print("DATABASE:", DB)

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "status":
        with connect() as db:
            status(db)
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        n = int(sys.argv[2])
        assert 1 <= n <= 2
        run(n)
    else:
        raise SystemExit("Usage: t26c_run.py run 1 | run 2 | status")

if __name__ == "__main__":
    main()
