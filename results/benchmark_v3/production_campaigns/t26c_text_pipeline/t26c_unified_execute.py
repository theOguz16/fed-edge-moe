import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import uuid
from pathlib import Path

import t26c_unified_run as plan

ROOT = plan.ROOT
WORKER = ROOT / "t26c_unified_worker.py"
RUNS = ROOT / "runs_v2"
TIMEOUT_SECONDS = 2400

CASE_FIELDS = (
    "case_id", "model", "precision", "direction",
    "cut", "batch_size", "context_tokens",
    "microbatch_size", "qos", "repeat", "cpu_threads",
)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def save(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def power_pairs(raw):
    import re
    cpu = list(re.finditer(
        r"CPU Power:\s*([\d.]+)\s*mW", raw, re.I
    ))
    pairs = []
    for i, match in enumerate(cpu):
        stop = (
            cpu[i + 1].start()
            if i + 1 < len(cpu) else len(raw)
        )
        gpu = re.search(
            r"GPU Power:\s*([\d.]+)\s*mW",
            raw[match.end():stop], re.I
        )
        if gpu:
            pairs.append((
                float(match.group(1)) / 1000,
                float(gpu.group(1)) / 1000,
            ))
    return pairs

def ensure_prepared(db):
    assert WORKER.is_file()
    assert db.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"
    metadata = dict(db.execute(
        "SELECT key,value FROM t26c_v2_metadata"
    ))
    assert metadata["protocol"] == plan.PROTOCOL
    assert db.execute(
        "SELECT COUNT(*) FROM t26c_v2_cases"
    ).fetchone()[0] == 288
    assert shutil.disk_usage(ROOT).free >= 4 * 1024**3

    old_sha = metadata.get("worker_sha256")
    if old_sha is not None:
        assert old_sha == sha(WORKER), (
            "WORKER_SOURCE_CHANGED_AFTER_FIRST_COMMIT"
        )

    plan.verify_legacy(db)

def find_case(db, args):
    query = (
        "SELECT * FROM t26c_v2_cases "
        "WHERE status='planned'"
    )
    values = []

    filters = {
        "model": args.model,
        "direction": args.direction,
        "cut": args.cut,
        "microbatch_size": args.micro,
        "qos": args.qos,
        "repeat": args.repeat,
    }

    for key, value in filters.items():
        if value is not None:
            query += f" AND {key}=?"
            values.append(value)

    query += " ORDER BY ordinal LIMIT 1"
    return db.execute(query, values).fetchone()

def stop_group(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

def validate(folder, case, worker_sha):
    result_path = folder / "result.json"
    assert result_path.is_file(), "RESULT_JSON_MISSING"
    obj = json.loads(result_path.read_text())

    expected_case = {
        "protocol": plan.PROTOCOL,
        **{key: case[key] for key in CASE_FIELDS},
    }

    assert obj["protocol"] == plan.PROTOCOL
    assert obj["case"] == expected_case
    assert obj["worker_sha256"] == worker_sha
    assert obj["model"] == case["model"]
    assert obj["precision"] == "fp32"
    assert obj["input_tokens_sha256"]
    assert obj["corpus_sha256"]
    assert obj["model_config_sha256"]
    assert obj["main_qos_class"] == {
        "background": 0x09,
        "utility": 0x11,
        "default": 0x15,
        "user_initiated": 0x19,
    }[case["qos"]]
    assert obj["pipeline_thread_qos_verified"] is True
    assert obj["gpu_kernel_overlap_proven"] is False
    assert obj["power_scope"] == "estimated_cpu_gpu_soc_not_wall"

    expected_order = (
        ["sequential", "pipeline"]
        if case["repeat"] == 1
        else ["pipeline", "sequential"]
    )
    assert obj["mode_order"] == expected_order
    assert set(obj["measurements"]) == {
        "sequential", "pipeline"
    }

    evidence = []
    for mode in ("sequential", "pipeline"):
        m = obj["measurements"][mode]
        assert m["numerical_status"] == "numerical_pass"
        assert m["token_agreement"] == 1.0
        assert m["power_window_s"] >= 10.0

        rounds = m["timed_round_count"]
        timings = m["timing_samples_ms"]
        assert rounds >= 3
        assert len(timings) == rounds
        assert all(
            math.isfinite(t) and t > 0 for t in timings
        )
        assert math.isclose(
            statistics.median(timings),
            m["median_wall_ms"],
            rel_tol=1e-10,
        )

        power_path = folder / (
            f"powermetrics_{mode}_power_v2.txt"
        )
        assert power_path.is_file()
        assert sha(power_path) == m["power_file_sha256"]

        pairs = power_pairs(
            power_path.read_text(errors="replace")
        )
        assert len(pairs) == m["power_sample_count"]
        assert len(pairs) >= 3

        cpu = statistics.mean(x[0] for x in pairs)
        gpu = statistics.mean(x[1] for x in pairs)
        throughput = (
            rounds * 4 * 128 / m["power_window_s"]
        )
        energy = (cpu + gpu) / throughput

        for a, b in (
            (cpu, m["cpu_power_w"]),
            (gpu, m["gpu_power_w"]),
            (throughput, m["sustained_prefill_tokens_s"]),
            (energy, m["gross_energy_j_token"]),
        ):
            assert math.isfinite(a) and a >= 0
            assert math.isclose(a, b, rel_tol=1e-9)

        evidence.append((
            case["case_id"],
            mode,
            m["numerical_status"],
            throughput,
            m["median_wall_ms"],
            cpu,
            gpu,
            energy,
            len(pairs),
            m["power_window_s"],
            str(result_path.resolve()),
            sha(result_path),
            str(power_path.resolve()),
            sha(power_path),
        ))

    return evidence

def record_failure(db, case_id, token, state, message):
    with db:
        db.execute("""
            UPDATE t26c_v2_attempts
            SET status=?,finished_utc=?,error=?
            WHERE attempt_id=?
        """, (state, plan.utc(), message[-6000:], token))
        db.execute("""
            UPDATE t26c_v2_cases
            SET status=?,current_attempt=NULL,last_error=?
            WHERE case_id=? AND current_attempt=?
        """, (
            "planned" if state == "interrupted" else state,
            message[-6000:], case_id, token,
        ))

def one_case(db, case, worker_sha):
    token = uuid.uuid4().hex
    folder = RUNS / case["case_id"] / token
    folder.mkdir(parents=True, exist_ok=False)

    case_obj = {
        "protocol": plan.PROTOCOL,
        **{key: case[key] for key in CASE_FIELDS},
    }
    save(folder / "case.json", case_obj)

    with db:
        db.execute("""
            INSERT INTO t26c_v2_attempts (
                attempt_id,case_id,started_utc,status,
                worker_sha256,folder
            ) VALUES (?,?,?,'running',?,?)
        """, (
            token, case["case_id"], plan.utc(),
            worker_sha, str(folder.resolve()),
        ))
        updated = db.execute("""
            UPDATE t26c_v2_cases
            SET status='running',current_attempt=?,last_error=NULL
            WHERE case_id=? AND status='planned'
        """, (token, case["case_id"]))
        assert updated.rowcount == 1

    print(
        f"\nCASE {case['model']} {case['direction']}",
        f"cut={case['cut']}",
        f"micro={case['microbatch_size']}",
        f"qos={case['qos']}",
        f"repeat={case['repeat']}",
        flush=True,
    )
    print("ATTEMPT:", folder, flush=True)

    interrupted = False
    timed_out = False
    with (
        (folder / "stdout.txt").open("w") as out,
        (folder / "stderr.txt").open("w") as err,
    ):
        proc = subprocess.Popen(
            [sys.executable, "-u", str(WORKER)],
            stdout=out,
            stderr=err,
            start_new_session=False,
            process_group=0,
            env={
                **os.environ,
                "T26C_V2_RUN_DIR": str(folder.resolve()),
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )
        (folder / "worker.pid").write_text(
            str(proc.pid) + "\n"
        )
        try:
            code = proc.wait(timeout=TIMEOUT_SECONDS)
        except KeyboardInterrupt:
            interrupted = True
            stop_group(proc)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_group(proc)
            code = proc.returncode

    stdout = (folder / "stdout.txt").read_text(
        errors="replace"
    )
    stderr = (folder / "stderr.txt").read_text(
        errors="replace"
    )

    evidence = None
    error = stderr
    if code == 0 and not interrupted and not timed_out:
        try:
            evidence = validate(folder, case, worker_sha)
        except Exception as exc:
            error += "\nVALIDATION: " + repr(exc)

    if evidence is None:
        if interrupted:
            state = "interrupted"
        elif "NUMERICAL_MISMATCH" in error or (
            "TOKEN_AGREEMENT_MISMATCH" in error
        ):
            state = "numerical_invalid"
        else:
            state = "failed"

        record_failure(
            db, case["case_id"], token, state,
            f"exit={code} timeout={timed_out}\n{error}",
        )

        print(
            "CASE PAUSED:", state,
            f"exit={code}",
            flush=True,
        )
        print("WORKER OUTPUT:", stdout[-1500:])
        print("WORKER ERROR:", error[-2500:])
        return False

    with db:
        saved_sha = db.execute("""
            SELECT value FROM t26c_v2_metadata
            WHERE key='worker_sha256'
        """).fetchone()
        if saved_sha is None:
            db.execute(
                "INSERT INTO t26c_v2_metadata VALUES (?,?)",
                ("worker_sha256", worker_sha),
            )
        else:
            assert saved_sha[0] == worker_sha

        for values in evidence:
            db.execute("""
                INSERT INTO t26c_v2_results
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, values + (token,))

        db.execute("""
            UPDATE t26c_v2_attempts
            SET status='completed',finished_utc=?
            WHERE attempt_id=?
        """, (plan.utc(), token))
        changed = db.execute("""
            UPDATE t26c_v2_cases
            SET status='completed',current_attempt=NULL
            WHERE case_id=? AND current_attempt=?
        """, (case["case_id"], token))
        assert changed.rowcount == 1

    seq, pipe = evidence
    print(
        "COMMITTED",
        f"speedup={pipe[3] / seq[3]:.3f}x",
        f"energy_ratio={pipe[7] / seq[7]:.3f}x",
        flush=True,
    )
    return True

def recover(db):
    running = list(db.execute("""
        SELECT a.attempt_id,a.folder,c.case_id
        FROM t26c_v2_attempts a
        JOIN t26c_v2_cases c ON a.case_id=c.case_id
        WHERE a.status='running'
    """))
    for item in running:
        pid_file = Path(item["folder"]) / "worker.pid"
        assert pid_file.is_file(), (
            "Cannot verify orphan worker PID", item["folder"]
        )
        pid = int(pid_file.read_text().strip())
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(
                f"WORKER_STILL_ACTIVE pid={pid}; "
                "refusing automatic recovery"
            )

    for item in running:
        record_failure(
            db, item["case_id"], item["attempt_id"],
            "interrupted", "RECOVERED_AFTER_PARENT_EXIT",
        )
    print("RECOVERED INTERRUPTED ATTEMPTS:", len(running))

def run(args):
    assert 1 <= args.limit <= 286
    assert WORKER.is_file()
    RUNS.mkdir(parents=True, exist_ok=True)

    with plan.LOCK.open("a+") as lock:
        fcntl.flock(
            lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
        )
        db = plan.connect()
        try:
            ensure_prepared(db)

            active = db.execute("""
                SELECT COUNT(*) FROM t26c_v2_cases
                WHERE status='running'
            """).fetchone()[0]
            assert active == 0, (
                "RUNNING_CASE_EXISTS; use recover "
                "only after verifying worker stopped"
            )

            worker_sha = sha(WORKER)
            for _ in range(args.limit):
                case = find_case(db, args)
                if case is None:
                    print("NO MATCHING PLANNED CASE")
                    break
                if not one_case(db, case, worker_sha):
                    break

            plan.report(db)
        finally:
            db.close()

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("run")
    command.add_argument("limit", type=int)
    command.add_argument("--model", choices=plan.MODELS)
    command.add_argument(
        "--direction", choices=plan.DIRECTIONS
    )
    command.add_argument("--cut", type=int)
    command.add_argument("--micro", type=int, choices=(1, 2))
    command.add_argument("--qos", choices=plan.QOS)
    command.add_argument("--repeat", type=int, choices=(1, 2))

    sub.add_parser("status")
    sub.add_parser("recover")

    args = parser.parse_args()

    if args.command == "run":
        run(args)
    elif args.command == "status":
        with plan.connect(readonly=True) as db:
            plan.report(db)
    else:
        with plan.LOCK.open("a+") as lock:
            fcntl.flock(
                lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
            with plan.connect() as db:
                ensure_prepared(db)
                recover(db)

if __name__ == "__main__":
    main()
