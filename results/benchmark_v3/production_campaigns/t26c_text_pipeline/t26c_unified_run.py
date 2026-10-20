import argparse
import fcntl
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT.parent / "t23_mac_v1/dataset_v3_t23_mac.sqlite"
LOCK = DB.parent / ".campaign.lock"

PROTOCOL = "T26C_UNIFIED_TEXT_PIPELINE_FP32_V2"
MODELS = {
    "distilgpt2": (2, 3, 4),
    "qwen2_5_1_5b": (7, 14, 21),
    "qwen3_1_7b": (7, 14, 21),
}
DIRECTIONS = ("cpu_mps", "mps_cpu")
MICROS = (1, 2)
QOS = ("background", "utility", "default", "user_initiated")
REPEATS = (1, 2)
EXPECTED = 288

def utc():
    return datetime.now(timezone.utc).isoformat()

def identity(payload):
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return "t26cv2_" + hashlib.sha256(raw).hexdigest()

def connect(readonly=False):
    if readonly:
        db = sqlite3.connect(
            f"file:{DB.resolve()}?mode=ro",
            uri=True,
            timeout=60,
        )
    else:
        db = sqlite3.connect(str(DB), timeout=60)

    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("PRAGMA foreign_keys=ON")

    if not readonly:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")

    return db

def cases():
    for model, cuts in MODELS.items():
        for cut, direction, micro, qos, repeat in product(
            cuts, DIRECTIONS, MICROS, QOS, REPEATS
        ):
            payload = {
                "protocol": PROTOCOL,
                "model": model,
                "precision": "fp32",
                "direction": direction,
                "cut": cut,
                "batch_size": 4,
                "context_tokens": 128,
                "microbatch_size": micro,
                "qos": qos,
                "repeat": repeat,
                "cpu_threads": 4,
            }
            yield identity(payload), payload

def verify_legacy(db):
    required = {
        "t26c_cases",
        "t26c_results",
        "t26c_metadata",
    }
    existing = {
        x[0] for x in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert required <= existing, (
        "T26C legacy tables missing", required - existing
    )

    rows = db.execute("""
        SELECT c.repeat, c.case_id, c.status,
               COUNT(r.mode) AS modes
        FROM t26c_cases c
        LEFT JOIN t26c_results r ON r.case_id=c.case_id
        GROUP BY c.repeat,c.case_id,c.status
        ORDER BY c.repeat
    """).fetchall()

    assert len(rows) == 2
    assert [
        (r["repeat"], r["status"], r["modes"])
        for r in rows
    ] == [
        (1, "completed", 2),
        (2, "completed", 2),
    ], [tuple(r) for r in rows]

    old_protocol = db.execute(
        "SELECT value FROM t26c_metadata "
        "WHERE key='protocol'"
    ).fetchone()
    assert old_protocol is not None
    assert old_protocol[0] == (
        "T26C_DISTILGPT2_CPU_MPS_FP32_PRODUCTION_V1"
    )

    return {r["repeat"]: r["case_id"] for r in rows}

def prepare():
    assert DB.is_file(), DB
    assert len(list(cases())) == EXPECTED

    with LOCK.open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )

        db = connect()
        try:
            assert db.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"

            legacy = verify_legacy(db)

            db.execute("""
                CREATE TABLE IF NOT EXISTS t26c_v2_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS t26c_v2_cases (
                    case_id TEXT PRIMARY KEY,
                    ordinal INTEGER UNIQUE NOT NULL,
                    model TEXT NOT NULL,
                    precision TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    cut INTEGER NOT NULL,
                    batch_size INTEGER NOT NULL,
                    context_tokens INTEGER NOT NULL,
                    microbatch_size INTEGER NOT NULL,
                    qos TEXT NOT NULL,
                    repeat INTEGER NOT NULL,
                    cpu_threads INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    legacy_case_id TEXT,
                    current_attempt TEXT,
                    last_error TEXT,
                    UNIQUE (
                        model,precision,direction,cut,
                        batch_size,context_tokens,
                        microbatch_size,qos,repeat
                    )
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS t26c_v2_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    started_utc TEXT NOT NULL,
                    finished_utc TEXT,
                    status TEXT NOT NULL,
                    worker_sha256 TEXT,
                    folder TEXT NOT NULL,
                    error TEXT,
                    FOREIGN KEY(case_id)
                        REFERENCES t26c_v2_cases(case_id)
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS t26c_v2_results (
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
                    power_path TEXT NOT NULL,
                    power_sha256 TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    PRIMARY KEY(case_id,mode),
                    FOREIGN KEY(case_id)
                        REFERENCES t26c_v2_cases(case_id),
                    FOREIGN KEY(attempt_id)
                        REFERENCES t26c_v2_attempts(attempt_id)
                )
            """)

            existing_protocol = db.execute("""
                SELECT value FROM t26c_v2_metadata
                WHERE key='protocol'
            """).fetchone()

            if existing_protocol:
                assert existing_protocol[0] == PROTOCOL
            else:
                db.execute(
                    "INSERT INTO t26c_v2_metadata VALUES (?,?)",
                    ("protocol", PROTOCOL),
                )
                db.execute(
                    "INSERT INTO t26c_v2_metadata VALUES (?,?)",
                    ("energy_scope",
                     "estimated_cpu_gpu_soc_not_wall"),
                )
                db.execute(
                    "INSERT INTO t26c_v2_metadata VALUES (?,?)",
                    ("throughput_scope",
                     "wikitext2_prefill_tokens_not_generation"),
                )
                db.execute(
                    "INSERT INTO t26c_v2_metadata VALUES (?,?)",
                    ("gpu_kernel_overlap_proven", "false"),
                )

            count = db.execute(
                "SELECT COUNT(*) FROM t26c_v2_cases"
            ).fetchone()[0]

            if count == 0:
                records = []

                for ordinal, (case_id, payload) in enumerate(
                    cases()
                ):
                    linked = (
                        payload["model"] == "distilgpt2"
                        and payload["direction"] == "cpu_mps"
                        and payload["cut"] == 3
                        and payload["microbatch_size"] == 1
                        and payload["qos"] == "default"
                    )

                    records.append((
                        case_id,
                        ordinal,
                        payload["model"],
                        payload["precision"],
                        payload["direction"],
                        payload["cut"],
                        payload["batch_size"],
                        payload["context_tokens"],
                        payload["microbatch_size"],
                        payload["qos"],
                        payload["repeat"],
                        payload["cpu_threads"],
                        "legacy_linked" if linked else "planned",
                        legacy[payload["repeat"]] if linked else None,
                    ))

                assert len(records) == EXPECTED

                db.executemany("""
                    INSERT INTO t26c_v2_cases (
                        case_id,ordinal,model,precision,
                        direction,cut,batch_size,context_tokens,
                        microbatch_size,qos,repeat,cpu_threads,
                        status,legacy_case_id
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, records)

            else:
                assert count == EXPECTED, count
                actual = {
                    r["case_id"] for r in db.execute(
                        "SELECT case_id FROM t26c_v2_cases"
                    )
                }
                planned = {cid for cid, _ in cases()}
                assert actual == planned, (
                    "Campaign identity mismatch"
                )

            db.commit()

            print("T26C V2 PREPARE: PASS")
            print("LEGACY MEASUREMENTS MODIFIED: NO")
            report(db)

        finally:
            db.close()

def report(db):
    rows = db.execute("""
        SELECT status,COUNT(*) AS n
        FROM t26c_v2_cases
        GROUP BY status ORDER BY status
    """).fetchall()
    counts = {r["status"]: r["n"] for r in rows}

    print("TOTAL CASES:", sum(counts.values()))
    print("CASE STATES:", counts)
    print("NEW MODE RESULTS:", db.execute(
        "SELECT COUNT(*) FROM t26c_v2_results"
    ).fetchone()[0])
    print("LEGACY LINKED:", counts.get("legacy_linked", 0))
    print("DATABASE:", DB)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "status")
    )
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    else:
        db = connect(readonly=True)
        try:
            report(db)
        finally:
            db.close()

if __name__ == "__main__":
    main()
