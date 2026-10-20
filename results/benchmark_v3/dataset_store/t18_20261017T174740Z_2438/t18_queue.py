import csv
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

T17 = Path(
    "results/benchmark_v3/dataset_v3_design/"
    "t17_20261017T160851Z_2438"
).resolve()

EXPECTED = {
    "placements": 3412,
    "workloads": 10,
    "sequential": 204720,
    "pipeline": 122616,
    "total": 327336,
}

THREADS = (1, 2, 4, 6, 8, 10)
BATCH = 2000

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def readcsv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def check_t17():
    checks = json.loads(
        (T17 / "checksums.json").read_text()
    )
    for name, expected in checks.items():
        assert sha(T17 / name) == expected, name

    plan = json.loads(
        (T17 / "planning_summary.json").read_text()
    )

    assert plan["unique_vision_placements"] == 3412
    assert plan["logical_sequential_candidates"] == 204720
    assert plan["logical_pipeline_candidates"] == 122616
    assert plan["logical_total_candidates"] == 327336

    placements = readcsv(T17 / "placement_catalog.csv")
    workloads = readcsv(T17 / "workload_catalog.csv")
    models = readcsv(T17 / "model_catalog.csv")

    assert len(placements) == 3412
    assert len(workloads) == 10
    assert len(models) == 3
    assert len({
        p["placement_id"] for p in placements
    }) == 3412
    assert len({
        w["workload_id"] for w in workloads
    }) == 10

    return placements, workloads

def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(path),
        timeout=30,
        isolation_level=None,
    )
    conn.execute("PRAGMA busy_timeout=30000")
    mode = conn.execute(
        "PRAGMA journal_mode=WAL"
    ).fetchone()[0]
    assert mode.lower() == "wal"
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS placements (
    placement_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    group_mapping TEXT NOT NULL,
    transition_count INTEGER NOT NULL,
    uses_cpu INTEGER NOT NULL,
    uses_accelerator INTEGER NOT NULL,
    manifest_sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workloads (
    workload_id TEXT PRIMARY KEY,
    batch_size INTEGER NOT NULL,
    microbatch_size INTEGER NOT NULL,
    microbatch_count INTEGER NOT NULL,
    pipeline_candidate INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    placement_id TEXT NOT NULL,
    workload_id TEXT NOT NULL,
    cpu_threads INTEGER NOT NULL,
    execution_mode TEXT NOT NULL,
    precision TEXT NOT NULL,
    hardware_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'planned',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    claim_token TEXT,
    started_utc TEXT,
    finished_utc TEXT,
    result_path TEXT,
    result_sha256 TEXT,
    last_error TEXT,
    FOREIGN KEY (placement_id)
        REFERENCES placements(placement_id),
    FOREIGN KEY (workload_id)
        REFERENCES workloads(workload_id),
    CHECK (state IN (
        'planned', 'queued', 'running',
        'completed', 'failed', 'unsupported'
    )),
    CHECK (execution_mode IN (
        'sequential', 'pipeline'
    ))
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    state TEXT NOT NULL,
    started_utc TEXT NOT NULL,
    finished_utc TEXT,
    message TEXT,
    FOREIGN KEY (task_id)
        REFERENCES tasks(task_id),
    CHECK (state IN (
        'running', 'interrupted',
        'completed', 'failed'
    ))
);

CREATE INDEX IF NOT EXISTS ix_tasks_state
ON tasks(state, task_id);

CREATE INDEX IF NOT EXISTS ix_tasks_placement
ON tasks(placement_id);

CREATE INDEX IF NOT EXISTS ix_attempts_task
ON attempts(task_id);
"""

def create_schema(conn):
    conn.executescript(SCHEMA)

def task_id(placement_id, workload_id, threads, mode):
    identity = (
        f"dataset_v3|mac_m4|mps|fp32|"
        f"{placement_id}|{workload_id}|"
        f"{threads}|{mode}"
    )
    return "task_" + hashlib.sha256(
        identity.encode()
    ).hexdigest()

def task_records(placements, workloads):
    for placement in placements:
        pid = placement["placement_id"]
        heterogeneous = (
            int(placement["uses_cpu"]) == 1
            and int(placement["uses_accelerator"]) == 1
        )

        for threads in THREADS:
            for workload in workloads:
                wid = workload["workload_id"]

                yield (
                    task_id(pid, wid, threads, "sequential"),
                    pid, wid, threads,
                    "sequential", "float32",
                    "mac_m4", "planned",
                )

                if (
                    heterogeneous
                    and int(workload["pipeline_candidate"]) == 1
                ):
                    yield (
                        task_id(pid, wid, threads, "pipeline"),
                        pid, wid, threads,
                        "pipeline", "float32",
                        "mac_m4", "planned",
                    )

def count_rows(conn, table):
    return conn.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

def init_database(path):
    placements, workloads = check_t17()
    conn = connect(path)
    create_schema(conn)

    provenance = {
        "protocol": "T18_resumable_queue_v1",
        "hardware_id": "mac_m4",
        "accelerator": "mps",
        "source_t17_path": str(T17),
        "t17_checksums_sha256": sha(
            T17 / "checksums.json"
        ),
        "t17_placement_sha256": sha(
            T17 / "placement_catalog.csv"
        ),
        "t17_workload_sha256": sha(
            T17 / "workload_catalog.csv"
        ),
    }

    existing = dict(
        conn.execute(
            "SELECT key, value FROM metadata"
        ).fetchall()
    )

    for key, value in provenance.items():
        if key in existing:
            assert existing[key] == str(value), (
                "Source mismatch; refusing to mix "
                f"datasets: {key}"
            )

    conn.execute("BEGIN IMMEDIATE")
    try:
        for key, value in provenance.items():
            conn.execute(
                "INSERT OR IGNORE INTO metadata "
                "(key,value) VALUES (?,?)",
                (key, str(value))
            )

        conn.executemany(
            "INSERT OR IGNORE INTO placements "
            "(placement_id,model,group_mapping,"
            "transition_count,uses_cpu,"
            "uses_accelerator,manifest_sha256) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (
                    p["placement_id"],
                    p["model"],
                    p["group_mapping"],
                    int(p["transition_count"]),
                    int(p["uses_cpu"]),
                    int(p["uses_accelerator"]),
                    p["manifest_sha256"],
                )
                for p in placements
            ]
        )

        conn.executemany(
            "INSERT OR IGNORE INTO workloads "
            "(workload_id,batch_size,microbatch_size,"
            "microbatch_count,pipeline_candidate) "
            "VALUES (?,?,?,?,?)",
            [
                (
                    w["workload_id"],
                    int(w["batch_size"]),
                    int(w["microbatch_size"]),
                    int(w["microbatch_count"]),
                    int(w["pipeline_candidate"]),
                )
                for w in workloads
            ]
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    insert_sql = (
        "INSERT OR IGNORE INTO tasks "
        "(task_id,placement_id,workload_id,"
        "cpu_threads,execution_mode,precision,"
        "hardware_id,state) "
        "VALUES (?,?,?,?,?,?,?,?)"
    )

    buffer = []
    generated = 0

    for record in task_records(placements, workloads):
        buffer.append(record)
        generated += 1

        if len(buffer) >= BATCH:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(insert_sql, buffer)
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            buffer.clear()

        if generated % 50000 == 0:
            print(
                f"CATALOG GENERATED: {generated}/327336",
                flush=True
            )

    if buffer:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany(insert_sql, buffer)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    assert generated == EXPECTED["total"]

    print(
        "QUEUE GENERATION: PASS",
        flush=True
    )
    conn.close()

def activate(conn, tid):
    conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.execute(
            "UPDATE tasks SET state='queued' "
            "WHERE task_id=? AND state='planned'",
            (tid,)
        )
        assert cursor.rowcount == 1
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

def claim(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT task_id FROM tasks "
            "WHERE state='queued' "
            "ORDER BY task_id LIMIT 1"
        ).fetchone()

        if row is None:
            conn.commit()
            return None

        tid = row[0]
        token = uuid.uuid4().hex

        updated = conn.execute(
            "UPDATE tasks SET state='running', "
            "claim_token=?, attempt_count=attempt_count+1,"
            "started_utc=? "
            "WHERE task_id=? AND state='queued'",
            (token, now(), tid)
        )
        assert updated.rowcount == 1

        conn.execute(
            "INSERT INTO attempts "
            "(attempt_id,task_id,state,started_utc) "
            "VALUES (?,?,?,?)",
            (token, tid, "running", now())
        )

        conn.commit()
        return tid, token
    except BaseException:
        conn.rollback()
        raise

def recover_interrupted(conn):
    # Only call at startup after confirming that
    # no benchmark worker is still running.
    conn.execute("BEGIN IMMEDIATE")
    try:
        running = conn.execute(
            "SELECT task_id,claim_token FROM tasks "
            "WHERE state='running'"
        ).fetchall()

        for tid, token in running:
            updated = conn.execute(
                "UPDATE attempts SET "
                "state='interrupted', finished_utc=?, "
                "message='recovered_after_interruption' "
                "WHERE attempt_id=? AND state='running'",
                (now(), token)
            )
            assert updated.rowcount == 1

            updated = conn.execute(
                "UPDATE tasks SET state='queued', "
                "claim_token=NULL, started_utc=NULL "
                "WHERE task_id=? AND claim_token=?",
                (tid, token)
            )
            assert updated.rowcount == 1

        conn.commit()
        return len(running)
    except BaseException:
        conn.rollback()
        raise

def finish(conn, tid, token, result_path):
    result_path = Path(result_path).resolve()
    assert result_path.is_file()

    content_sha = sha(result_path)

    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT state,claim_token FROM tasks "
            "WHERE task_id=?",
            (tid,)
        ).fetchone()

        assert row == ("running", token), (
            "Task is not owned by this attempt"
        )

        updated = conn.execute(
            "UPDATE tasks SET state='completed', "
            "finished_utc=?, result_path=?, "
            "result_sha256=?, claim_token=NULL "
            "WHERE task_id=? AND state='running'",
            (now(), str(result_path), content_sha, tid)
        )
        assert updated.rowcount == 1

        updated = conn.execute(
            "UPDATE attempts SET state='completed', "
            "finished_utc=? "
            "WHERE attempt_id=? AND state='running'",
            (now(), token)
        )
        assert updated.rowcount == 1

        conn.commit()
    except BaseException:
        conn.rollback()
        raise

def integrity(conn, full_catalog=True):
    result = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]
    assert result == "ok", result

    assert conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall() == []

    states = dict(
        conn.execute(
            "SELECT state,COUNT(*) FROM tasks "
            "GROUP BY state"
        ).fetchall()
    )

    if full_catalog:
        assert count_rows(conn, "placements") == 3412
        assert count_rows(conn, "workloads") == 10
        assert count_rows(conn, "tasks") == 327336

        modes = dict(
            conn.execute(
                "SELECT execution_mode,COUNT(*) "
                "FROM tasks GROUP BY execution_mode"
            ).fetchall()
        )
        assert modes == {
            "sequential": 204720,
            "pipeline": 122616,
        }

        model_counts = dict(
            conn.execute(
                "SELECT p.model,COUNT(*) "
                "FROM tasks t JOIN placements p "
                "ON t.placement_id=p.placement_id "
                "GROUP BY p.model"
            ).fetchall()
        )

        assert model_counts == {
            "resnet50": 6072,
            "convnext_base": 160632,
            "convnext_large": 160632,
        }

        assert sum(states.values()) == 327336

        for key, filename in (
            ("t17_checksums_sha256", "checksums.json"),
            ("t17_placement_sha256", "placement_catalog.csv"),
            ("t17_workload_sha256", "workload_catalog.csv"),
        ):
            stored = conn.execute(
                "SELECT value FROM metadata WHERE key=?",
                (key,)
            ).fetchone()
            assert stored is not None
            assert stored[0] == sha(T17 / filename)

    completed = conn.execute(
        "SELECT result_path,result_sha256 "
        "FROM tasks WHERE state='completed'"
    ).fetchall()

    for path, expected in completed:
        assert path and expected
        assert sha(path) == expected, (
            f"Completed result missing/corrupt: {path}"
        )

    print("SQLITE INTEGRITY: PASS")
    print("FOREIGN KEYS: PASS")
    print("RESULT ARTIFACT SHA256: PASS")
    print("TASK STATES:", states)

def selftest(path):
    path = Path(path)
    assert not path.exists(), (
        "Self-test database already exists"
    )

    conn = connect(path)
    create_schema(conn)

    conn.execute(
        "INSERT INTO placements VALUES "
        "('synthetic_placement','synthetic_model',"
        "'CA',1,1,1,'synthetic_hash')"
    )
    conn.execute(
        "INSERT INTO workloads VALUES "
        "('synthetic_workload',1,1,1,0)"
    )

    for i in range(2):
        conn.execute(
            "INSERT INTO tasks "
            "(task_id,placement_id,workload_id,"
            "cpu_threads,execution_mode,precision,"
            "hardware_id,state) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                f"synthetic_task_{i}",
                "synthetic_placement",
                "synthetic_workload",
                1,
                "sequential",
                "float32",
                "synthetic_only",
                "planned",
            )
        )

    activate(conn, "synthetic_task_0")
    activate(conn, "synthetic_task_1")

    first = claim(conn)
    assert first is not None

    # Simulate an abrupt worker disappearance:
    # close without completing the claimed task.
    conn.close()
    conn = connect(path)

    recovered = recover_interrupted(conn)
    assert recovered == 1

    assert conn.execute(
        "SELECT COUNT(*) FROM attempts "
        "WHERE state='interrupted'"
    ).fetchone()[0] == 1

    resumed = claim(conn)
    assert resumed is not None
    assert resumed[0] == first[0]
    assert resumed[1] != first[1]

    artifact = path.with_name(
        "synthetic_result.json"
    )

    atomic_json(artifact, {
        "type": "synthetic_infrastructure_test",
        "real_benchmark_measurement": False,
        "task_id": resumed[0],
    })

    finish(
        conn, resumed[0], resumed[1],
        artifact
    )

    assert conn.execute(
        "SELECT COUNT(*) FROM tasks "
        "WHERE state='completed'"
    ).fetchone()[0] == 1

    assert conn.execute(
        "SELECT attempt_count FROM tasks "
        "WHERE task_id=?",
        (resumed[0],)
    ).fetchone()[0] == 2

    # The completed task must not be claimable again.
    next_task = claim(conn)
    assert next_task is not None
    assert next_task[0] != resumed[0]

    assert conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    conn.close()

    print("INTERRUPTED TASK RECOVERY: PASS")
    print("COMPLETED TASK DEDUPLICATION: PASS")
    print("SYNTHETIC RESULT HASH RECORD: PASS")
    print("SELF-TEST ISOLATED FROM REAL DATASET: PASS")

def backup(source, destination):
    source = Path(source)
    destination = Path(destination)

    assert source.is_file()
    assert not destination.exists()

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    temporary = destination.with_name(
        destination.name + ".tmp"
    )
    assert not temporary.exists()

    src = sqlite3.connect(str(source))
    dst = sqlite3.connect(str(temporary))

    try:
        src.backup(dst, pages=1000, sleep=0.05)

        assert dst.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        src_count = src.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0]
        dst_count = dst.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0]

        assert src_count == dst_count
    finally:
        dst.close()
        src.close()

    os.replace(temporary, destination)

    digest = sha(destination)
    assert len(digest) == 64

    atomic_json(
        destination.with_suffix(
            destination.suffix + ".sha256.json"
        ),
        {
            "type": "sqlite_consistent_backup",
            "source": str(source.resolve()),
            "backup": str(destination.resolve()),
            "sha256": digest,
            "bytes": destination.stat().st_size,
            "task_count": dst_count,
            "created_utc": now(),
        }
    )

    print("CONSISTENT SQLITE BACKUP: PASS")
    print("BACKUP SHA256:", digest)
    print("BACKUP PATH:", destination)

def main():
    assert len(sys.argv) >= 3, (
        "Usage: t18_queue.py "
        "init|verify|selftest|backup PATH [DEST]"
    )

    action = sys.argv[1]
    path = Path(sys.argv[2]).resolve()

    if action == "init":
        print("T17 SOURCE CHECKSUMS: VALIDATING", flush=True)
        init_database(path)
        conn = connect(path)
        try:
            integrity(conn)
        finally:
            conn.close()

        print("T18 CATALOG INITIALIZATION: PASS")

    elif action == "verify":
        conn = connect(path)
        try:
            integrity(conn)
        finally:
            conn.close()

        print("T18 DATASET INTEGRITY: PASS")

    elif action == "selftest":
        selftest(path)
        print("T18 RESUME SELF-TEST: PASS")

    elif action == "backup":
        assert len(sys.argv) == 4
        backup(path, Path(sys.argv[3]).resolve())
        print("T18 BACKUP: PASS")

    else:
        raise ValueError(
            f"Unknown action: {action}"
        )

if __name__ == "__main__":
    main()
