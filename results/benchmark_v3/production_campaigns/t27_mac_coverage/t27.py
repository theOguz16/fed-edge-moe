import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T23_DIR = ROOT.parent / "t23_mac_v1"
T23_SCRIPT = T23_DIR / "t23.py"
DB_PATH = T23_DIR / "dataset_v3_t23_mac.sqlite"
CSV_PATH = ROOT / "t27_selection_v1.csv"
PLAN_PATH = ROOT / "t27_selection_v1.json"

sys.path.insert(0, str(T23_DIR))
import t23

assert Path(t23.__file__).resolve() == T23_SCRIPT.resolve()
assert t23.DB.resolve() == DB_PATH.resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_plan(db):
    assert CSV_PATH.is_file(), CSV_PATH
    assert PLAN_PATH.is_file(), PLAN_PATH

    manifest = json.loads(PLAN_PATH.read_text())
    assert manifest["protocol"] == "T27_MAC_STRATIFIED_COVERAGE_V1"
    assert manifest["selected_new_tasks"] == 108
    assert manifest["selection_sha256"] == digest(CSV_PATH)

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 108
    rows.sort(key=lambda x: int(x["selection_order"]))
    assert [int(x["selection_order"]) for x in rows] == list(range(108))
    assert len({x["task_id"] for x in rows}) == 108
    assert Counter(x["model"] for x in rows) == {
        "resnet50": 36,
        "convnext_base": 36,
        "convnext_large": 36,
    }

    sql = """
        SELECT t.task_id,t.state,t.claim_token,
               t.cpu_threads,t.attempt_count,
               p.model,p.group_mapping,
               w.batch_size,w.microbatch_size,
               c.run_order,
               m.task_id AS measurement_id
        FROM tasks t
        JOIN t23_campaign_tasks c ON c.task_id=t.task_id
        JOIN placements p ON p.placement_id=t.placement_id
        JOIN workloads w ON w.workload_id=t.workload_id
        LEFT JOIN t23_measurements m ON m.task_id=t.task_id
        WHERE t.task_id=?
    """

    for item in rows:
        record = db.execute(sql, (item["task_id"],)).fetchone()
        assert record is not None, item["task_id"]

        assert record["model"] == item["model"]
        assert record["group_mapping"] == item["placement_mapping"]
        assert record["run_order"] == int(item["original_run_order"])
        assert record["batch_size"] == int(item["batch_size"])
        assert record["microbatch_size"] == int(item["microbatch_size"])
        assert record["cpu_threads"] == int(item["cpu_threads"])

        assert record["state"] in (
            "queued", "completed", "failed", "running"
        )
        assert (
            record["measurement_id"] is not None
        ) == (record["state"] == "completed")

        item["state"] = record["state"]

    assert db.execute("""
        SELECT COUNT(*) FROM t23_campaign_tasks
    """).fetchone()[0] == 860

    return rows


def report(db):
    rows = verify_plan(db)
    counts = Counter(x["state"] for x in rows)

    print("\n=== T27 SELECTED TASK STATUS ===", flush=True)
    print("Selection SHA256:", digest(CSV_PATH), flush=True)
    print("Worker SHA256:", digest(T23_SCRIPT), flush=True)
    print("Selected:", len(rows), flush=True)
    print("States:", dict(counts), flush=True)
    print(
        "T23 total measurements:",
        db.execute(
            "SELECT COUNT(*) FROM t23_measurements"
        ).fetchone()[0],
        flush=True,
    )
    print(
        "T23 total queued:",
        db.execute(
            "SELECT COUNT(*) FROM t23_campaign_tasks c "
            "JOIN tasks t ON c.task_id=t.task_id "
            "WHERE t.state='queued'"
        ).fetchone()[0],
        flush=True,
    )
    return rows


def claim_selected(db, task_id):
    db.execute("BEGIN IMMEDIATE")

    try:
        record = db.execute("""
            SELECT t.task_id,p.model,p.group_mapping,
                   w.batch_size,w.microbatch_size,
                   t.cpu_threads,t.attempt_count
            FROM tasks t
            JOIN t23_campaign_tasks c ON c.task_id=t.task_id
            JOIN placements p ON p.placement_id=t.placement_id
            JOIN workloads w ON w.workload_id=t.workload_id
            WHERE t.task_id=? AND t.state='queued'
        """, (task_id,)).fetchone()

        assert record is not None, task_id

        data = dict(record)
        token = uuid.uuid4().hex

        updated = db.execute("""
            UPDATE tasks
            SET state='running',
                claim_token=?,
                started_utc=?,
                attempt_count=attempt_count+1
            WHERE task_id=? AND state='queued'
        """, (token, t23.utc(), task_id))

        assert updated.rowcount == 1

        db.execute("""
            INSERT INTO attempts
                (attempt_id,task_id,state,started_utc)
            VALUES (?,?,'running',?)
        """, (token, task_id, t23.utc()))

        db.commit()
        data["token"] = token
        return data

    except BaseException:
        db.rollback()
        raise


def interrupt_selected(db, task):
    db.execute("BEGIN IMMEDIATE")

    try:
        db.execute("""
            UPDATE attempts
            SET state='interrupted',
                finished_utc=?,
                message='T27 interrupted after worker termination'
            WHERE attempt_id=? AND state='running'
        """, (t23.utc(), task["token"]))

        updated = db.execute("""
            UPDATE tasks
            SET state='queued',
                claim_token=NULL,
                started_utc=NULL
            WHERE task_id=?
              AND state='running'
              AND claim_token=?
        """, (task["task_id"], task["token"]))

        assert updated.rowcount == 1
        db.commit()

    except BaseException:
        db.rollback()
        raise


def stop_worker(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
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


def execute_task(db, item, index, total):
    task = claim_selected(db, item["task_id"])

    folder = (
        T23_DIR / "results" / task["task_id"]
        / ("attempt_" + task["token"])
    )

    try:
        folder.mkdir(parents=True, exist_ok=False)

        task_path = folder / "task.json"
        result_path = folder / "result.json"
        stdout_path = folder / "stdout.txt"
        stderr_path = folder / "stderr.txt"

        t23.atomic_json(task_path, task)

        print(
            f"[{index}/{total}]",
            task["model"],
            "batch=", task["batch_size"],
            "micro=", task["microbatch_size"],
            "threads=", task["cpu_threads"],
            "task=", task["task_id"][:16],
            flush=True,
        )

        with stdout_path.open("w") as stdout, \
             stderr_path.open("w") as stderr:

            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(T23_SCRIPT),
                    "worker",
                    str(task_path),
                    str(result_path),
                ],
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env={
                    **os.environ,
                    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                    "TORCH_HOME": os.environ.get(
                        "TORCH_HOME",
                        str(Path.home() / ".cache/torch"),
                    ),
                },
            )

            try:
                exit_code = proc.wait(timeout=t23.TIMEOUT)

            except KeyboardInterrupt:
                stop_worker(proc)
                interrupt_selected(db, task)
                print(
                    "\nT27 PAUSED: worker terminated; "
                    "task returned to queue.",
                    flush=True,
                )
                return False

            except subprocess.TimeoutExpired:
                stop_worker(proc)
                exit_code = 124

        if exit_code == 0 and result_path.is_file():
            t23.complete(db, task, result_path)
            summary = stdout_path.read_text(
                errors="replace"
            ).strip()[-240:]
            print("COMMITTED:", summary, flush=True)

        else:
            error = (
                f"exit={exit_code}\n"
                + stderr_path.read_text(
                    errors="replace"
                )[-2600:]
            )
            t23.failure(db, task, error)
            print(
                "FAILED RECORDED:",
                task["task_id"][:16],
                error[-400:],
                flush=True,
            )

        return True

    except Exception as exc:
        current = db.execute(
            "SELECT state,claim_token FROM tasks WHERE task_id=?",
            (task["task_id"],),
        ).fetchone()

        if (
            current["state"] == "running"
            and current["claim_token"] == task["token"]
        ):
            t23.failure(
                db, task,
                "T27 runner exception: " + repr(exc),
            )

        raise


def run(limit):
    assert DB_PATH.is_file(), DB_PATH
    assert limit >= 1

    # Same lock used by the original T23 runner.
    with (T23_DIR / ".campaign.lock").open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB,
        )

        db = t23.connect()
        try:
            assert db.execute(
                "PRAGMA quick_check"
            ).fetchone()[0] == "ok"

            rows = report(db)

            # Never recover unrelated T23 tasks automatically.
            running = [x for x in rows if x["state"] == "running"]
            assert not running, (
                "Previously running selected task detected. "
                "Inspect worker processes before recovery.",
                [x["task_id"] for x in running],
            )

            t23.free_space()

            queue = [
                x for x in rows
                if x["state"] == "queued"
            ][:limit]

            print(
                "\nT27 SESSION START",
                "limit=", limit,
                "selected_for_session=", len(queue),
                flush=True,
            )

            for index, item in enumerate(queue, 1):
                t23.free_space()
                continued = execute_task(
                    db, item, index, len(queue)
                )
                if not continued:
                    break

            report(db)
            print("\nT27 SESSION FINISHED", flush=True)
            print("ORIGINAL T23 SOURCE CODE MODIFIED: NO")

        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "run"))
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    assert Path.cwd().resolve() == ROOT.parents[3], (
        "Run from the fed-edge-moe repository root"
    )
    assert DB_PATH.is_file()

    if args.command == "status":
        with t23.connect(readonly=True) as db:
            assert db.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"
            report(db)
        print("SQLITE MODIFIED: NO")
    else:
        run(args.limit)


if __name__ == "__main__":
    main()
