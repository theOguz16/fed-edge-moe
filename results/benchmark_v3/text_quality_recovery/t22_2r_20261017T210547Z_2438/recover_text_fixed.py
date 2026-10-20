import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()

OLD = BASE / (
    "text_quality_campaigns/"
    "t22_2_20261017T205158Z_2438"
)

OLD_SCRIPT = OLD / "t22_2.py"
OLD_DB = OLD / "dataset_v3_with_text_quality.sqlite"
OLD_CORPUS = OLD / "wikitext2_raw_test.parquet"

WORKER = ROOT / "t22_2_recovery_worker_fixed.py"
DB = ROOT / "dataset_v3_text_quality_recovered.sqlite"
CORPUS = ROOT / "wikitext2_raw_test.parquet"

CASES = [
    ("distilgpt2", "bf16"),
    ("qwen2_5_1_5b", "bf16"),
    ("qwen3_1_7b", "fp32"),
    ("qwen3_1_7b", "fp16"),
    ("qwen3_1_7b", "bf16"),
]

EXPECTED = [
    (model, precision)
    for model in (
        "distilgpt2",
        "qwen2_5_1_5b",
        "qwen3_1_7b",
    )
    for precision in ("fp32", "fp16", "bf16")
]

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(4 * 1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

def save(path, obj):
    atomic(
        path,
        (
            json.dumps(
                obj, indent=2, ensure_ascii=False
            ) + "\n"
        ).encode()
    )

def connect(path=DB):
    db = sqlite3.connect(str(path), timeout=90)
    db.execute("PRAGMA busy_timeout=90000")
    db.execute("PRAGMA synchronous=FULL")
    return db

def disk_guard():
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise RuntimeError(
            "Disk guard: less than 4 GiB available"
        )

def create_worker():
    source = OLD_SCRIPT.read_text()

    changes = [
        (
            'PRECISIONS = ("fp32", "fp16")',
            'PRECISIONS = ("fp32", "fp16", "bf16")'
        ),
        (
            '''            dtype = (
                torch.float32
                if precision == "fp32"
                else torch.float16
            )''',
            '''            dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[precision]'''
        ),
        (
            '''                    scores = logits.float()

                    nll_sum = float(''',
            '''                    scores = logits.float()

                    if not bool(torch.isfinite(scores).all().item()):
                        raise RuntimeError(
                            f"NONFINITE_LOGITS at window {index}"
                        )

                    nll_sum = float('''
        ),
    ]

    # Recovery v2: use the actual recovered database.
    # The old worker pointed at a nonexistent SQLite filename.
    changes.append((
        'DB = ROOT / "dataset_v3_with_text_quality.sqlite"',
        'DB = ROOT / "dataset_v3_text_quality_recovered.sqlite"'
    ))

    for original, replacement in changes:
        if source.count(original) != 1:
            raise RuntimeError(
                "Source patch did not match uniquely; "
                "refusing unsafe modification."
            )
        source = source.replace(
            original, replacement, 1
        )

    compile(source, str(WORKER), "exec")
    content = source.encode()

    if WORKER.exists():
        assert WORKER.read_bytes() == content
    else:
        atomic(WORKER, content)

def copy_corpus():
    assert OLD_CORPUS.is_file()

    if not CORPUS.exists():
        temp = CORPUS.with_suffix(".tmp")
        shutil.copyfile(OLD_CORPUS, temp)
        assert sha(temp) == sha(OLD_CORPUS)
        os.replace(temp, CORPUS)

    assert sha(CORPUS) == sha(OLD_CORPUS)

    # Retain the original tokenized streams.
    for old_file in OLD.glob("*_token_ids.zlib"):
        new_file = ROOT / old_file.name

        if not new_file.exists():
            temp = ROOT / (old_file.name + ".tmp")
            shutil.copyfile(old_file, temp)
            assert sha(temp) == sha(old_file)
            os.replace(temp, new_file)

        assert sha(new_file) == sha(old_file)

def prepare_database():
    if DB.exists():
        db = connect()
        try:
            assert db.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"

            assert db.execute(
                "SELECT COUNT(*) "
                "FROM vision_quality_predictions"
            ).fetchone()[0] == 60000
        finally:
            db.close()
        return

    src = sqlite3.connect(
        f"file:{OLD_DB}?mode=ro",
        uri=True
    )

    temporary = ROOT / (
        "snapshot_" + uuid.uuid4().hex + ".tmp"
    )
    dest = connect(temporary)

    try:
        assert src.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        assert src.execute(
            "SELECT COUNT(*) FROM text_quality_jobs "
            "WHERE status='completed'"
        ).fetchone()[0] == 3

        fp16 = src.execute(
            """
            SELECT status FROM text_quality_jobs
            WHERE model='qwen2_5_1_5b'
              AND precision='fp16'
            """
        ).fetchone()

        assert fp16 == ("failed",)

        assert src.execute(
            """
            SELECT COUNT(*) FROM text_quality_windows
            WHERE model='qwen2_5_1_5b'
              AND precision='fp16'
            """
        ).fetchone()[0] == 0

        src.backup(dest)

        dest.execute("""
            CREATE TABLE text_quality_failure_evidence (
                model TEXT NOT NULL,
                precision TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                source TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                PRIMARY KEY(model,precision,failure_type)
            )
        """)

        logs = sorted(
            OLD.glob(
                "logs/qwen2_5_1_5b_fp16_*/stderr.txt"
            )
        )

        evidence = []

        for log in logs:
            raw = log.read_text(errors="replace")

            if "Non-finite NLL at window 0" in raw:
                evidence.append({
                    "relative_log": str(
                        log.relative_to(OLD)
                    ),
                    "sha256": sha(log),
                })

        assert evidence, (
            "Original FP16 failure log not found"
        )

        dest.execute(
            """
            INSERT INTO text_quality_failure_evidence
            VALUES (?,?,?,?,?,?)
            """,
            (
                "qwen2_5_1_5b",
                "fp16",
                "nonfinite_nll_window0",
                json.dumps(evidence),
                str(OLD),
                utc(),
            )
        )

        dest.execute(
            """
            UPDATE text_quality_jobs
            SET status='numerical_invalid',
                updated_utc=?
            WHERE model='qwen2_5_1_5b'
              AND precision='fp16'
            """,
            (utc(),)
        )

        dest.commit()

        assert dest.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

    finally:
        src.close()
        dest.close()

    os.replace(temporary, DB)

def job_state(model, precision):
    db = connect()
    try:
        state = db.execute(
            """
            SELECT status,expected_windows
            FROM text_quality_jobs
            WHERE model=? AND precision=?
            """,
            (model, precision)
        ).fetchone()

        count = db.execute(
            """
            SELECT COUNT(*)
            FROM text_quality_windows
            WHERE model=? AND precision=?
            """,
            (model, precision)
        ).fetchone()[0]

        return state, count
    finally:
        db.close()

def record_failure(model, precision, kind, log):
    db = connect()
    try:
        with db:
            db.execute(
                """
                UPDATE text_quality_jobs
                SET status=?,updated_utc=?
                WHERE model=? AND precision=?
                """,
                (kind, utc(), model, precision)
            )

            db.execute(
                """
                INSERT OR REPLACE INTO
                text_quality_failure_evidence
                VALUES (?,?,?,?,?,?)
                """,
                (
                    model,
                    precision,
                    kind,
                    json.dumps({
                        "log": str(log.relative_to(ROOT)),
                        "sha256": sha(log),
                    }),
                    "T22_2R",
                    utc(),
                )
            )
    finally:
        db.close()

def export_csv(path, records):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)

def agreements(db, model, comparison):
    def read(precision):
        return {
            index: (input_sha, blob)
            for index, input_sha, blob in db.execute(
                """
                SELECT window_index,input_sha256,
                       predicted_token_ids_zlib
                FROM text_quality_windows
                WHERE model=? AND precision=?
                """,
                (model, precision)
            )
        }

    baseline = read("fp32")
    candidate = read(comparison)

    matching = set(baseline) & set(candidate)
    equal = 0
    total = 0

    for index in sorted(matching):
        source_sha, source_blob = baseline[index]
        other_sha, other_blob = candidate[index]

        assert source_sha == other_sha

        a = zlib.decompress(source_blob)
        b = zlib.decompress(other_blob)

        assert len(a) == len(b)
        assert len(a) % 4 == 0

        total += len(a) // 4

        equal += sum(
            a[i:i+4] == b[i:i+4]
            for i in range(0, len(a), 4)
        )

    return {
        "model": model,
        "baseline": "fp32",
        "comparison": comparison,
        "matched_tokens": total,
        "agreement": equal / total if total else None,
    }

def report():
    db = connect()
    try:
        assert db.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        rows = []

        for model, precision in EXPECTED:
            state, count = job_state(model, precision)

            n, tokens, total_nll, correct = db.execute(
                """
                SELECT COUNT(*),
                       COALESCE(SUM(tokens_scored),0),
                       COALESCE(SUM(nll_sum),0),
                       COALESCE(SUM(correct_next_token),0)
                FROM text_quality_windows
                WHERE model=? AND precision=?
                """,
                (model, precision)
            ).fetchone()

            assert n == count

            status = state[0] if state else "planned"
            expected = state[1] if state else None

            if status == "completed":
                assert expected == count

            mean_nll = (
                total_nll / tokens if tokens else None
            )

            rows.append({
                "model": model,
                "precision": precision,
                "status": status,
                "completed_windows": count,
                "expected_windows": expected,
                "scored_tokens": tokens,
                "mean_nll": mean_nll,
                "perplexity_at_256": (
                    __import__("math").exp(mean_nll)
                    if mean_nll is not None
                    and mean_nll < 700 else None
                ),
                "next_token_accuracy": (
                    correct / tokens if tokens else None
                ),
            })

        comparison_rows = [
            agreements(db, model, precision)
            for model in (
                "distilgpt2",
                "qwen2_5_1_5b",
                "qwen3_1_7b",
            )
            for precision in ("fp16", "bf16")
        ]

        for table, count in (
            ("tasks", 327336),
            ("energy_measurements", 18),
            ("pipeline_measurements", 24),
            ("vision_quality_predictions", 60000),
        ):
            assert db.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == count

    finally:
        db.close()

    export_csv(ROOT / "text_quality_summary.csv", rows)
    export_csv(
        ROOT / "precision_agreement.csv",
        comparison_rows
    )

    finished = all(
        row["status"] in (
            "completed", "numerical_invalid"
        )
        for row in rows
    )

    save(ROOT / "recovery_report.json", {
        "protocol":
            "T22_2R_WikiText2_PPL256_recovery_v1",
        "source": str(OLD),
        "finished": finished,
        "measured_jobs": sum(
            row["status"] == "completed"
            for row in rows
        ),
        "numerically_invalid_jobs": sum(
            row["status"] == "numerical_invalid"
            for row in rows
        ),
        "results": rows,
        "precision_agreement": comparison_rows,
        "old_database_modified": False,
        "all_precision_labels_distinct": True,
    })

    print("\n=== T22.2R TEXT QUALITY RESULTS ===")
    for row in rows:
        ppl = row["perplexity_at_256"]
        print(
            f"{row['model']:16s} "
            f"{row['precision']:4s} "
            f"{row['status']:18s} "
            f"windows={row['completed_windows']} "
            f"PPL={ppl}",
            flush=True,
        )

    for row in comparison_rows:
        print(
            "AGREEMENT",
            row["model"],
            f"fp32->{row['comparison']}",
            row["agreement"],
        )

    print(
        "MEASURED JOBS:",
        sum(r["status"] == "completed" for r in rows),
        "/9",
    )
    print(
        "NUMERICALLY INVALID JOBS:",
        sum(
            r["status"] == "numerical_invalid"
            for r in rows
        ),
    )
    print("OLD DATABASE MODIFIED: NO")
    print(
        "T22.2R CAMPAIGN:",
        "FINISHED_WITH_RECORDED_INVALID_RESULTS"
        if finished else "INCOMPLETE_RESUMABLE"
    )
    print("NEW SQLITE:", DB)
    print("RESULT DIRECTORY:", ROOT)

def main():
    with (ROOT / ".runner.lock").open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )

        disk_guard()
        assert OLD_SCRIPT.is_file()
        assert OLD_DB.is_file()

        create_worker()
        copy_corpus()
        prepare_database()

        plan = {
            "protocol":
                "T22_2R_WikiText2_PPL256_recovery_v1",
            "original_script_sha256": sha(OLD_SCRIPT),
            "worker_sha256": sha(WORKER),
            "corpus_sha256": sha(CORPUS),
            "cases": [list(c) for c in CASES],
            "source": str(OLD),
        }

        plan_path = ROOT / "recovery_plan_fixed.json"

        if plan_path.exists():
            assert json.loads(
                plan_path.read_text()
            ) == plan, (
                "Recovery plan changed; refusing resume."
            )
        else:
            save(plan_path, plan)

        stop = {"requested": False, "child": None}

        def interrupt(_sig, _frame):
            stop["requested"] = True
            proc = stop["child"]
            if proc and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass

        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)

        print("=== T22.2R TEXT QUALITY RECOVERY ===")
        print("COMPLETED OLD JOBS: 3 PRESERVED")
        print("QWEN2.5 FP16: NUMERICAL INVALID PRESERVED")
        print("ADDITIONAL PRECISION: BF16")
        print("NEW JOBS MAXIMUM: 5")
        print("AUTO-DOWNLOAD: NO")
        print("OLD SQLITE: READ-ONLY")
        print("RESULT DIRECTORY:", ROOT, flush=True)

        for model, precision in CASES:
            if stop["requested"]:
                break

            state, count = job_state(model, precision)

            if (
                state is not None
                and state[0] == "completed"
                and count == state[1]
            ):
                print(
                    "SKIP VERIFIED:",
                    model, precision,
                    flush=True
                )
                continue

            disk_guard()

            folder = ROOT / "logs" / (
                f"{model}_{precision}_{uuid.uuid4().hex}"
            )
            folder.mkdir(parents=True)

            error_path = folder / "stderr.txt"

            print(
                "\nRUN:", model, precision,
                "previous_windows=", count,
                flush=True
            )

            with error_path.open("w") as err:
                proc = subprocess.Popen(
                    [
                        sys.executable, "-u",
                        str(WORKER),
                        "worker", model, precision
                    ],
                    stdout=subprocess.PIPE,
                    stderr=err,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "TOKENIZERS_PARALLELISM": "false",
                    },
                )

                stop["child"] = proc

                for line in proc.stdout:
                    print(line, end="", flush=True)

                code = proc.wait()
                stop["child"] = None

            if code == 0:
                status, completed = job_state(
                    model, precision
                )
                assert status[0] == "completed"
                assert completed == status[1]
                print(
                    "COMMITTED:", model, precision,
                    flush=True
                )
                continue

            error = error_path.read_text(
                errors="replace"
            )

            if stop["requested"] or code == 130:
                print(
                    "INTERRUPTED: completed windows retained",
                    flush=True
                )
                break

            if (
                code == 1
                and (
                    "Non-finite NLL at window" in error
                    or "NONFINITE_LOGITS at window" in error
                )
            ):
                record_failure(
                    model, precision,
                    "numerical_invalid",
                    error_path
                )

                print(
                    "NUMERICAL INVALID:",
                    model, precision,
                    "— preserved, moving to next precision",
                    flush=True
                )
                continue

            record_failure(
                model, precision,
                "failed_runtime",
                error_path
            )

            print(
                "RUNTIME FAILURE:",
                model, precision,
                f"exit={code}",
                flush=True
            )

            for line in error.splitlines()[-25:]:
                print(line)

            print(
                "STOPPED FOR INVESTIGATION; "
                "successful windows preserved.",
                flush=True
            )
            break

        report()

if __name__ == "__main__":
    if sys.argv[1:] != ["run"]:
        raise SystemExit(
            "Usage: python recover_text.py run"
        )

    main()
