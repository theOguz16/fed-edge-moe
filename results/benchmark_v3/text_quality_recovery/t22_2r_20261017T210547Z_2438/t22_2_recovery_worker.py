import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import time
import urllib.request
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()

SOURCE = BASE / (
    "quality_campaigns/"
    "t22_1_20261017T201950Z_2438/"
    "dataset_v3_with_quality.sqlite"
)
INV = BASE / (
    "model_integration/"
    "t22_20261017T201408Z_2438/"
    "model_inventory.json"
)
DB = ROOT / "dataset_v3_with_text_quality.sqlite"
DATA = ROOT / "wikitext2_raw_test.parquet"

DATA_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATA_SHA = (
    "5f1bea067869d04849c0f975a2b29c4ff47d867"
    "f484f5010ea5e861eab246d91"
)
DATA_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/"
    "resolve/" + DATA_REV +
    "/wikitext-2-raw-v1/test-00000-of-00001.parquet"
)

MODELS = (
    "distilgpt2",
    "qwen2_5_1_5b",
    "qwen3_1_7b",
)
PRECISIONS = ("fp32", "fp16", "bf16")
WINDOW = 256
MIN_DISK_GIB = 4

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(4 * 1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def atomic(path, content):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def save(path, obj):
    atomic(
        path,
        (json.dumps(
            obj, indent=2, ensure_ascii=False
        ) + "\n").encode()
    )

def disk_guard():
    if shutil.disk_usage(ROOT).free < MIN_DISK_GIB * 1024**3:
        raise RuntimeError("DISK GUARD: under 4 GiB free")

def connection(path=DB):
    db = sqlite3.connect(str(path), timeout=90)
    db.execute("PRAGMA busy_timeout=90000")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    return db

def download_corpus():
    if DATA.exists():
        assert sha(DATA) == DATA_SHA, (
            "WikiText source checksum mismatch"
        )
        return

    disk_guard()
    temp = DATA.with_suffix(".download")

    req = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "FedEdgeMoE-V3/1.0"}
    )
    with urllib.request.urlopen(req, timeout=90) as src:
        with temp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())

    assert sha(temp) == DATA_SHA, (
        "Downloaded corpus SHA256 mismatch"
    )
    os.replace(temp, DATA)

def corpus_text():
    import pyarrow.parquet as pq

    table = pq.read_table(DATA, columns=["text"])
    texts = table.column("text").to_pylist()

    assert len(texts) == 4358
    assert all(isinstance(t, str) for t in texts)

    # Deterministic row order and separator.
    return "\n".join(texts)

def choose_snapshot(model):
    inventory = json.loads(INV.read_text())

    record = next(
        entry
        for entry in inventory["text_checkpoints"]
        if entry["model"] == model
    )

    expected_repo = {
        "distilgpt2": "models--distilgpt2",
        "qwen2_5_1_5b":
            "models--Qwen--Qwen2.5-1.5B-Instruct",
        "qwen3_1_7b":
            "models--Qwen--Qwen3-1.7B",
    }[model]

    candidates = [
        entry
        for entry in record["candidate_snapshots"]
        if entry["repository_cache_name"] == expected_repo
        and entry["candidate_complete"]
    ]

    assert len(candidates) == 1, (
        f"Unambiguous local snapshot missing: {model}"
    )

    entry = candidates[0]
    snapshot = Path(entry["snapshot_path"])

    assert snapshot.is_dir()
    assert sha(snapshot / "config.json") == (
        entry["config_sha256"]
    )

    return snapshot, entry

def weight_fingerprint(snapshot):
    files = sorted(
        p for p in snapshot.iterdir()
        if p.is_file()
        and p.name.endswith((".safetensors", ".bin"))
    )

    assert files, "No local weight files"

    # Refuse a partially cached sharded checkpoint.
    for index in snapshot.glob("*.index.json"):
        info = json.loads(index.read_text())
        expected = set(
            info.get("weight_map", {}).values()
        )
        assert expected
        assert all(
            (snapshot / name).is_file()
            for name in expected
        ), "Missing weight shard"

    checks = {
        p.name: sha(p)
        for p in files
    }

    return hashlib.sha256(
        json.dumps(
            checks, sort_keys=True
        ).encode()
    ).hexdigest(), checks

def validate_source():
    assert SOURCE.is_file()
    assert INV.is_file()

    c = sqlite3.connect(
        f"file:{SOURCE}?mode=ro", uri=True
    )
    try:
        assert c.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        expected = {
            "tasks": 327336,
            "energy_measurements": 18,
            "pipeline_measurements": 24,
            "vision_quality_predictions": 60000,
        }

        for table, count in expected.items():
            assert c.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == count

        assert c.execute(
            "SELECT COUNT(*) FROM vision_quality_jobs "
            "WHERE status='completed'"
        ).fetchone()[0] == 6
    finally:
        c.close()

def initialize():
    import torch
    import transformers

    disk_guard()
    validate_source()
    download_corpus()

    assert torch.backends.mps.is_available()

    for name in MODELS:
        choose_snapshot(name)

    plan = {
        "protocol": "T22_2_WikiText2_raw_PPL256_v1",
        "source_sha256": sha(SOURCE),
        "inventory_sha256": sha(INV),
        "script_sha256": sha(Path(__file__)),
        "data_sha256": DATA_SHA,
        "data_revision": DATA_REV,
        "models": list(MODELS),
        "precisions": list(PRECISIONS),
        "window_size": WINDOW,
        "window_stride": WINDOW,
        "batch_size": 1,
        "device": "mps",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "use_chat_template": False,
        "model_downloads": False,
    }

    plan_path = ROOT / "campaign_plan.json"

    if plan_path.exists():
        assert json.loads(
            plan_path.read_text()
        ) == plan, "Source or environment changed on resume"
    else:
        save(plan_path, plan)

    if DB.exists():
        db = connection()
        try:
            assert db.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"
            assert dict(db.execute(
                "SELECT key,value FROM text_quality_meta"
            )) == {
                "protocol": plan["protocol"],
                "data_sha256": DATA_SHA,
                "source_sha256": plan["source_sha256"],
            }
        finally:
            db.close()
        return

    temp = ROOT / (
        "build_" + uuid.uuid4().hex + ".sqlite"
    )

    original = sqlite3.connect(
        f"file:{SOURCE}?mode=ro", uri=True
    )
    dest = sqlite3.connect(str(temp))

    try:
        original.backup(dest)

        dest.executescript("""
            CREATE TABLE text_quality_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE text_quality_jobs (
                model TEXT NOT NULL,
                precision TEXT NOT NULL,
                status TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                weight_fingerprint TEXT NOT NULL,
                tokenizer_fingerprint TEXT NOT NULL,
                token_stream_sha256 TEXT NOT NULL,
                total_input_tokens INTEGER NOT NULL,
                expected_windows INTEGER NOT NULL,
                updated_utc TEXT NOT NULL,
                PRIMARY KEY(model,precision)
            );

            CREATE TABLE text_quality_windows (
                model TEXT NOT NULL,
                precision TEXT NOT NULL,
                window_index INTEGER NOT NULL,
                input_sha256 TEXT NOT NULL,
                tokens_scored INTEGER NOT NULL,
                nll_sum REAL NOT NULL,
                correct_next_token INTEGER NOT NULL,
                predicted_token_ids_zlib BLOB NOT NULL,
                PRIMARY KEY(model,precision,window_index),
                FOREIGN KEY(model,precision)
                  REFERENCES text_quality_jobs(model,precision)
            );
        """)

        dest.executemany(
            "INSERT INTO text_quality_meta VALUES (?,?)",
            [
                ("protocol", plan["protocol"]),
                ("data_sha256", DATA_SHA),
                ("source_sha256", plan["source_sha256"]),
            ]
        )

        dest.commit()
        assert dest.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
    finally:
        dest.close()
        original.close()

    os.replace(temp, DB)

def encode_ids(ids):
    return struct.pack("<" + "I" * len(ids), *ids)

def windows_of(ids):
    return [
        ids[i:i + WINDOW]
        for i in range(0, len(ids), WINDOW)
        if len(ids[i:i + WINDOW]) >= 2
    ]

def run_worker(name, precision):
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer
    )

    assert name in MODELS
    assert precision in PRECISIONS
    assert torch.backends.mps.is_available()
    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"

    interrupted = {"value": False}

    def stop(_sig, _frame):
        interrupted["value"] = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    snapshot, entry = choose_snapshot(name)
    checkpoint_fingerprint, weight_checks = (
        weight_fingerprint(snapshot)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    tokenizer_files = sorted(
        p for p in snapshot.iterdir()
        if p.is_file()
        and (
            "tokenizer" in p.name
            or p.name in (
                "vocab.json", "merges.txt",
                "special_tokens_map.json"
            )
        )
    )

    tokenizer_fingerprint = hashlib.sha256(
        json.dumps({
            p.name: sha(p)
            for p in tokenizer_files
        }, sort_keys=True).encode()
    ).hexdigest()

    ids = tokenizer.encode(
        corpus_text(),
        add_special_tokens=False
    )

    assert len(ids) > 10000
    assert min(ids) >= 0

    stream = encode_ids(ids)
    stream_sha = hashlib.sha256(stream).hexdigest()

    token_file = ROOT / f"{name}_token_ids.zlib"
    compressed = zlib.compress(stream, 6)

    if token_file.exists():
        assert zlib.decompress(
            token_file.read_bytes()
        ) == stream
    else:
        atomic(token_file, compressed)

    slices = windows_of(ids)
    assert len(slices) > 20

    db = connection()

    try:
        previous = db.execute(
            """
            SELECT snapshot_path,weight_fingerprint,
                   tokenizer_fingerprint,
                   token_stream_sha256,
                   total_input_tokens,expected_windows
            FROM text_quality_jobs
            WHERE model=? AND precision=?
            """,
            (name, precision),
        ).fetchone()

        identity = (
            str(snapshot),
            checkpoint_fingerprint,
            tokenizer_fingerprint,
            stream_sha,
            len(ids),
            len(slices),
        )

        if previous is not None:
            assert tuple(previous) == identity, (
                "Checkpoint/tokenizer changed; refusing "
                "to mix incompatible observations."
            )

        with db:
            db.execute(
                """
                INSERT INTO text_quality_jobs VALUES
                (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model,precision)
                DO UPDATE SET
                    status='running',
                    updated_utc=excluded.updated_utc
                """,
                (
                    name, precision, "running",
                    *identity, now()
                ),
            )

        completed = set(
            row[0] for row in db.execute(
                """
                SELECT window_index
                FROM text_quality_windows
                WHERE model=? AND precision=?
                """,
                (name, precision)
            )
        )

        print(
            f"WORKER {name}/{precision}: "
            f"tokens={len(ids)} "
            f"windows={len(slices)} "
            f"done={len(completed)}",
            flush=True,
        )

        if len(completed) < len(slices):
            dtype = {
                "fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
            }[precision]

            model = AutoModelForCausalLM.from_pretrained(
                str(snapshot),
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                use_safetensors=any(
                    p.name.endswith(".safetensors")
                    for p in snapshot.iterdir()
                ),
            )

            model = model.to(
                device="mps", dtype=dtype
            ).eval()

            for index, token_slice in enumerate(slices):
                if index in completed:
                    continue

                if interrupted["value"]:
                    with db:
                        db.execute(
                            """
                            UPDATE text_quality_jobs
                            SET status='interrupted',
                                updated_utc=?
                            WHERE model=? AND precision=?
                            """,
                            (now(), name, precision)
                        )
                    print("INTERRUPTED; committed windows saved")
                    return 130

                if index % 25 == 0:
                    disk_guard()

                token_bytes = encode_ids(token_slice)
                token_hash = hashlib.sha256(
                    token_bytes
                ).hexdigest()

                x = torch.tensor(
                    [token_slice],
                    dtype=torch.long,
                    device="mps",
                )

                with torch.inference_mode():
                    logits = model(
                        input_ids=x,
                        use_cache=False
                    ).logits[0, :-1, :]

                    labels = x[0, 1:]
                    scores = logits.float()

                    if not bool(torch.isfinite(scores).all().item()):
                        raise RuntimeError(
                            f"NONFINITE_LOGITS at window {index}"
                        )

                    nll_sum = float(
                        F.cross_entropy(
                            scores,
                            labels,
                            reduction="sum"
                        ).item()
                    )

                    predictions = (
                        scores.argmax(dim=-1)
                        .cpu().tolist()
                    )

                if not math.isfinite(nll_sum):
                    raise RuntimeError(
                        f"Non-finite NLL at window {index}"
                    )

                assert len(predictions) == (
                    len(token_slice) - 1
                )

                correct = sum(
                    int(p == t)
                    for p, t in zip(
                        predictions,
                        token_slice[1:]
                    )
                )

                pred_blob = zlib.compress(
                    encode_ids(predictions), 6
                )

                with db:
                    db.execute(
                        """
                        INSERT INTO text_quality_windows
                        VALUES (?,?,?,?,?,?,?,?)
                        """,
                        (
                            name, precision, index,
                            token_hash,
                            len(predictions),
                            nll_sum, correct,
                            pred_blob,
                        ),
                    )

                if (index + 1) % 100 == 0:
                    print(
                        f"PROGRESS {name}/{precision}: "
                        f"{index+1}/{len(slices)}",
                        flush=True,
                    )

                del x, logits, scores

        n, token_count, nll_total, correct_total = (
            db.execute(
                """
                SELECT COUNT(*),
                       COALESCE(SUM(tokens_scored),0),
                       COALESCE(SUM(nll_sum),0),
                       COALESCE(SUM(correct_next_token),0)
                FROM text_quality_windows
                WHERE model=? AND precision=?
                """,
                (name, precision),
            ).fetchone()
        )

        assert n == len(slices)
        assert token_count == sum(
            len(item)-1 for item in slices
        )
        assert token_count > 0

        ppl = math.exp(nll_total / token_count)

        with db:
            db.execute(
                """
                UPDATE text_quality_jobs
                SET status='completed',updated_utc=?
                WHERE model=? AND precision=?
                """,
                (now(), name, precision)
            )

        print(
            f"COMPLETED {name}/{precision}: "
            f"scored_tokens={token_count} "
            f"NLL={nll_total/token_count:.5f} "
            f"PPL@256={ppl:.4f} "
            f"next_token_acc={correct_total/token_count:.4f}",
            flush=True,
        )

        return 0
    finally:
        db.close()

def mark_failure(name, precision):
    db = connection()
    try:
        with db:
            db.execute(
                """
                UPDATE text_quality_jobs
                SET status='failed',updated_utc=?
                WHERE model=? AND precision=?
                  AND status='running'
                """,
                (now(), name, precision)
            )
    finally:
        db.close()

def produce_report():
    db = sqlite3.connect(
        f"file:{DB}?mode=ro", uri=True
    )

    try:
        assert db.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        results = []

        for model in MODELS:
            for precision in PRECISIONS:
                meta = db.execute(
                    """
                    SELECT status,expected_windows
                    FROM text_quality_jobs
                    WHERE model=? AND precision=?
                    """,
                    (model, precision)
                ).fetchone()

                n, tokens, nll, correct = db.execute(
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

                results.append({
                    "model": model,
                    "precision": precision,
                    "status": meta[0] if meta else "planned",
                    "windows_completed": n,
                    "windows_total": meta[1] if meta else None,
                    "scored_tokens": tokens,
                    "mean_nll": (
                        nll/tokens if tokens else None
                    ),
                    "perplexity_at_256": (
                        math.exp(nll/tokens)
                        if tokens and nll/tokens < 700
                        else None
                    ),
                    "next_token_accuracy": (
                        correct/tokens if tokens else None
                    ),
                })

        agreements = []

        for model in MODELS:
            fp32 = {
                row[0]: (row[1], row[2])
                for row in db.execute(
                    """
                    SELECT window_index,input_sha256,
                           predicted_token_ids_zlib
                    FROM text_quality_windows
                    WHERE model=? AND precision='fp32'
                    """,
                    (model,)
                )
            }

            fp16 = {
                row[0]: (row[1], row[2])
                for row in db.execute(
                    """
                    SELECT window_index,input_sha256,
                           predicted_token_ids_zlib
                    FROM text_quality_windows
                    WHERE model=? AND precision='fp16'
                    """,
                    (model,)
                )
            }

            equal = total = 0

            for idx in sorted(set(fp32) & set(fp16)):
                hash_a, blob_a = fp32[idx]
                hash_b, blob_b = fp16[idx]

                assert hash_a == hash_b

                data_a = zlib.decompress(blob_a)
                data_b = zlib.decompress(blob_b)

                assert len(data_a) == len(data_b)
                assert len(data_a) % 4 == 0

                total += len(data_a) // 4
                equal += sum(
                    data_a[i:i+4] == data_b[i:i+4]
                    for i in range(0, len(data_a), 4)
                )

            agreements.append({
                "model": model,
                "matched_predicted_tokens": total,
                "same_token_predictions": equal,
                "fp32_fp16_token_agreement": (
                    equal/total if total else None
                ),
            })

        for table, expected in (
            ("tasks", 327336),
            ("energy_measurements", 18),
            ("pipeline_measurements", 24),
            ("vision_quality_predictions", 60000),
        ):
            assert db.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == expected

    finally:
        db.close()

    def export(filename, rows):
        path = ROOT / filename
        temp = path.with_name(path.name + ".tmp")
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0])
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)

    export("text_quality_summary.csv", results)
    export("precision_agreement.csv", agreements)

    complete = all(
        row["status"] == "completed"
        and row["windows_completed"] == row["windows_total"]
        for row in results
    )

    save(ROOT / "report.json", {
        "protocol": "T22_2_WikiText2_raw_PPL256_v1",
        "complete": complete,
        "results": results,
        "agreements": agreements,
        "corpus_sha256": DATA_SHA,
        "evaluation": "full_model_MPS_teacher_forced",
        "context_tokens": WINDOW,
        "generation_quality_measured": False,
        "kl_measured": False,
        "selfcheck_nli_measured": False,
        "no_new_vision_or_pipeline_tasks_claimed": True,
    })

    print("\n=== T22.2 TEXT QUALITY STATUS ===")
    for row in results:
        print(
            row["model"], row["precision"],
            row["status"],
            f"{row['windows_completed']}/"
            f"{row['windows_total']}",
            "PPL=", row["perplexity_at_256"]
        )

    for row in agreements:
        print(
            "AGREEMENT", row["model"],
            row["fp32_fp16_token_agreement"]
        )

    print("COMPLETE JOBS:", sum(
        x["status"] == "completed" for x in results
    ), "/6")
    print("SOURCE DATABASE MODIFIED: NO")
    print(
        "T22.2 TEXT QUALITY CAMPAIGN:",
        "PASS" if complete else "INCOMPLETE — RESUMABLE"
    )
    print("NEW SQLITE:", DB)
    print("RESULT DIRECTORY:", ROOT)

def coordinator():
    lock_path = ROOT / ".runner.lock"

    with lock_path.open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )

        initialize()

        stop = {"requested": False, "child": None}

        def interrupt(_sig, _frame):
            stop["requested"] = True
            child = stop["child"]
            if child and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass

        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)

        print("=== T22.2 FULL TEXT QUALITY CAMPAIGN ===")
        print("MODELS: 3")
        print("PRECISIONS: FP32, FP16")
        print("CORPUS: WikiText-2 raw test, all 4358 rows")
        print("CONTEXT: 256 tokens")
        print("DEVICE: Full MPS")
        print("WEIGHT DOWNLOADS: DISABLED")
        print("SQLITE: DURABLE WINDOW-BY-WINDOW")
        print("CTRL+C: STOP AND RESUME")
        print("RESULT DIRECTORY:", ROOT, flush=True)

        for model in MODELS:
            for precision in PRECISIONS:
                if stop["requested"]:
                    break

                disk_guard()
                with connection() as db:
                    state = db.execute(
                        """
                        SELECT status,expected_windows
                        FROM text_quality_jobs
                        WHERE model=? AND precision=?
                        """,
                        (model, precision),
                    ).fetchone()

                    count = db.execute(
                        """
                        SELECT COUNT(*)
                        FROM text_quality_windows
                        WHERE model=? AND precision=?
                        """,
                        (model, precision),
                    ).fetchone()[0]

                if (
                    state
                    and state[0] == "completed"
                    and count == state[1]
                ):
                    print(
                        "SKIP VERIFIED:", model, precision,
                        flush=True
                    )
                    continue

                attempt = ROOT / "logs" / (
                    model + "_" + precision
                    + "_" + uuid.uuid4().hex
                )
                attempt.mkdir(parents=True)

                with (attempt / "stderr.txt").open("w") as err:
                    proc = subprocess.Popen(
                        [
                            sys.executable, "-u",
                            str(Path(__file__).resolve()),
                            "worker", model, precision
                        ],
                        stderr=err,
                        env={
                            **os.environ,
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                            "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                        },
                        start_new_session=True,
                    )
                    stop["child"] = proc
                    code = proc.wait()
                    stop["child"] = None

                if code != 0:
                    mark_failure(model, precision)

                    print(
                        f"WORKER FAILED: {model}/{precision} "
                        f"exit={code}",
                        flush=True,
                    )

                    lines = (
                        attempt / "stderr.txt"
                    ).read_text(errors="replace").splitlines()

                    for line in lines[-20:]:
                        print(line)

                    print(
                        "Saved progress preserved; "
                        "stopping for diagnosis."
                    )
                    stop["requested"] = True
                    break

            if stop["requested"]:
                break

        produce_report()

if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        coordinator()
    elif (
        len(sys.argv) == 4
        and sys.argv[1] == "worker"
    ):
        raise SystemExit(run_worker(
            sys.argv[2], sys.argv[3]
        ))
    else:
        raise SystemExit(
            "Usage: python t22_2.py run"
        )
