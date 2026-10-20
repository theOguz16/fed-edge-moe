import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
import re
import resource
import shutil
import signal
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
DB = BASE / (
    "production_campaigns/t23_mac_v1/"
    "dataset_v3_t23_mac.sqlite"
)
CORPUS = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)
HUB = Path.home() / ".cache/huggingface/hub"
SNAPSHOTS = {
    "distilgpt2": (
        "models--distilgpt2",
        "2290a62682d06624634c1f46a6ad5be0f47f38aa"
    ),
    "qwen2_5_1_5b": (
        "models--Qwen--Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    ),
    "qwen3_1_7b": (
        "models--Qwen--Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    ),
}
PRECISIONS = ("fp32", "fp16", "bf16")
PLACEMENTS = (
    "full_cpu", "full_mps",
    "cpu_to_mps", "mps_to_cpu"
)
QOS = {
    "background": 0x09,
    "utility": 0x11,
    "default": 0x15,
    "user_initiated": 0x19,
}
BATCHES = (1, 2)
THREADS = 4
CONTEXT = 128
WARMUPS = 1
MIN_TIMED_CALLS = 3
TARGET_SECONDS = 8
CASE_TIMEOUT = 900
PROTOCOL = "T25_MAC_TEXT_PREFILL_V1"

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(4 * 1024 * 1024), b""
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

def snapshot(model):
    repo, revision = SNAPSHOTS[model]
    path = HUB / repo / "snapshots" / revision
    if not (path / "config.json").is_file():
        raise RuntimeError(
            "LOCAL_MODEL_MISSING " + str(path)
        )
    return path

def disk_guard():
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < 4:
        raise RuntimeError(
            f"DISK_GUARD free={free:.2f} GiB"
        )
    return free

def connect():
    db = sqlite3.connect(
        str(DB), timeout=90, isolation_level=None
    )
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=90000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    return db

def provenance():
    path = ROOT / "model_provenance.json"
    if path.is_file():
        data = json.loads(path.read_text())
        assert data["protocol"] == PROTOCOL
        assert data["corpus_sha256"] == sha(CORPUS)
        for name in SNAPSHOTS:
            spec = data["models"][name]
            assert spec["snapshot"] == str(
                snapshot(name)
            )
            for file in spec["weight_files"]:
                target = Path(file["path"])
                assert target.is_file()
                assert target.stat().st_size == file["size"]
        return data

    data = {
        "protocol": PROTOCOL,
        "corpus": str(CORPUS),
        "corpus_sha256": sha(CORPUS),
        "models": {},
    }

    for name in SNAPSHOTS:
        folder = snapshot(name)
        files = sorted(
            list(folder.glob("*.safetensors"))
            + list(folder.glob("pytorch_model*.bin"))
        )
        assert files, name

        data["models"][name] = {
            "snapshot": str(folder),
            "config_sha256": sha(
                folder / "config.json"
            ),
            "weight_files": [
                {
                    "path": str(p),
                    "size": p.stat().st_size,
                    "sha256": sha(p),
                }
                for p in files
            ],
        }

    save(path, data)
    return data

def prepare():
    disk_guard()
    assert DB.is_file()
    assert CORPUS.is_file()

    source = provenance()
    db = connect()

    try:
        assert db.execute(
            "PRAGMA quick_check"
        ).fetchone()[0] == "ok"
        assert db.execute(
            "SELECT COUNT(*) FROM t23_measurements"
        ).fetchone()[0] == 90
        assert db.execute(
            "SELECT COUNT(*) FROM t24r_measurements"
        ).fetchone()[0] == 203
        assert db.execute(
            "SELECT COUNT(*) FROM t24_1_results"
        ).fetchone()[0] == 13

        db.execute("""
            CREATE TABLE IF NOT EXISTS t25_cases (
                case_id TEXT PRIMARY KEY,
                ordinal INTEGER UNIQUE NOT NULL,
                model TEXT NOT NULL,
                precision TEXT NOT NULL,
                placement TEXT NOT NULL,
                qos TEXT NOT NULL,
                batch_size INTEGER NOT NULL,
                context_tokens INTEGER NOT NULL,
                cpu_threads INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'planned',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                claim_token TEXT,
                result_path TEXT,
                result_sha256 TEXT,
                last_error TEXT
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS t25_attempts (
                attempt_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                started_utc TEXT NOT NULL,
                finished_utc TEXT,
                status TEXT NOT NULL,
                exit_code INTEGER,
                message TEXT,
                FOREIGN KEY(case_id)
                    REFERENCES t25_cases(case_id)
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS t25_results (
                case_id TEXT PRIMARY KEY,
                numerical_allclose INTEGER NOT NULL,
                token_agreement REAL NOT NULL,
                max_abs_logit_error REAL NOT NULL,
                numerical_status TEXT NOT NULL,
                prefill_tokens_s REAL NOT NULL,
                requests_s REAL NOT NULL,
                latency_median_ms REAL NOT NULL,
                latency_p95_ms REAL NOT NULL,
                elapsed_s REAL NOT NULL,
                cpu_power_w REAL NOT NULL,
                gpu_power_w REAL NOT NULL,
                gross_energy_j_token REAL NOT NULL,
                power_sample_count INTEGER NOT NULL,
                rss_peak_bytes INTEGER,
                mps_allocated_bytes INTEGER,
                layer_count INTEGER NOT NULL,
                cut_layer INTEGER,
                result_path TEXT NOT NULL,
                result_sha256 TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                FOREIGN KEY(case_id)
                    REFERENCES t25_cases(case_id)
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS t25_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        manifest_sha = sha(
            ROOT / "model_provenance.json"
        )
        old = db.execute(
            "SELECT value FROM t25_metadata "
            "WHERE key='source_manifest_sha256'"
        ).fetchone()

        if old is not None:
            assert old[0] == manifest_sha
        else:
            db.execute(
                "INSERT INTO t25_metadata VALUES (?,?)",
                ("source_manifest_sha256", manifest_sha)
            )

        count = db.execute(
            "SELECT COUNT(*) FROM t25_cases"
        ).fetchone()[0]

        if count:
            assert count == 288
            return

        records = []
        for qos in QOS:
            for batch in BATCHES:
                for precision in PRECISIONS:
                    for placement in PLACEMENTS:
                        for model in SNAPSHOTS:
                            ident = {
                                "protocol": PROTOCOL,
                                "model": model,
                                "precision": precision,
                                "placement": placement,
                                "qos": qos,
                                "batch": batch,
                                "context": CONTEXT,
                                "threads": THREADS,
                            }
                            case_id = "t25_" + hashlib.sha256(
                                json.dumps(
                                    ident, sort_keys=True
                                ).encode()
                            ).hexdigest()

                            records.append((
                                case_id, len(records), model,
                                precision, placement, qos,
                                batch, CONTEXT, THREADS
                            ))

        assert len(records) == 288

        db.execute("BEGIN IMMEDIATE")
        try:
            db.executemany("""
                INSERT INTO t25_cases (
                    case_id,ordinal,model,precision,
                    placement,qos,batch_size,
                    context_tokens,cpu_threads
                ) VALUES (?,?,?,?,?,?,?,?,?)
            """, records)
            db.commit()
        except BaseException:
            db.rollback()
            raise

        print(
            "T25 CASES PLANNED:", len(records),
            flush=True
        )
    finally:
        db.close()

def set_qos(name):
    lib = ctypes.CDLL(
        "/usr/lib/libSystem.B.dylib"
    )
    setter = lib.pthread_set_qos_class_self_np
    getter = lib.qos_class_self
    setter.argtypes = [
        ctypes.c_uint, ctypes.c_int
    ]
    setter.restype = ctypes.c_int
    getter.restype = ctypes.c_uint

    code = setter(QOS[name], 0)
    actual = getter()
    if code != 0 or actual != QOS[name]:
        raise RuntimeError(
            "QOS_VERIFICATION_FAILED "
            f"requested={name} actual={actual}"
        )

def power_pairs(text):
    cpu_matches = list(re.finditer(
        r"CPU Power:\s*([\d.]+)\s*mW",
        text, re.IGNORECASE
    ))
    pairs = []

    for index, match in enumerate(cpu_matches):
        end = (
            cpu_matches[index + 1].start()
            if index + 1 < len(cpu_matches)
            else len(text)
        )
        section = text[match.end():end]
        gpu = re.search(
            r"GPU Power:\s*([\d.]+)\s*mW",
            section, re.IGNORECASE
        )

        if gpu:
            a = float(match.group(1)) / 1000
            b = float(gpu.group(1)) / 1000
            if (
                math.isfinite(a)
                and math.isfinite(b)
                and a >= 0 and b >= 0
            ):
                pairs.append((a, b))

    return pairs

def worker(case, destination):
    import torch
    import pyarrow.parquet as pq
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )

    if os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) != "0":
        raise RuntimeError(
            "MPS CPU fallback must be disabled"
        )

    assert torch.backends.mps.is_available()

    set_qos(case["qos"])
    torch.set_num_threads(case["cpu_threads"])
    torch.manual_seed(2026)

    name = case["model"]
    location = case["placement"]
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[case["precision"]]

    folder = snapshot(name)

    tokenizer = AutoTokenizer.from_pretrained(
        str(folder),
        local_files_only=True,
        trust_remote_code=False,
    )

    # The fixed prefill workload comes from WikiText-2
    # raw test. It is not a text-generation workload.
    raw = pq.read_table(
        CORPUS, columns=["text"]
    )["text"].to_pylist()

    excerpt = "\n".join(
        line.strip()
        for line in raw
        if isinstance(line, str) and line.strip()
    )[:4000]

    tokens = tokenizer.encode(
        excerpt,
        add_special_tokens=False
    )
    assert len(tokens) >= CONTEXT * 2, (
        "Insufficient fixed-context tokens"
    )

    sequences = [
        tokens[0:CONTEXT],
        tokens[CONTEXT:CONTEXT*2],
    ]
    input_ids = torch.tensor(
        sequences[:case["batch_size"]],
        dtype=torch.long,
    )

    token_sha = hashlib.sha256(
        input_ids.numpy().tobytes()
    ).hexdigest()

    model = AutoModelForCausalLM.from_pretrained(
        str(folder),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()

    # Reference from the same checkpoint converted to
    # FP32. This is not an original FP32 checkpoint claim.
    with torch.inference_mode():
        baseline = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).logits.float().detach().cpu()

    assert torch.isfinite(baseline).all().item()

    if name == "distilgpt2":
        blocks = list(model.transformer.h)
        embedding_modules = [
            model.transformer.wte,
            model.transformer.wpe,
        ]
        output_modules = [
            model.transformer.ln_f,
            model.lm_head,
        ]
    else:
        blocks = list(model.model.layers)
        embedding_modules = [
            model.model.embed_tokens,
            model.model.rotary_emb,
        ]
        output_modules = [
            model.model.norm,
            model.lm_head,
        ]

    n_layers = len(blocks)
    assert n_layers >= 2

    cut = n_layers // 2

    if location == "full_cpu":
        devices = ["cpu"] * n_layers
    elif location == "full_mps":
        devices = ["mps"] * n_layers
    elif location == "cpu_to_mps":
        devices = (
            ["cpu"] * cut
            + ["mps"] * (n_layers - cut)
        )
    elif location == "mps_to_cpu":
        devices = (
            ["mps"] * cut
            + ["cpu"] * (n_layers - cut)
        )
    else:
        raise ValueError(location)

    first_device = devices[0]
    final_device = devices[-1]

    # DistilGPT2 ties embedding and LM-head weights.
    # Separate the tied parameter when endpoints differ,
    # otherwise moving the head also moves the embedding.
    embeddings = model.get_input_embeddings()
    head = model.get_output_embeddings()

    if (
        first_device != final_device
        and embeddings.weight is head.weight
    ):
        head.weight = torch.nn.Parameter(
            head.weight.detach().clone(),
            requires_grad=False,
        )

    for module in embedding_modules:
        module.to(
            device=first_device, dtype=dtype
        )

    for module, device in zip(blocks, devices):
        module.to(device=device, dtype=dtype)

    for module in output_modules:
        module.to(
            device=final_device, dtype=dtype
        )

    for module, device in zip(blocks, devices):
        parameters = list(module.parameters())
        assert parameters
        assert all(
            p.device.type == device
            for p in parameters
        )

    assert embeddings.weight.device.type == first_device
    assert head.weight.device.type == final_device

    # Preserve Transformers' own forward implementation.
    # Each decoder block receives hidden states, masks and
    # rotary embeddings on the block's actual device.
    move_cache = {}
    handles = []

    def move(value, destination):
        if isinstance(value, torch.Tensor):
            if value.device.type == destination:
                return value

            key = (id(value), destination)
            if key not in move_cache:
                move_cache[key] = value.to(destination)
            return move_cache[key]

        if isinstance(value, tuple):
            return tuple(
                move(x, destination) for x in value
            )
        if isinstance(value, list):
            return [
                move(x, destination) for x in value
            ]
        if isinstance(value, dict):
            return {
                k: move(v, destination)
                for k, v in value.items()
            }
        return value

    def register(block, device):
        def hook(_module, args, kwargs):
            return (
                move(args, device),
                move(kwargs, device),
            )
        return block.register_forward_pre_hook(
            hook, with_kwargs=True
        )

    for block, device in zip(blocks, devices):
        handles.append(register(block, device))

    if "mps" in devices:
        torch.mps.synchronize()

    run_inputs = input_ids.to(first_device)

    def forward():
        move_cache.clear()
        output = model(
            input_ids=run_inputs,
            use_cache=False,
            return_dict=True,
        ).logits

        if "mps" in devices:
            torch.mps.synchronize()

        return output.float().cpu()

    try:
        with torch.inference_mode():
            actual = forward()

        if not torch.isfinite(actual).all().item():
            raise RuntimeError(
                "NUMERICAL_INVALID nonfinite_logits"
            )

        max_error = float(
            (actual - baseline).abs().max().item()
        )
        token_agreement = float(
            (
                actual.argmax(dim=-1)
                == baseline.argmax(dim=-1)
            ).float().mean().item()
        )
        tolerance = (
            1e-3
            if case["precision"] == "fp32"
            else 5e-2
        )
        allclose = bool(torch.allclose(
            actual, baseline,
            atol=tolerance, rtol=tolerance
        ))

        numerical_status = (
            "numerical_pass"
            if allclose
            else "numerical_mismatch_measured"
        )

        del actual

        with torch.inference_mode():
            for _ in range(WARMUPS):
                forward()

        power_path = destination.with_name(
            "powermetrics_raw.txt"
        )

        samples_ms = []
        completed_calls = 0

        with power_path.open("w") as sensor_file:
            sensor = subprocess.Popen(
                [
                    "sudo", "-n", "powermetrics",
                    "--samplers", "cpu_power,gpu_power",
                    "-i", "1000", "-n", "2000",
                ],
                stdout=sensor_file,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )

            started = time.monotonic()
            try:
                with torch.inference_mode():
                    while (
                        time.monotonic() - started
                        < TARGET_SECONDS
                        or completed_calls < MIN_TIMED_CALLS
                    ):
                        t0 = time.perf_counter_ns()
                        forward()
                        elapsed_ns = (
                            time.perf_counter_ns() - t0
                        )
                        assert elapsed_ns > 0

                        samples_ms.append(
                            elapsed_ns / 1e6
                        )
                        completed_calls += 1
            finally:
                elapsed_s = (
                    time.monotonic() - started
                )

                if sensor.poll() is None:
                    sensor.send_signal(signal.SIGINT)

                try:
                    sensor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    sensor.kill()
                    sensor.wait()

        raw_power = power_path.read_text(
            errors="replace"
        )
        pairs = power_pairs(raw_power)

        if len(pairs) < 2:
            raise RuntimeError(
                "POWER_MISSING "
                f"pairs={len(pairs)} "
                f"sensor_exit={sensor.returncode} "
                f"tail={raw_power[-400:]!r}"
            )

        assert completed_calls >= MIN_TIMED_CALLS
        assert elapsed_s > 0

        cpu_w = statistics.mean(
            x[0] for x in pairs
        )
        gpu_w = statistics.mean(
            x[1] for x in pairs
        )
        processed_tokens = (
            completed_calls
            * case["batch_size"]
            * CONTEXT
        )

        tokens_s = processed_tokens / elapsed_s
        requests_s = (
            completed_calls * case["batch_size"]
        ) / elapsed_s

        ordered = sorted(samples_ms)
        pos = .95 * (len(ordered) - 1)
        low = int(pos)
        high = min(low + 1, len(ordered) - 1)
        p95 = ordered[low] + (
            ordered[high] - ordered[low]
        ) * (pos - low)

        rss_peak = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss

        mps_allocated = None
        if "mps" in devices:
            mps_allocated = int(
                torch.mps.current_allocated_memory()
            )

        result = {
            "protocol": PROTOCOL,
            "case_id": case["case_id"],
            "attempt_token": case["claim_token"],
            "model": name,
            "precision": case["precision"],
            "placement": location,
            "layer_count": n_layers,
            "cut_layer": (
                cut if first_device != final_device
                else None
            ),
            "actual_layer_devices": devices,
            "qos": case["qos"],
            "qos_main_thread_verified": True,
            "qos_worker_threads_verified": False,
            "execution_mode": "sequential_prefill",
            "context_tokens": CONTEXT,
            "batch_size": case["batch_size"],
            "cpu_threads": case["cpu_threads"],
            "input_tokens_sha256": token_sha,
            "corpus_sha256": sha(CORPUS),
            "reference": "same_checkpoint_full_cpu_fp32",
            "numerical_allclose": allclose,
            "numerical_status": numerical_status,
            "token_agreement": token_agreement,
            "max_abs_logit_error": max_error,
            "timing_samples_ms": samples_ms,
            "timed_calls": completed_calls,
            "measurement_seconds": elapsed_s,
            "prefill_tokens_s": tokens_s,
            "requests_s": requests_s,
            "latency_median_ms":
                statistics.median(samples_ms),
            "latency_p95_ms": p95,
            "cpu_power_w": cpu_w,
            "gpu_power_w": gpu_w,
            "gross_energy_j_token":
                (cpu_w + gpu_w) / tokens_s,
            "power_sample_count": len(pairs),
            "power_raw_sha256": sha(power_path),
            "power_scope":
                "estimated_mac_cpu_gpu_soc_gross",
            "power_time_alignment":
                "same_run_untimestamped_samples",
            "rss_peak_bytes": int(rss_peak),
            "rss_peak_scope": "whole_worker_process",
            "mps_allocated_bytes": mps_allocated,
            "mps_memory_scope": "snapshot_not_peak",
            "torch_version": torch.__version__,
            "created_utc": now(),
        }

        save(destination, result)

        print(
            "RESULT",
            name,
            case["precision"],
            location,
            case["qos"],
            f"quality={numerical_status}",
            f"tokens_s={tokens_s:.2f}",
            f"power_w={cpu_w+gpu_w:.3f}",
            f"energy_j_token="
            f"{result['gross_energy_j_token']:.6f}",
            flush=True,
        )

    finally:
        for handle in handles:
            handle.remove()

def recover(db):
    rows = list(db.execute("""
        SELECT case_id,claim_token
        FROM t25_cases WHERE status='running'
    """))

    db.execute("BEGIN IMMEDIATE")
    try:
        for row in rows:
            db.execute("""
                UPDATE t25_attempts
                SET status='interrupted',
                    finished_utc=?
                WHERE attempt_id=?
                  AND status='running'
            """, (now(), row["claim_token"]))

            db.execute("""
                UPDATE t25_cases
                SET status='planned',claim_token=NULL
                WHERE case_id=?
            """, (row["case_id"],))

        # Power failures are retryable on a later session,
        # while their previous attempts remain recorded.
        db.execute("""
            UPDATE t25_cases
            SET status='planned',claim_token=NULL
            WHERE status='power_missing'
        """)

        db.commit()
    except BaseException:
        db.rollback()
        raise

    return len(rows)

def claim(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("""
            SELECT * FROM t25_cases
            WHERE status='planned'
            ORDER BY ordinal LIMIT 1
        """).fetchone()

        if row is None:
            db.commit()
            return None

        case = dict(row)
        token = uuid.uuid4().hex

        cursor = db.execute("""
            UPDATE t25_cases
            SET status='running',
                claim_token=?,
                attempt_count=attempt_count+1
            WHERE case_id=? AND status='planned'
        """, (token, case["case_id"]))
        assert cursor.rowcount == 1

        db.execute("""
            INSERT INTO t25_attempts (
                attempt_id,case_id,started_utc,status
            ) VALUES (?,?,?,'running')
        """, (token, case["case_id"], now()))

        db.commit()
        case["claim_token"] = token
        return case
    except BaseException:
        db.rollback()
        raise

def finish(db, case, path):
    data = json.loads(path.read_text())

    assert data["case_id"] == case["case_id"]
    assert data["attempt_token"] == case["claim_token"]
    assert data["power_sample_count"] >= 2
    assert data["qos_main_thread_verified"]
    assert len(data["actual_layer_devices"]) == (
        data["layer_count"]
    )

    power_path = path.with_name(
        "powermetrics_raw.txt"
    )
    assert sha(power_path) == data["power_raw_sha256"]

    for key in (
        "prefill_tokens_s",
        "requests_s",
        "latency_median_ms",
        "latency_p95_ms",
        "gross_energy_j_token",
        "cpu_power_w",
        "gpu_power_w",
    ):
        value = data[key]
        assert value is not None
        assert math.isfinite(value)
        assert value >= 0

    expected_energy = (
        data["cpu_power_w"]
        + data["gpu_power_w"]
    ) / data["prefill_tokens_s"]

    assert math.isclose(
        expected_energy,
        data["gross_energy_j_token"],
        rel_tol=1e-9,
    )

    result_hash = sha(path)
    values = (
        case["case_id"],
        int(data["numerical_allclose"]),
        data["token_agreement"],
        data["max_abs_logit_error"],
        data["numerical_status"],
        data["prefill_tokens_s"],
        data["requests_s"],
        data["latency_median_ms"],
        data["latency_p95_ms"],
        data["measurement_seconds"],
        data["cpu_power_w"],
        data["gpu_power_w"],
        data["gross_energy_j_token"],
        data["power_sample_count"],
        data["rss_peak_bytes"],
        data["mps_allocated_bytes"],
        data["layer_count"],
        data["cut_layer"],
        str(path.resolve()),
        result_hash,
        now(),
    )
    assert len(values) == 21

    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute(
            "INSERT INTO t25_results VALUES ("
            + ",".join("?" for _ in values)
            + ")",
            values
        )

        cursor = db.execute("""
            UPDATE t25_cases
            SET status='completed',
                claim_token=NULL,
                result_path=?,
                result_sha256=?,
                last_error=NULL
            WHERE case_id=? AND claim_token=?
        """, (
            str(path.resolve()),
            result_hash,
            case["case_id"],
            case["claim_token"],
        ))
        assert cursor.rowcount == 1

        db.execute("""
            UPDATE t25_attempts
            SET status='completed',
                finished_utc=?,exit_code=0
            WHERE attempt_id=?
        """, (now(), case["claim_token"]))

        db.commit()
    except BaseException:
        db.rollback()
        raise

def mark(db, case, status, code, message):
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("""
            UPDATE t25_cases
            SET status=?,claim_token=NULL,last_error=?
            WHERE case_id=? AND claim_token=?
        """, (
            status, message[-3000:],
            case["case_id"], case["claim_token"]
        ))
        db.execute("""
            UPDATE t25_attempts
            SET status=?,finished_utc=?,
                exit_code=?,message=?
            WHERE attempt_id=?
        """, (
            status, now(), code,
            message[-3000:], case["claim_token"]
        ))
        db.commit()
    except BaseException:
        db.rollback()
        raise

def report(db):
    print("\n=== T25 TEXT DATASET STATUS ===")

    for row in db.execute("""
        SELECT status,COUNT(*)
        FROM t25_cases GROUP BY status
    """):
        print(row[0], row[1])

    print(
        "MEASURED PERFORMANCE AND ENERGY:",
        db.execute(
            "SELECT COUNT(*) FROM t25_results"
        ).fetchone()[0],
    )
    print(
        "NUMERICAL MISMATCHES WITH MEASUREMENTS:",
        db.execute("""
            SELECT COUNT(*) FROM t25_results
            WHERE numerical_allclose=0
        """).fetchone()[0],
    )
    print(
        "VISION POWER CAMPAIGN:",
        db.execute(
            "SELECT COUNT(*) FROM t24r_measurements"
        ).fetchone()[0],
        "+",
        db.execute(
            "SELECT COUNT(*) FROM t24_1_results"
        ).fetchone()[0],
    )
    print("FREE DISK GiB:", round(disk_guard(), 2))
    print("SQLITE:", DB)

def run(limit):
    lock_path = BASE / (
        "production_campaigns/t23_mac_v1/"
        ".campaign.lock"
    )

    with lock_path.open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )

        prepare()
        db = connect()
        try:
            print(
                "RECOVERED INTERRUPTED:",
                recover(db),
                flush=True
            )

            consecutive_errors = 0

            for index in range(limit):
                disk_guard()
                case = claim(db)

                if case is None:
                    print("T25 QUEUE FINISHED")
                    break

                token = case["claim_token"]
                folder = (
                    ROOT / "runs" / case["case_id"]
                    / ("attempt_" + token)
                )
                folder.mkdir(parents=True)

                case_file = folder / "case.json"
                result_file = folder / "result.json"
                stdout_file = folder / "stdout.txt"
                stderr_file = folder / "stderr.txt"
                save(case_file, case)

                print(
                    f"[{index+1}/{limit}] RUN",
                    case["model"],
                    case["precision"],
                    case["placement"],
                    case["qos"],
                    "batch=", case["batch_size"],
                    flush=True,
                )

                interrupted = False
                timed_out = False

                with stdout_file.open("w") as out, \
                     stderr_file.open("w") as err:
                    proc = subprocess.Popen(
                        [
                            sys.executable, "-u",
                            str(Path(__file__).resolve()),
                            "worker",
                            str(case_file),
                            str(result_file),
                        ],
                        stdout=out,
                        stderr=err,
                        start_new_session=False,
                        env={
                            **os.environ,
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                            "HF_HUB_OFFLINE": "1",
                            "TRANSFORMERS_OFFLINE": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                        },
                    )
                    try:
                        code = proc.wait(
                            timeout=CASE_TIMEOUT
                        )
                    except KeyboardInterrupt:
                        interrupted = True
                        proc.send_signal(signal.SIGINT)
                        try:
                            code = proc.wait(timeout=12)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            code = proc.wait()
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        proc.kill()
                        code = proc.wait()

                error = stderr_file.read_text(
                    errors="replace"
                )

                if interrupted:
                    mark(
                        db, case, "interrupted",
                        code, error
                    )
                    db.execute("""
                        UPDATE t25_cases
                        SET status='planned'
                        WHERE case_id=?
                    """, (case["case_id"],))
                    print("INTERRUPTED — RESUMABLE")
                    break

                if code == 0 and result_file.is_file():
                    finish(db, case, result_file)
                    consecutive_errors = 0
                    print(
                        "COMMITTED",
                        stdout_file.read_text(
                            errors="replace"
                        )[-350:].strip(),
                        flush=True
                    )
                    continue

                if "POWER_MISSING" in error:
                    status = "power_missing"
                elif "NUMERICAL_INVALID" in error:
                    status = "numerical_invalid"
                elif (
                    "not implemented" in error.lower()
                    or "not supported" in error.lower()
                    or "unsupported" in error.lower()
                ):
                    status = "unsupported"
                elif timed_out:
                    status = "timeout"
                else:
                    status = "failed_runtime"

                mark(db, case, status, code, error)

                print(
                    "RECORDED FAILURE:",
                    status,
                    "exit=", code,
                    error[-900:],
                    flush=True
                )

                if status == "power_missing":
                    print(
                        "POWER FAILURE: CAMPAIGN PAUSED",
                        flush=True
                    )
                    break

                if code in (-6, -9, -11):
                    print(
                        "NATIVE CRASH: CAMPAIGN PAUSED",
                        flush=True
                    )
                    break

                if status == "failed_runtime":
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                if consecutive_errors >= 3:
                    print(
                        "THREE CONSECUTIVE RUNTIME "
                        "FAILURES: CAMPAIGN PAUSED",
                        flush=True
                    )
                    break

            report(db)
        finally:
            db.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("run", "worker", "status")
    )
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()

    if args.command == "worker":
        assert len(args.extra) == 2
        case = json.loads(
            Path(args.extra[0]).read_text()
        )
        worker(case, Path(args.extra[1]))
        return

    if args.command == "status":
        db = connect()
        try:
            report(db)
        finally:
            db.close()
        return

    assert len(args.extra) == 1
    limit = int(args.extra[0])
    assert limit >= 1
    run(limit)

if __name__ == "__main__":
    main()
