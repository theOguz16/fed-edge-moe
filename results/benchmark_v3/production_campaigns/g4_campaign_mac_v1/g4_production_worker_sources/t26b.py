import ctypes
import fcntl
import hashlib
import json
import math
import os
import queue
import resource
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()
DB = BASE / (
    "production_campaigns/t23_mac_v1/"
    "dataset_v3_t23_mac.sqlite"
)
T05 = BASE / (
    "validation/t05_20261015T173344Z_2438/"
    "metadata.json"
)
HOLDOUT = BASE / (
    "holdout_thread_confirmation/"
    "t12_20261015T192923Z_2438/"
    "holdout_images.json"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/"
    "imagenetv2-matched-frequency-format-val"
).resolve()
MANIFEST_ROOT = Path(
    "configs/partition_manifests_v2"
).resolve()
LOCK = DB.parent / ".campaign.lock"

PROTOCOL = "T26B_CONVNEXT_TRUE_PIPELINE_POWER_V1"
MODELS = ("convnext_base", "convnext_large")
QOS = {
    "background": 0x09,
    "utility": 0x11,
    "default": 0x15,
    "user_initiated": 0x19,
}
CUTS = (1, 3, 5)
DIRECTIONS = ("cpu_mps", "mps_cpu")
MICROS = (1, 2)
REPEATS = (1, 2)
BATCH = 8
THREADS = 4
WARMUPS = 2
MIN_ROUNDS = 6
MIN_WINDOW_S = 10.0
TIMEOUT_S = 1500

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def disk_guard():
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < 5:
        raise RuntimeError(
            f"DISK_GUARD free={free:.2f} GiB"
        )
    return free

def connect(readonly=False):
    assert DB.is_file()
    if readonly:
        db = sqlite3.connect(
            f"file:{DB}?mode=ro", uri=True,
            timeout=60
        )
    else:
        db = sqlite3.connect(
            str(DB), timeout=90,
            isolation_level=None
        )
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=90000")
    db.execute("PRAGMA foreign_keys=ON")
    return db

def set_qos(name):
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    setter = lib.pthread_set_qos_class_self_np
    getter = lib.qos_class_self
    setter.argtypes = [ctypes.c_uint, ctypes.c_int]
    setter.restype = ctypes.c_int
    getter.restype = ctypes.c_uint
    requested = QOS[name]
    ret = setter(requested, 0)
    actual = getter()
    if ret != 0 or actual != requested:
        raise RuntimeError(
            "QOS_VERIFICATION_FAILED "
            f"name={name} ret={ret} actual={actual}"
        )
    return int(actual)

def power_pairs(raw):
    import re
    matches = list(re.finditer(
        r"CPU Power:\s*([\d.]+)\s*mW",
        raw, re.IGNORECASE
    ))
    output = []
    for i, match in enumerate(matches):
        stop = (
            matches[i+1].start()
            if i + 1 < len(matches)
            else len(raw)
        )
        gpu = re.search(
            r"GPU Power:\s*([\d.]+)\s*mW",
            raw[match.end():stop],
            re.IGNORECASE
        )
        if gpu:
            a = float(match.group(1)) / 1000
            b = float(gpu.group(1)) / 1000
            if (
                math.isfinite(a)
                and math.isfinite(b)
                and a >= 0 and b >= 0
            ):
                output.append((a, b))
    return output

def mapping_for(direction, cut):
    assert direction in DIRECTIONS
    assert cut in CUTS
    a, b = (
        ("C", "A")
        if direction == "cpu_mps"
        else ("A", "C")
    )
    return a * cut + b * (6 - cut)

def cases():
    rows = []
    for repeat in REPEATS:
        for qos in QOS:
            for micro in MICROS:
                for cut in CUTS:
                    for direction in DIRECTIONS:
                        for model in MODELS:
                            bits = mapping_for(direction, cut)
                            ident = {
                                "protocol": PROTOCOL,
                                "model": model,
                                "direction": direction,
                                "cut": cut,
                                "bits": bits,
                                "micro": micro,
                                "qos": qos,
                                "repeat": repeat,
                            }
                            case_id = "t26b_" + hashlib.sha256(
                                json.dumps(
                                    ident, sort_keys=True
                                ).encode()
                            ).hexdigest()
                            rows.append((
                                case_id, len(rows), model,
                                direction, cut, bits,
                                micro, qos, repeat
                            ))
    assert len(rows) == 192
    return rows

def initialize(db):
    disk_guard()
    assert T05.is_file()
    assert HOLDOUT.is_file()

    assert db.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    for table, expected in (
        ("t23_measurements", 90),
        ("t24r_measurements", 203),
        ("t24_1_results", 13),
        ("t25_results", 256),
        ("t26a_results", 96),
    ):
        actual = db.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        assert actual == expected, (table, actual)

    db.execute("""
        CREATE TABLE IF NOT EXISTS t26b_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS t26b_cases (
            case_id TEXT PRIMARY KEY,
            ordinal INTEGER NOT NULL UNIQUE,
            model TEXT NOT NULL,
            direction TEXT NOT NULL,
            cut INTEGER NOT NULL,
            mapping TEXT NOT NULL,
            microbatch INTEGER NOT NULL,
            qos TEXT NOT NULL,
            repeat INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'planned',
            current_attempt TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS t26b_attempts (
            attempt_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            started_utc TEXT NOT NULL,
            finished_utc TEXT,
            status TEXT NOT NULL,
            exit_code INTEGER,
            error TEXT,
            folder TEXT NOT NULL,
            FOREIGN KEY(case_id)
                REFERENCES t26b_cases(case_id)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS t26b_results (
            case_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            numerical_status TEXT NOT NULL,
            top1_agreement INTEGER NOT NULL,
            max_abs_error REAL NOT NULL,
            throughput_images_s REAL NOT NULL,
            round_throughput_median REAL NOT NULL,
            batch_wall_median_ms REAL NOT NULL,
            microbatch_latency_median_ms REAL NOT NULL,
            overlap_median_ms REAL NOT NULL,
            cpu_power_w REAL NOT NULL,
            gpu_power_w REAL NOT NULL,
            gross_energy_j_image REAL NOT NULL,
            power_samples INTEGER NOT NULL,
            power_window_s REAL NOT NULL,
            rss_peak_bytes INTEGER,
            mps_memory_snapshot_bytes INTEGER,
            result_path TEXT NOT NULL,
            result_sha256 TEXT NOT NULL,
            power_sha256 TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            PRIMARY KEY(case_id,mode),
            FOREIGN KEY(case_id)
                REFERENCES t26b_cases(case_id)
        )
    """)

    meta = {
        "protocol": PROTOCOL,
        "runner_sha256": sha(Path(__file__)),
        "t05_sha256": sha(T05),
        "holdout_sha256": sha(HOLDOUT),
        "models": json.dumps(MODELS),
        "batch": str(BATCH),
        "cpu_threads": str(THREADS),
        "precision": "fp32",
        "qos_main_and_pipeline_threads": "verified",
        "pytorch_native_thread_qos": "not_verified",
        "power_boundary": "estimated_mac_cpu_gpu_soc",
        "kernel_overlap_proven": "false",
    }

    for key, value in meta.items():
        old = db.execute(
            "SELECT value FROM t26b_metadata WHERE key=?",
            (key,)
        ).fetchone()
        if old is None:
            db.execute(
                "INSERT INTO t26b_metadata VALUES (?,?)",
                (key, value)
            )
        else:
            assert old[0] == value, (
                "Protocol/source changed", key
            )

    db.execute("BEGIN IMMEDIATE")
    try:
        for row in cases():
            db.execute("""
                INSERT OR IGNORE INTO t26b_cases (
                    case_id,ordinal,model,direction,cut,
                    mapping,microbatch,qos,repeat
                ) VALUES (?,?,?,?,?,?,?,?,?)
            """, row)

        count = db.execute(
            "SELECT COUNT(*) FROM t26b_cases"
        ).fetchone()[0]
        assert count == 192

        interrupted = list(db.execute("""
            SELECT case_id,current_attempt
            FROM t26b_cases
            WHERE status='running'
        """))

        for row in interrupted:
            db.execute("""
                UPDATE t26b_attempts
                SET status='interrupted',finished_utc=?
                WHERE attempt_id=?
                  AND status='running'
            """, (utc(), row["current_attempt"]))
            db.execute("""
                UPDATE t26b_cases
                SET status='planned',current_attempt=NULL
                WHERE case_id=?
            """, (row["case_id"],))

        db.commit()
    except BaseException:
        db.rollback()
        raise

    for row in db.execute("""
        SELECT result_path,result_sha256
        FROM t26b_results
    """):
        path = Path(row[0])
        assert path.is_file()
        assert sha(path) == row[1]

def claim(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("""
            SELECT * FROM t26b_cases
            WHERE status='planned'
            ORDER BY ordinal LIMIT 1
        """).fetchone()
        if row is None:
            db.commit()
            return None

        case = dict(row)
        token = uuid.uuid4().hex
        folder = ROOT / "runs" / case["case_id"] / token
        folder.mkdir(parents=True)

        changed = db.execute("""
            UPDATE t26b_cases
            SET status='running',current_attempt=?
            WHERE case_id=? AND status='planned'
        """, (token, case["case_id"]))
        assert changed.rowcount == 1

        db.execute("""
            INSERT INTO t26b_attempts (
                attempt_id,case_id,started_utc,
                status,folder
            ) VALUES (?,?,?,'running',?)
        """, (token, case["case_id"], utc(), str(folder)))
        db.commit()
        return case, token, folder
    except BaseException:
        db.rollback()
        raise

def fail(db, case, token, status, code, message):
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("""
            UPDATE t26b_attempts
            SET status=?,finished_utc=?,
                exit_code=?,error=?
            WHERE attempt_id=?
              AND status='running'
        """, (
            status, utc(), code,
            message[-4000:], token
        ))

        next_status = (
            "planned" if status == "interrupted"
            else "failed"
        )
        changed = db.execute("""
            UPDATE t26b_cases
            SET status=?,current_attempt=NULL
            WHERE case_id=? AND current_attempt=?
        """, (
            next_status,case["case_id"],token
        ))
        assert changed.rowcount == 1
        db.commit()
    except BaseException:
        db.rollback()
        raise

def commit(db, case, token, folder):
    path = folder / "result.json"
    obj = json.loads(path.read_text())
    assert obj["protocol"] == PROTOCOL
    assert obj["case_id"] == case["case_id"]
    assert obj["model"] == case["model"]
    assert obj["mapping"] == case["mapping"]
    assert obj["qos"] == case["qos"]
    assert obj["repeat"] == case["repeat"]
    assert obj["microbatch"] == case["microbatch"]
    assert obj["precision"] == "fp32"
    assert obj["pipeline_threads_verified"]
    assert obj["main_qos_verified"]
    assert obj["pipeline_separate_threads"]
    assert obj["kernel_overlap_proven"] is False

    evidence = []
    for mode in ("sequential", "pipeline"):
        m = obj["measurements"][mode]
        power_path = folder / (
            "powermetrics_" + mode + ".txt"
        )

        assert power_path.is_file()
        assert sha(power_path) == m["power_sha256"]
        assert m["power_samples"] >= 2
        assert len(m["rounds"]) >= MIN_ROUNDS
        assert m["power_window_s"] > 0

        throughput = m["throughput_images_s"]
        cpu = m["cpu_power_w"]
        gpu = m["gpu_power_w"]
        energy = m["gross_energy_j_image"]
        assert throughput > 0
        assert all(
            math.isfinite(x) and x >= 0
            for x in (throughput, cpu, gpu, energy)
        )
        assert math.isclose(
            throughput,
            BATCH*len(m["rounds"]) /
            m["power_window_s"],
            rel_tol=1e-9
        )
        assert math.isclose(
            energy, (cpu+gpu)/throughput,
            rel_tol=1e-9,
            abs_tol=1e-12
        )

        evidence.append((
            case["case_id"], mode,
            m["numerical_status"],
            int(m["top1_agreement"]),
            m["max_abs_error"],
            throughput,
            m["round_throughput_median"],
            m["batch_wall_median_ms"],
            m["microbatch_latency_median_ms"],
            m["overlap_median_ms"],
            cpu, gpu, energy,
            m["power_samples"],
            m["power_window_s"],
            obj["rss_peak_bytes"],
            obj["mps_memory_snapshot_bytes"],
            str(path.resolve()),
            sha(path),
            m["power_sha256"],
            token, utc()
        ))

    db.execute("BEGIN IMMEDIATE")
    try:
        for values in evidence:
            assert len(values) == 22
            db.execute(
                "INSERT INTO t26b_results VALUES ("
                + ",".join("?" for _ in values)
                + ")",
                values
            )

        changed = db.execute("""
            UPDATE t26b_cases
            SET status='completed',current_attempt=NULL
            WHERE case_id=? AND current_attempt=?
        """, (case["case_id"], token))
        assert changed.rowcount == 1

        db.execute("""
            UPDATE t26b_attempts
            SET status='completed',
                finished_utc=?,exit_code=0
            WHERE attempt_id=? AND status='running'
        """, (utc(), token))
        db.commit()
    except BaseException:
        db.rollback()
        raise

    seq = obj["measurements"]["sequential"]
    pip = obj["measurements"]["pipeline"]
    print(
        "COMMITTED",
        case["model"],
        case["direction"],
        f"cut={case['cut']}",
        f"micro={case['microbatch']}",
        f"qos={case['qos']}",
        f"repeat={case['repeat']}",
        f"speedup={pip['throughput_images_s']/seq['throughput_images_s']:.3f}x",
        f"seq_J/image={seq['gross_energy_j_image']:.6f}",
        f"pipe_J/image={pip['gross_energy_j_image']:.6f}",
        flush=True
    )

def stop_group(proc):
    if proc.poll() is not None:
        return
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

def report(db):
    print("\n=== T26B CONVNEXT PIPELINE STATUS ===")
    for row in db.execute("""
        SELECT status,COUNT(*)
        FROM t26b_cases GROUP BY status
    """):
        print(row[0], row[1])
    print(
        "POWER+PERFORMANCE MODE MEASUREMENTS:",
        db.execute(
            "SELECT COUNT(*) FROM t26b_results"
        ).fetchone()[0], "/384"
    )
    print("PIPELINE MODEL COUNT: 2")
    print("SOURCE DATABASE:", DB)

def run(limit):
    with LOCK.open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )
        db = connect()
        try:
            initialize(db)
            print("T26B PROTOCOL VERIFIED", flush=True)
            for index in range(limit):
                disk_guard()
                claimed = claim(db)
                if claimed is None:
                    print("T26B QUEUE FINISHED")
                    break

                case, token, folder = claimed
                case_file = folder / "case.json"
                result_file = folder / "result.json"
                stdout = folder / "stdout.txt"
                stderr = folder / "stderr.txt"

                save(case_file, {
                    **case,
                    "attempt_token": token,
                })

                print(
                    f"[{index+1}/{limit}]",
                    case["model"],
                    case["direction"],
                    "cut=", case["cut"],
                    "micro=", case["microbatch"],
                    "qos=", case["qos"],
                    "repeat=", case["repeat"],
                    flush=True
                )

                interrupted = False
                timeout = False
                with stdout.open("w") as out, \
                     stderr.open("w") as err:
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
                        process_group=0,
                        env={
                            **os.environ,
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                        },
                    )
                    try:
                        code = proc.wait(timeout=TIMEOUT_S)
                    except KeyboardInterrupt:
                        interrupted = True
                        stop_group(proc)
                        code = proc.returncode
                    except subprocess.TimeoutExpired:
                        timeout = True
                        stop_group(proc)
                        code = proc.returncode

                errors = stderr.read_text(errors="replace")

                if interrupted:
                    fail(
                        db, case, token,
                        "interrupted", code, errors
                    )
                    print("INTERRUPTED — RESUMABLE")
                    break

                if timeout or code != 0:
                    stop_group(proc)
                    status = (
                        "timeout" if timeout
                        else (
                            "power_missing"
                            if "POWER_MISSING" in errors
                            else "failed_runtime"
                        )
                    )
                    fail(
                        db, case, token,
                        status, code, errors
                    )
                    print(
                        "FAILED — CAMPAIGN PAUSED",
                        status, "exit=", code,
                        errors[-1800:],
                        flush=True
                    )
                    break

                if not result_file.is_file():
                    fail(
                        db, case, token,
                        "missing_result", code,
                        "Worker returned 0 without result.json"
                    )
                    print("MISSING RESULT — PAUSED")
                    break

                try:
                    commit(db, case, token, folder)
                except Exception as exc:
                    fail(
                        db, case, token,
                        "validation_failed", code, repr(exc)
                    )
                    print(
                        "RESULT VALIDATION FAILED:",
                        repr(exc), flush=True
                    )
                    break

            report(db)
        finally:
            db.close()

def percentile(values, q):
    s = sorted(values)
    p = (len(s)-1)*q
    lo = int(p)
    hi = min(lo+1,len(s)-1)
    return s[lo]+(s[hi]-s[lo])*(p-lo)

def worker(case, destination):
    import torch
    from PIL import Image
    from torch import nn
    from torchvision.models import (
        convnext_base, convnext_large,
        ConvNeXt_Base_Weights,
        ConvNeXt_Large_Weights,
    )

    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()

    torch.set_num_threads(THREADS)
    torch.manual_seed(2026)

    main_qos = set_qos(case["qos"])

    model_specs = {
        "convnext_base": (
            convnext_base,
            ConvNeXt_Base_Weights.IMAGENET1K_V1,
            "convnext_base__fine41.json",
        ),
        "convnext_large": (
            convnext_large,
            ConvNeXt_Large_Weights.IMAGENET1K_V1,
            "convnext_large__fine41.json",
        ),
    }
    factory, weights, manifest_name = model_specs[
        case["model"]
    ]

    meta = json.loads(T05.read_text())
    manifest_path = MANIFEST_ROOT / manifest_name
    manifest = json.loads(manifest_path.read_text())

    expected = (
        [["features.0"]]
        + [[f"features.1.{i}"] for i in range(3)]
        + [["features.2"]]
        + [[f"features.3.{i}"] for i in range(3)]
        + [["features.4"]]
        + [[f"features.5.{i}"] for i in range(27)]
        + [["features.6"]]
        + [[f"features.7.{i}"] for i in range(3)]
        + [["avgpool", "classifier"]]
    )
    assert manifest["group_count"] == 41
    assert [
        group["modules"] for group in manifest["groups"]
    ] == expected

    checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )
    expected_weight_sha = meta["models"][
        case["model"]
    ]["weight_sha256"]
    assert checkpoint.is_file()
    assert sha(checkpoint) == expected_weight_sha

    records = json.loads(HOLDOUT.read_text())
    assert len(records) >= BATCH
    transform = weights.transforms()

    tensors = []
    image_shas = []
    for record in records[:BATCH]:
        path = IMAGE_ROOT / record["path"]
        assert path.is_file()
        assert sha(path) == record["sha256"]
        with Image.open(path) as image:
            tensors.append(
                transform(image.convert("RGB"))
            )
        image_shas.append(record["sha256"])

    inputs = torch.stack(tensors)
    assert tuple(inputs.shape) == (BATCH,3,224,224)

    model = factory(weights=None).eval().cpu()
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    del state

    with torch.inference_mode():
        reference = model(inputs).detach().clone()
    assert tuple(reference.shape) == (BATCH,1000)
    assert bool(torch.isfinite(reference).all().item())

    stages = [
        model.features[0],
        model.features[1],
        nn.Sequential(
            model.features[2],model.features[3]
        ),
        nn.Sequential(
            model.features[4],model.features[5]
        ),
        nn.Sequential(
            model.features[6],model.features[7]
        ),
        nn.Sequential(
            model.avgpool,model.classifier
        ),
    ]

    assert len(stages) == 6
    with torch.inference_mode():
        grouped = inputs
        for stage in stages:
            grouped = stage(grouped)
    assert torch.allclose(
        grouped,reference,rtol=1e-5,atol=1e-5
    )

    bits = mapping_for(
        case["direction"],case["cut"]
    )
    assert bits == case["mapping"]

    dev1 = "cpu" if bits[0] == "C" else "mps"
    dev2 = "cpu" if bits[-1] == "C" else "mps"
    cut = case["cut"]

    for i,stage in enumerate(stages):
        stage.to(dev1 if i < cut else dev2)
    torch.mps.synchronize()

    first = stages[:cut]
    second = stages[cut:]

    micro = case["microbatch"]
    chunks = [
        inputs[i:i+micro]
        for i in range(0,BATCH,micro)
    ]
    assert len(chunks) >= 2

    def stage_one(index,x):
        began = time.perf_counter_ns()
        y = x.to(dev1)
        for stage in first:
            y = stage(y)
        if dev1 == "mps":
            torch.mps.synchronize()

        computed = time.perf_counter_ns()
        transfer_began = computed
        transfer_ended = computed

        if dev2 == "cpu" and y.device.type == "mps":
            y = y.to("cpu")
            torch.mps.synchronize()
            transfer_ended = time.perf_counter_ns()

        event = {
            "index":index,
            "stage1_start_ns":began,
            "stage1_end_ns":computed,
            "transfer_start_ns":transfer_began,
            "transfer_end_ns":transfer_ended,
        }
        return y,event

    def stage_two(index,y,event):
        event = dict(event)

        if dev2 == "mps" and y.device.type != "mps":
            transfer_began = time.perf_counter_ns()
            y = y.to("mps")
            torch.mps.synchronize()
            event["transfer_start_ns"] = transfer_began
            event["transfer_end_ns"] = time.perf_counter_ns()

        began = time.perf_counter_ns()
        for stage in second:
            y = stage(y)
        if dev2 == "mps":
            torch.mps.synchronize()
        computed = time.perf_counter_ns()

        y = y.to("cpu").detach().clone()
        if dev2 == "mps":
            torch.mps.synchronize()

        event.update({
            "stage2_start_ns":began,
            "stage2_end_ns":computed,
            "output_ready_ns":time.perf_counter_ns(),
        })
        return y,event

    def sequential():
        values = [None]*len(chunks)
        events = []
        started = time.perf_counter_ns()
        with torch.inference_mode():
            for index,x in enumerate(chunks):
                y,e = stage_one(index,x)
                z,e = stage_two(index,y,e)
                values[index] = z
                events.append(e)
        ended = time.perf_counter_ns()
        return torch.cat(values),events,started,ended,None

    def pipeline():
        channel = queue.Queue(maxsize=2)
        stop = threading.Event()
        errors = []
        values = [None]*len(chunks)
        events = [None]*len(chunks)
        qos_actual = {}

        def producer():
            try:
                qos_actual["producer"] = set_qos(
                    case["qos"]
                )
                with torch.inference_mode():
                    for index,x in enumerate(chunks):
                        if stop.is_set():
                            break
                        y,e = stage_one(index,x)
                        while not stop.is_set():
                            try:
                                channel.put(
                                    (index,y,e),timeout=0.25
                                )
                                break
                            except queue.Full:
                                continue
                    while not stop.is_set():
                        try:
                            channel.put(None,timeout=0.25)
                            break
                        except queue.Full:
                            continue
            except BaseException:
                errors.append(traceback.format_exc())
                stop.set()

        def consumer():
            try:
                qos_actual["consumer"] = set_qos(
                    case["qos"]
                )
                with torch.inference_mode():
                    while True:
                        try:
                            item = channel.get(timeout=0.25)
                        except queue.Empty:
                            if stop.is_set():
                                break
                            continue
                        if item is None:
                            break
                        index,y,e = item
                        z,e = stage_two(index,y,e)
                        values[index] = z
                        events[index] = e
            except BaseException:
                errors.append(traceback.format_exc())
                stop.set()

        t1 = threading.Thread(
            target=producer,daemon=True
        )
        t2 = threading.Thread(
            target=consumer,daemon=True
        )

        started = time.perf_counter_ns()
        t2.start()
        t1.start()
        t1.join(timeout=180)
        t2.join(timeout=180)

        if t1.is_alive() or t2.is_alive():
            stop.set()
            raise RuntimeError(
                "PIPELINE_THREAD_TIMEOUT"
            )
        if errors:
            raise RuntimeError(
                "PIPELINE_THREAD_ERROR\n"
                + "\n".join(errors)
            )
        assert all(x is not None for x in values)
        assert all(x is not None for x in events)
        assert qos_actual == {
            "producer":QOS[case["qos"]],
            "consumer":QOS[case["qos"]],
        },qos_actual

        ended = time.perf_counter_ns()
        return (
            torch.cat(values),
            events,started,ended,qos_actual
        )

    def overlap_ns(events):
        overlap = 0
        for a in events:
            for b in events:
                if a["index"] == b["index"]:
                    continue
                left = max(
                    a["stage1_start_ns"],
                    b["stage2_start_ns"]
                )
                right = min(
                    a["stage1_end_ns"],
                    b["stage2_end_ns"]
                )
                overlap += max(0,right-left)
        return overlap

    def run_mode(mode):
        if mode == "pipeline":
            return pipeline()
        return sequential()

    mode_order = (
        ("sequential","pipeline")
        if case["repeat"] == 1
        else ("pipeline","sequential")
    )

    measurements = {}
    qos_evidence = {}

    for mode in mode_order:
        print(
            "MODE WARMUP:",mode,flush=True
        )
        for _ in range(WARMUPS):
            actual,_,_,_,qos_threads = run_mode(mode)
            assert torch.isfinite(actual).all().item()
            if qos_threads is not None:
                qos_evidence.update(qos_threads)

        print(
            "MODE MEASUREMENT:",mode,flush=True
        )
        power_path = destination.with_name(
            "powermetrics_" + mode + ".txt"
        )

        rounds = []
        all_micro_latencies = []
        numeric_ok = True
        top1_ok = True
        max_error = 0.0

        with power_path.open("w") as sensor_file:
            sensor = subprocess.Popen(
                [
                    "sudo","-n","powermetrics",
                    "--samplers","cpu_power,gpu_power",
                    "-i","1000","-n","2000",
                ],
                stdout=sensor_file,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            started_window = time.monotonic()
            try:
                while (
                    len(rounds) < MIN_ROUNDS
                    or time.monotonic()-started_window
                       < MIN_WINDOW_S
                ):
                    actual,events,start,end,qos_threads = (
                        run_mode(mode)
                    )
                    assert len(events) == len(chunks)
                    if qos_threads is not None:
                        qos_evidence.update(qos_threads)

                    if not torch.isfinite(actual).all().item():
                        raise RuntimeError(
                            "NUMERICAL_INVALID nonfinite_logits"
                        )

                    difference = float(
                        (actual-reference).abs().max().item()
                    )
                    max_error = max(max_error,difference)
                    numeric_ok = (
                        numeric_ok
                        and bool(torch.allclose(
                            actual,reference,
                            atol=1e-3,rtol=1e-3
                        ))
                    )
                    top1_ok = (
                        top1_ok
                        and bool(torch.equal(
                            actual.argmax(dim=1),
                            reference.argmax(dim=1)
                        ))
                    )

                    wall_s = (end-start)/1e9
                    assert wall_s > 0
                    latencies = [
                        (
                            e["output_ready_ns"]
                            - e["stage1_start_ns"]
                        )/1e6 for e in events
                    ]
                    all_micro_latencies.extend(latencies)

                    rounds.append({
                        "round":len(rounds)+1,
                        "batch_wall_ms":wall_s*1000,
                        "throughput_images_s":BATCH/wall_s,
                        "microbatch_latency_median_ms":
                            statistics.median(latencies),
                        "stage_overlap_ms":
                            overlap_ns(events)/1e6,
                        "events":events,
                    })
            finally:
                window_s = (
                    time.monotonic()-started_window
                )
                if sensor.poll() is None:
                    sensor.send_signal(signal.SIGINT)
                try:
                    sensor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    sensor.kill()
                    sensor.wait()

        raw = power_path.read_text(errors="replace")
        power = power_pairs(raw)
        if len(power) < 2:
            raise RuntimeError(
                "POWER_MISSING "
                f"mode={mode} samples={len(power)} "
                f"sensor_exit={sensor.returncode} "
                f"tail={raw[-500:]!r}"
            )

        cpu_w = statistics.mean(p[0] for p in power)
        gpu_w = statistics.mean(p[1] for p in power)
        throughput = BATCH*len(rounds)/window_s
        assert throughput > 0

        measurements[mode] = {
            "rounds":rounds,
            "numerical_status":(
                "numerical_pass"
                if numeric_ok and top1_ok
                else "numerical_mismatch_measured"
            ),
            "top1_agreement":top1_ok,
            "max_abs_error":max_error,
            "throughput_images_s":throughput,
            "round_throughput_median":
                statistics.median(
                    r["throughput_images_s"]
                    for r in rounds
                ),
            "batch_wall_median_ms":
                statistics.median(
                    r["batch_wall_ms"]
                    for r in rounds
                ),
            "microbatch_latency_median_ms":
                statistics.median(all_micro_latencies),
            "overlap_median_ms":
                statistics.median(
                    r["stage_overlap_ms"]
                    for r in rounds
                ),
            "cpu_power_w":cpu_w,
            "gpu_power_w":gpu_w,
            "gross_energy_j_image":
                (cpu_w+gpu_w)/throughput,
            "power_samples":len(power),
            "power_window_s":window_s,
            "power_sha256":sha(power_path),
        }
        print(
            "MODE FINISHED:",mode,
            f"images/s={throughput:.3f}",
            f"power_samples={len(power)}",
            flush=True
        )

    assert qos_evidence == {
        "producer":QOS[case["qos"]],
        "consumer":QOS[case["qos"]],
    }

    rss = int(resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss)
    mps_memory = int(
        torch.mps.current_allocated_memory()
    )

    result = {
        "protocol":PROTOCOL,
        "case_id":case["case_id"],
        "model":case["model"],
        "precision":"fp32",
        "mapping":bits,
        "direction":case["direction"],
        "cut":cut,
        "microbatch":micro,
        "batch_size":BATCH,
        "qos":case["qos"],
        "repeat":case["repeat"],
        "qos_main_class":main_qos,
        "qos_thread_classes":qos_evidence,
        "main_qos_verified":True,
        "pipeline_threads_verified":True,
        "pytorch_native_threads_qos_verified":False,
        "pipeline_separate_threads":True,
        "kernel_overlap_proven":False,
        "cpu_thread_count":THREADS,
        "checkpoint_sha256":expected_weight_sha,
        "manifest_sha256":sha(manifest_path),
        "input_image_sha256_list":image_shas,
        "mode_order":mode_order,
        "measurements":measurements,
        "rss_peak_bytes":rss,
        "rss_scope":"whole_worker_process",
        "mps_memory_snapshot_bytes":mps_memory,
        "mps_memory_scope":"snapshot_not_peak",
        "power_scope":"estimated_cpu_gpu_soc_gross",
        "power_alignment":
            "mode_scoped_untimestamped_samples",
        "torch_version":torch.__version__,
        "created_utc":utc(),
    }

    save(destination,result)

def status():
    db = connect(readonly=True)
    try:
        report(db)
    finally:
        db.close()

def retry_failed():
    with LOCK.open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )
        db = connect()
        try:
            initialize(db)
            changed = db.execute("""
                UPDATE t26b_cases
                SET status='planned',current_attempt=NULL
                WHERE status='failed'
            """)
            print(
                "REQUEUED FAILED CASES:",
                changed.rowcount
            )
            print("OLD FAILED ATTEMPTS: PRESERVED")
        finally:
            db.close()

def main():
    if len(sys.argv)==2 and sys.argv[1]=="status":
        status()
        return

    if len(sys.argv)==2 and sys.argv[1]=="retry-failed":
        retry_failed()
        return

    if len(sys.argv)==4 and sys.argv[1]=="worker":
        case = json.loads(
            Path(sys.argv[2]).read_text()
        )
        worker(case,Path(sys.argv[3]))
        return

    if len(sys.argv)==3 and sys.argv[1]=="run":
        limit = int(sys.argv[2])
        assert limit >= 1
        run(limit)
        return

    raise SystemExit(
        "Usage: t26b.py run 192 | status | retry-failed"
    )

if __name__=="__main__":
    main()
