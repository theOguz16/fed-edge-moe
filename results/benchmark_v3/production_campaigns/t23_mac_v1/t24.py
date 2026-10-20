import ctypes
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DB = ROOT / "dataset_v3_t23_mac.sqlite"
RESULTS = ROOT / "t24_results"

BASE = Path("results/benchmark_v3").resolve()
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
MANIFESTS = Path(
    "configs/partition_manifests_v2"
).resolve()

MODELS = {
    "resnet50": (
        "resnet50__stage6.json",
        "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
    ),
    "convnext_base": (
        "convnext_base__fine41.json",
        "6075fbad31c688f9ed013922a59daf0fc2f4dc12e6e43bc01fa840c7bd1ca2cd",
    ),
    "convnext_large": (
        "convnext_large__fine41.json",
        "ea097f822cdb2cab40905cec3949f4bdf6d6bf1d546c3ec5e43f0b1c6275e144",
    ),
}
QOS = {
    "background": 0x09,
    "utility": 0x11,
    "default": 0x15,
    "user_initiated": 0x19,
}
PRECISIONS = ("fp32", "fp16", "bf16")
WORKLOADS = ("batch1_micro1", "batch8_micro2")
THREADS = 4
MEASURE_SECONDS = 8.0
MAX_SECONDS = 600
MIN_FREE_GIB = 4.0
PROTOCOL = "T24_MAC_VISION_MULTIAXIS_V1"

def utc():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def connect():
    conn = sqlite3.connect(
        str(DB), timeout=60, isolation_level=None
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def disk_guard():
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < MIN_FREE_GIB:
        raise RuntimeError(
            f"Disk guard: only {free:.2f} GiB free"
        )
    return free

def placements(conn, model):
    rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT placement_id,group_mapping,
                   transition_count
            FROM placements
            WHERE model=?
            """,
            (model,)
        )
    ]
    cpu = next(
        r for r in rows
        if set(r["group_mapping"]) == {"C"}
    )
    mps = next(
        r for r in rows
        if set(r["group_mapping"]) == {"A"}
    )

    hybrids = [
        r for r in rows
        if r["transition_count"] == 1
        and r["group_mapping"][0] == "A"
        and r["group_mapping"][-1] == "C"
    ]

    center = len(mps["group_mapping"]) // 2
    hybrid = min(
        hybrids,
        key=lambda r: (
            abs(
                r["group_mapping"].count("A")
                - center
            ),
            r["placement_id"],
        )
    )

    return [cpu, mps, hybrid]

def initialize():
    assert DB.is_file(), (
        "T23 database not found; original 90 "
        "measurements must be present"
    )
    disk_guard()
    conn = connect()

    try:
        assert conn.execute(
            "PRAGMA quick_check"
        ).fetchone()[0] == "ok"

        assert conn.execute(
            "SELECT COUNT(*) FROM t23_measurements"
        ).fetchone()[0] >= 90

        assert conn.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0] == 327336

        conn.execute("""
            CREATE TABLE IF NOT EXISTS t24_cases (
                case_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL UNIQUE,
                model TEXT NOT NULL,
                placement_id TEXT NOT NULL,
                mapping TEXT NOT NULL,
                workload_id TEXT NOT NULL,
                precision TEXT NOT NULL,
                qos TEXT NOT NULL,
                cpu_threads INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'planned',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                attempt_token TEXT,
                result_path TEXT,
                result_sha256 TEXT,
                last_error TEXT,
                FOREIGN KEY(placement_id)
                    REFERENCES placements(placement_id),
                FOREIGN KEY(workload_id)
                    REFERENCES workloads(workload_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS t24_attempts (
                token TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                started_utc TEXT NOT NULL,
                finished_utc TEXT,
                status TEXT NOT NULL,
                exit_code INTEGER,
                error TEXT,
                FOREIGN KEY(case_id)
                    REFERENCES t24_cases(case_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS t24_measurements (
                case_id TEXT PRIMARY KEY,
                throughput_items_s REAL NOT NULL,
                batch_latency_median_ms REAL NOT NULL,
                batch_latency_p95_ms REAL NOT NULL,
                max_abs_logit_error REAL NOT NULL,
                top1_agreement INTEGER NOT NULL,
                numerical_allclose INTEGER NOT NULL,
                rss_bytes INTEGER,
                mps_allocated_bytes INTEGER,
                qos_main_thread_verified INTEGER NOT NULL,
                power_status TEXT NOT NULL,
                cpu_power_w REAL,
                gpu_power_w REAL,
                soc_cpu_gpu_power_w REAL,
                gross_energy_j_item REAL,
                power_sample_count INTEGER NOT NULL,
                measurement_seconds REAL NOT NULL,
                result_sha256 TEXT NOT NULL,
                FOREIGN KEY(case_id)
                    REFERENCES t24_cases(case_id)
            )
        """)

        existing = conn.execute(
            "SELECT COUNT(*) FROM t24_cases"
        ).fetchone()[0]

        if existing:
            assert existing == 216, (
                "Unexpected existing T24 plan"
            )
            return

        records = []

        # Interleaved order avoids running one model
        # or one precision exclusively for hours.
        for workload in WORKLOADS:
            for qos in QOS:
                for precision in PRECISIONS:
                    for placement_index in range(3):
                        for model in MODELS:
                            placement = placements(
                                conn, model
                            )[placement_index]

                            identity = {
                                "protocol": PROTOCOL,
                                "model": model,
                                "placement_id":
                                    placement["placement_id"],
                                "workload": workload,
                                "precision": precision,
                                "qos": qos,
                                "threads": THREADS,
                            }

                            case_id = "t24_" + hashlib.sha256(
                                json.dumps(
                                    identity, sort_keys=True
                                ).encode()
                            ).hexdigest()

                            records.append((
                                case_id, len(records), model,
                                placement["placement_id"],
                                placement["group_mapping"],
                                workload, precision, qos,
                                THREADS,
                            ))

        assert len(records) == 216

        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany(
                """
                INSERT INTO t24_cases (
                    case_id,ordinal,model,placement_id,
                    mapping,workload_id,precision,qos,
                    cpu_threads
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                records
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

        save(ROOT / "t24_protocol.json", {
            "protocol": PROTOCOL,
            "cases": len(records),
            "models": list(MODELS),
            "precisions": list(PRECISIONS),
            "qos": list(QOS),
            "placements_per_model": 3,
            "workloads": list(WORKLOADS),
            "threads": THREADS,
            "measurement_seconds": MEASURE_SECONDS,
            "pipeline": "not_measured",
            "text_partition": "not_measured",
            "cpu_gpu_power":
                "powermetrics_optional_estimated_soc",
            "energy_scope":
                "gross_cpu_plus_gpu_power_over_throughput",
            "qos_scope":
                "verified_main_thread_only",
        })
    finally:
        conn.close()

def set_qos(name):
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    setter = lib.pthread_set_qos_class_self_np
    getter = lib.qos_class_self

    setter.argtypes = [ctypes.c_uint, ctypes.c_int]
    setter.restype = ctypes.c_int
    getter.restype = ctypes.c_uint

    result = setter(QOS[name], 0)
    actual = getter()

    if result != 0 or actual != QOS[name]:
        raise RuntimeError(
            f"QoS not verified: requested={name}, "
            f"set_result={result}, actual={actual}"
        )

    return True

def read_power(text):
    # macOS powermetrics reports CPU/GPU Power in mW.
    cpu = [
        float(x) / 1000
        for x in re.findall(
            r"CPU Power:\s*([\d.]+)\s*mW",
            text, re.IGNORECASE
        )
    ]
    gpu = [
        float(x) / 1000
        for x in re.findall(
            r"GPU Power:\s*([\d.]+)\s*mW",
            text, re.IGNORECASE
        )
    ]

    n = min(len(cpu), len(gpu))
    pairs = list(zip(cpu[:n], gpu[:n]))

    pairs = [
        (a, b) for a, b in pairs
        if math.isfinite(a) and math.isfinite(b)
        and a >= 0 and b >= 0
    ]

    return pairs

def worker(case, destination):
    import torch
    from torch import nn
    from PIL import Image
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        convnext_base, ConvNeXt_Base_Weights,
        convnext_large, ConvNeXt_Large_Weights,
    )

    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()

    verified_qos = set_qos(case["qos"])
    torch.set_num_threads(case["cpu_threads"])
    torch.manual_seed(2026)

    name = case["model"]

    if name == "resnet50":
        weight = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=None).eval()
    elif name == "convnext_base":
        weight = ConvNeXt_Base_Weights.IMAGENET1K_V1
        model = convnext_base(weights=None).eval()
    else:
        weight = ConvNeXt_Large_Weights.IMAGENET1K_V1
        model = convnext_large(weights=None).eval()

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlparse(weight.url).path).name
    )

    assert checkpoint.is_file()
    assert digest(checkpoint) == MODELS[name][1]
    assert json.loads(T05.read_text())[
        "models"
    ][name]["weight_sha256"] == MODELS[name][1]

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state, strict=True)
    del state

    batch = int(
        case["workload_id"].split("_")[0][5:]
    )
    micro = int(
        case["workload_id"].split("_")[1][5:]
    )
    assert batch % micro == 0

    records = json.loads(HOLDOUT.read_text())
    assert len(records) >= batch

    image_tensors = []
    input_hashes = []

    for record in records[:batch]:
        image_path = IMAGE_ROOT / record["path"]
        assert digest(image_path) == record["sha256"]

        with Image.open(image_path) as img:
            image_tensors.append(
                weight.transforms()(img.convert("RGB"))
            )
        input_hashes.append(record["sha256"])

    inputs = torch.stack(image_tensors)

    with torch.inference_mode():
        reference = model(inputs).detach().clone()

    manifest_path = MANIFESTS / MODELS[name][0]
    assert digest(manifest_path) == (
        sqlite_manifest_sha(case["model"])
    )
    manifest = json.loads(manifest_path.read_text())

    bits = case["mapping"]

    if name == "resnet50":
        stages = [
            nn.Sequential(
                model.conv1, model.bn1,
                model.relu, model.maxpool
            ),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.Sequential(
                model.avgpool,
                nn.Flatten(1),
                model.fc
            ),
        ]
    else:
        stages = [
            nn.Sequential(
                *[
                    model.get_submodule(part)
                    for part in group["modules"]
                ]
            )
            for group in manifest["groups"]
        ]

    assert len(stages) == len(bits)

    precision_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[case["precision"]]

    devices = [
        "cpu" if b == "C" else "mps"
        for b in bits
    ]
    has_mps = "mps" in devices

    # Precision is an actual model-compute dtype.
    for stage, device in zip(stages, devices):
        stage.to(device=device, dtype=precision_dtype)

    if has_mps:
        torch.mps.synchronize()

    def run_once():
        outputs = []

        for offset in range(0, batch, micro):
            x = inputs[offset:offset + micro].to(
                dtype=precision_dtype
            )

            for stage, device in zip(stages, devices):
                if x.device.type != device:
                    x = x.to(device)

                x = stage(x)

            outputs.append(x.float().cpu())

        if has_mps:
            torch.mps.synchronize()

        return torch.cat(outputs)

    with torch.inference_mode():
        output = run_once()

    finite = bool(torch.isfinite(output).all().item())
    if not finite:
        raise RuntimeError(
            "NUMERICAL_INVALID nonfinite_output"
        )

    diff = float(
        (reference - output).abs().max().item()
    )
    atol = 1e-3 if case["precision"] == "fp32" else 5e-2
    rtol = 1e-3 if case["precision"] == "fp32" else 5e-2

    allclose = bool(torch.allclose(
        reference, output, atol=atol, rtol=rtol
    ))
    top1 = bool(torch.equal(
        reference.argmax(1), output.argmax(1)
    ))

    if not allclose or not top1:
        raise RuntimeError(
            "NUMERICAL_MISMATCH "
            f"allclose={allclose} top1={top1} "
            f"max_error={diff}"
        )

    with torch.inference_mode():
        for _ in range(2):
            run_once()

    power_process = None
    power_output = ""
    power_status = "not_measured"

    cmd = [
        "sudo", "-n", "powermetrics",
        "--samplers", "cpu_power,gpu_power",
        "-i", "1000",
        "-n", "30",
    ]

    try:
        if shutil.which("powermetrics") and \
                shutil.which("sudo"):
            power_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            power_status = "sensor_attempted"
    except Exception as ex:
        power_status = (
            "sensor_launch_failed:" + type(ex).__name__
        )

    times = []
    start = time.monotonic()
    total_items = 0

    try:
        with torch.inference_mode():
            while time.monotonic() - start < MEASURE_SECONDS:
                if has_mps:
                    torch.mps.synchronize()

                t0 = time.perf_counter_ns()
                run_once()
                duration = (
                    time.perf_counter_ns() - t0
                ) / 1e6

                times.append(duration)
                total_items += batch
    finally:
        elapsed = time.monotonic() - start

        if power_process is not None:
            if power_process.poll() is None:
                try:
                    os.killpg(
                        power_process.pid, signal.SIGINT
                    )
                except ProcessLookupError:
                    pass

            try:
                power_output, _ = power_process.communicate(
                    timeout=8
                )
            except subprocess.TimeoutExpired:
                os.killpg(
                    power_process.pid, signal.SIGKILL
                )
                power_output, _ = power_process.communicate()

    assert times and elapsed > 0

    throughput = total_items / elapsed

    # Store raw sensor output for audit; no fabricated power.
    power_path = destination.with_name(
        "powermetrics_raw.txt"
    )
    power_path.write_text(power_output)
    pairs = read_power(power_output)

    cpu_power = gpu_power = combined = energy = None

    if len(pairs) >= 2:
        cpu_power = statistics.mean(x[0] for x in pairs)
        gpu_power = statistics.mean(x[1] for x in pairs)
        combined = cpu_power + gpu_power
        energy = combined / throughput

        power_status = (
            "measured_gross_estimate_"
            "without_per_sample_timestamps"
        )
    else:
        power_status = (
            "not_measured_insufficient_sensor_samples"
        )

    rss = None
    try:
        rss = int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True
        ).strip()) * 1024
    except Exception:
        pass

    mps_memory = (
        int(torch.mps.current_allocated_memory())
        if has_mps else None
    )

    ordered = sorted(times)
    index = 0.95 * (len(ordered) - 1)
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    p95 = ordered[lo] + (
        ordered[hi] - ordered[lo]
    ) * (index - lo)

    result = {
        "protocol": PROTOCOL,
        "case_id": case["case_id"],
        "attempt_token": case["attempt_token"],
        "created_utc": utc(),
        "model": name,
        "precision": case["precision"],
        "qos": case["qos"],
        "qos_main_thread_verified": verified_qos,
        "qos_worker_threads_verified": False,
        "placement_mapping": bits,
        "cpu_threads": case["cpu_threads"],
        "batch_size": batch,
        "microbatch_size": micro,
        "execution_mode": "sequential",
        "input_sha256": input_hashes,
        "weight_sha256": digest(checkpoint),
        "manifest_sha256": digest(manifest_path),
        "reference": "full_cpu_fp32",
        "comparison_atol": atol,
        "comparison_rtol": rtol,
        "numerical_allclose": allclose,
        "top1_agreement": top1,
        "max_abs_logit_error": diff,
        "timing_samples_ms": times,
        "timed_batches": len(times),
        "timed_items": total_items,
        "measurement_seconds": elapsed,
        "throughput_items_s": throughput,
        "batch_latency_median_ms":
            statistics.median(times),
        "batch_latency_p95_ms": p95,
        "rss_bytes": rss,
        "mps_allocated_bytes": mps_memory,
        "memory_measurement": "snapshot_not_peak",
        "power_status": power_status,
        "power_boundary": "estimated_mac_cpu_plus_gpu_soc",
        "power_sample_count": len(pairs),
        "cpu_power_w": cpu_power,
        "gpu_power_w": gpu_power,
        "soc_cpu_gpu_power_w": combined,
        "gross_energy_j_item": energy,
        "power_raw_sha256": digest(power_path),
        "power_warning":
            "No individual sample timestamp alignment; "
            "not wall-plug energy and not idle-adjusted",
        "pipeline_measured": False,
        "torch_version": torch.__version__,
    }

    save(destination, result)

    print(
        f"RESULT {name} {case['precision']} "
        f"qos={case['qos']} "
        f"throughput={throughput:.3f} item/s "
        f"power={combined} W "
        f"energy={energy} J/item",
        flush=True
    )

def sqlite_manifest_sha(model):
    # Read the verified catalog; do not assume
    # every model has the same manifest.
    conn = sqlite3.connect(
        f"file:{DB}?mode=ro",
        uri=True
    )
    try:
        rows = conn.execute(
            "SELECT DISTINCT manifest_sha256 "
            "FROM placements WHERE model=?",
            (model,)
        ).fetchall()
        assert len(rows) == 1
        return rows[0][0]
    finally:
        conn.close()

def finish(conn, case, path):
    data = json.loads(path.read_text())
    assert data["case_id"] == case["case_id"]
    assert data["attempt_token"] == case["attempt_token"]

    file_hash = digest(path)

    fields = (
        "case_id,throughput_items_s,"
        "batch_latency_median_ms,batch_latency_p95_ms,"
        "max_abs_logit_error,top1_agreement,"
        "numerical_allclose,rss_bytes,mps_allocated_bytes,"
        "qos_main_thread_verified,power_status,"
        "cpu_power_w,gpu_power_w,soc_cpu_gpu_power_w,"
        "gross_energy_j_item,power_sample_count,"
        "measurement_seconds,result_sha256"
    )

    values = (
        case["case_id"],
        data["throughput_items_s"],
        data["batch_latency_median_ms"],
        data["batch_latency_p95_ms"],
        data["max_abs_logit_error"],
        int(data["top1_agreement"]),
        int(data["numerical_allclose"]),
        data["rss_bytes"],
        data["mps_allocated_bytes"],
        int(data["qos_main_thread_verified"]),
        data["power_status"],
        data["cpu_power_w"],
        data["gpu_power_w"],
        data["soc_cpu_gpu_power_w"],
        data["gross_energy_j_item"],
        data["power_sample_count"],
        data["measurement_seconds"],
        file_hash,
    )

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            f"INSERT INTO t24_measurements ({fields}) "
            f"VALUES ({','.join('?' for _ in values)})",
            values
        )
        conn.execute(
            """
            UPDATE t24_cases
            SET status='completed',
                attempt_token=NULL,
                result_path=?,
                result_sha256=?,
                last_error=NULL
            WHERE case_id=? AND attempt_token=?
            """,
            (
                str(path.resolve()), file_hash,
                case["case_id"], case["attempt_token"]
            )
        )
        conn.execute(
            """
            UPDATE t24_attempts
            SET status='completed',finished_utc=?,
                exit_code=0
            WHERE token=?
            """,
            (utc(), case["attempt_token"])
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

def mark_error(conn, case, status, code, message):
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """
            UPDATE t24_cases
            SET status=?,attempt_token=NULL,
                last_error=?
            WHERE case_id=? AND attempt_token=?
            """,
            (
                status, message[-3000],
                case["case_id"], case["attempt_token"]
            )
        )
        conn.execute(
            """
            UPDATE t24_attempts
            SET status=?,finished_utc=?,
                exit_code=?,error=?
            WHERE token=?
            """,
            (
                status, utc(), code, message[-3000],
                case["attempt_token"]
            )
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

def recover(conn):
    active = conn.execute(
        """
        SELECT case_id,attempt_token
        FROM t24_cases WHERE status='running'
        """
    ).fetchall()

    conn.execute("BEGIN IMMEDIATE")
    try:
        for row in active:
            conn.execute(
                """
                UPDATE t24_attempts
                SET status='interrupted',finished_utc=?
                WHERE token=?
                """,
                (utc(), row["attempt_token"])
            )
            conn.execute(
                """
                UPDATE t24_cases
                SET status='planned',attempt_token=NULL
                WHERE case_id=?
                """,
                (row["case_id"],)
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    return len(active)

def claim(conn):
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            """
            SELECT * FROM t24_cases
            WHERE status='planned'
            ORDER BY ordinal LIMIT 1
            """
        ).fetchone()

        if row is None:
            conn.commit()
            return None

        case = dict(row)
        token = uuid.uuid4().hex

        conn.execute(
            """
            UPDATE t24_cases
            SET status='running',attempt_token=?,
                attempt_count=attempt_count+1
            WHERE case_id=? AND status='planned'
            """,
            (token, case["case_id"])
        )
        conn.execute(
            """
            INSERT INTO t24_attempts
            (token,case_id,started_utc,status)
            VALUES (?,?,?,'running')
            """,
            (token, case["case_id"], utc())
        )
        conn.commit()

        case["attempt_token"] = token
        return case
    except BaseException:
        conn.rollback()
        raise

def report(conn):
    print("\n=== T24 MULTIAXIS DATASET STATUS ===")

    for row in conn.execute(
        """
        SELECT status,COUNT(*) AS n
        FROM t24_cases GROUP BY status
        ORDER BY status
        """
    ):
        print(row["status"], row["n"])

    completed = conn.execute(
        "SELECT COUNT(*) FROM t24_measurements"
    ).fetchone()[0]

    powered = conn.execute(
        """
        SELECT COUNT(*) FROM t24_measurements
        WHERE gross_energy_j_item IS NOT NULL
        """
    ).fetchone()[0]

    print("VALID PERFORMANCE MEASUREMENTS:", completed)
    print("ENERGY ESTIMATES:", powered)
    print("ORIGINAL T23 MEASUREMENTS:", conn.execute(
        "SELECT COUNT(*) FROM t23_measurements"
    ).fetchone()[0])
    print("DISK FREE GiB:", round(disk_guard(), 2))
    print("SQLITE:", DB)
    print("SOURCE DATABASE MODIFIED: NO")

def run(limit):
    with (ROOT / ".campaign.lock").open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )

        initialize()
        conn = connect()

        try:
            print(
                "RECOVERED:", recover(conn),
                flush=True
            )

            for index in range(limit):
                disk_guard()
                case = claim(conn)

                if case is None:
                    print("T24 QUEUE FINISHED")
                    break

                folder = RESULTS / case["case_id"] / (
                    "attempt_" + case["attempt_token"]
                )
                folder.mkdir(parents=True)
                casefile = folder / "case.json"
                result = folder / "result.json"
                stdout = folder / "stdout.txt"
                stderr = folder / "stderr.txt"

                save(casefile, case)

                print(
                    f"[{index+1}/{limit}]",
                    case["model"],
                    case["precision"],
                    case["qos"],
                    case["mapping"][:12],
                    case["workload_id"],
                    flush=True
                )

                with stdout.open("w") as out, \
                     stderr.open("w") as err:

                    proc = subprocess.Popen(
                        [
                            sys.executable, "-u",
                            str(Path(__file__).resolve()),
                            "worker", str(casefile),
                            str(result)
                        ],
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                        }
                    )

                    try:
                        code = proc.wait(timeout=MAX_SECONDS)
                    except KeyboardInterrupt:
                        try:
                            os.killpg(proc.pid, signal.SIGINT)
                            proc.wait(timeout=8)
                        except (
                            subprocess.TimeoutExpired,
                            ProcessLookupError
                        ):
                            if proc.poll() is None:
                                os.killpg(
                                    proc.pid, signal.SIGKILL
                                )
                                proc.wait()

                        print(
                            "INTERRUPTED — resume later",
                            flush=True
                        )
                        break
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        code = 124

                if code == 0 and result.is_file():
                    finish(conn, case, result)
                    print(
                        "COMMITTED",
                        stdout.read_text(
                            errors="replace"
                        )[-320:],
                        flush=True
                    )
                else:
                    error = stderr.read_text(
                        errors="replace"
                    )[-2500:]

                    if "NUMERICAL_INVALID" in error:
                        status = "numerical_invalid"
                    elif "NUMERICAL_MISMATCH" in error:
                        status = "numerical_mismatch"
                    elif "not implemented" in error.lower() \
                         or "not supported" in error.lower():
                        status = "unsupported"
                    elif code == 124:
                        status = "timeout"
                    else:
                        status = "failed_runtime"

                    mark_error(
                        conn, case, status, code, error
                    )

                    print(
                        "RECORDED FAILURE:",
                        status, "exit=", code,
                        error[-450:],
                        flush=True
                    )

            report(conn)
        finally:
            conn.close()

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "worker":
        case = json.loads(
            Path(sys.argv[2]).read_text()
        )
        worker(case, Path(sys.argv[3]))
        return

    if len(sys.argv) != 3 or sys.argv[1] != "run":
        raise SystemExit("Usage: python t24.py run 24")

    limit = int(sys.argv[2])
    assert limit > 0
    run(limit)

if __name__ == "__main__":
    main()
