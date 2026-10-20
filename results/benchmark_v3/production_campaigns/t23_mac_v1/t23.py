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
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()

SOURCE = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "dataset_v3_text_quality_recovered.sqlite"
)
DB = ROOT / "dataset_v3_t23_mac.sqlite"
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
WORKLOADS = (
    "batch1_micro1",
    "batch4_micro1",
    "batch4_micro2",
    "batch8_micro2",
)
THREADS = (2, 4, 8)
PLACEMENTS_PER_MODEL = 24
WARMUPS = 2
TIMED = 6
TIMEOUT = 480
MIN_FREE_GIB = 4

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def connect(path=DB, readonly=False):
    if readonly:
        db = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=60,
        )
    else:
        db = sqlite3.connect(
            str(path),
            timeout=60,
            isolation_level=None,
        )
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=60000")
    db.execute("PRAGMA foreign_keys=ON")
    return db

def free_space():
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < MIN_FREE_GIB:
        raise RuntimeError(
            f"Disk safety stop: {free:.2f} GiB free"
        )
    return free

def rank_key(value):
    return hashlib.sha256(
        ("T23_MAC_V1|" + value).encode()
    ).hexdigest()

def select_placements(db, model):
    candidates = [
        dict(row)
        for row in db.execute(
            """
            SELECT placement_id,group_mapping,
                   transition_count
            FROM placements WHERE model=?
            """,
            (model,)
        )
    ]

    assert candidates

    # Explicitly include full CPU and full MPS.
    mandatory = [
        row for row in candidates
        if len(set(row["group_mapping"])) == 1
    ]
    assert len(mandatory) == 2

    selected = list(mandatory)
    used = {r["placement_id"] for r in selected}

    # Deterministic stratification by number of cuts
    # and CPU/MPS direction at the endpoints.
    bins = {}

    for row in candidates:
        if row["placement_id"] in used:
            continue

        bits = row["group_mapping"]
        key = (
            row["transition_count"],
            bits[0], bits[-1]
        )
        bins.setdefault(key, []).append(row)

    for key in bins:
        bins[key].sort(
            key=lambda r: rank_key(r["placement_id"])
        )

    keys = sorted(bins)

    while len(selected) < PLACEMENTS_PER_MODEL:
        added = False

        for key in keys:
            if bins[key]:
                row = bins[key].pop(0)
                selected.append(row)
                added = True

                if len(selected) == PLACEMENTS_PER_MODEL:
                    break

        assert added, "Placement selection exhausted"

    assert len({
        r["placement_id"] for r in selected
    }) == PLACEMENTS_PER_MODEL

    return selected

def prepare():
    assert SOURCE.is_file(), SOURCE
    assert T05.is_file()
    assert HOLDOUT.is_file()
    free_space()

    source = connect(SOURCE, readonly=True)
    try:
        assert source.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
        assert source.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0] == 327336
        assert source.execute(
            "SELECT COUNT(*) FROM vision_quality_predictions"
        ).fetchone()[0] == 60000

        if not DB.exists():
            tmp = ROOT / ("snapshot_" + uuid.uuid4().hex)
            dest = connect(tmp)
            try:
                source.backup(dest)
                assert dest.execute(
                    "PRAGMA integrity_check"
                ).fetchone()[0] == "ok"
            finally:
                dest.close()
            os.replace(tmp, DB)
    finally:
        source.close()

    db = connect()
    try:
        db.execute("""
            CREATE TABLE IF NOT EXISTS t23_campaign_tasks (
                task_id TEXT PRIMARY KEY,
                run_order INTEGER NOT NULL UNIQUE,
                model TEXT NOT NULL,
                placement_mapping TEXT NOT NULL,
                selection_protocol TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id)
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS t23_measurements (
                task_id TEXT PRIMARY KEY,
                batch_size INTEGER NOT NULL,
                microbatch_size INTEGER NOT NULL,
                cpu_threads INTEGER NOT NULL,
                model TEXT NOT NULL,
                placement_mapping TEXT NOT NULL,
                numerical_pass INTEGER NOT NULL,
                top1_agreement INTEGER NOT NULL,
                max_abs_error REAL NOT NULL,
                latency_median_ms REAL NOT NULL,
                latency_p95_ms REAL NOT NULL,
                throughput_images_s REAL NOT NULL,
                rss_bytes INTEGER,
                mps_allocated_bytes INTEGER,
                result_sha256 TEXT NOT NULL,
                recorded_utc TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id)
            )
        """)

        existing = db.execute(
            "SELECT COUNT(*) FROM t23_campaign_tasks"
        ).fetchone()[0]

        if existing:
            assert existing == (
                len(MODELS)
                * PLACEMENTS_PER_MODEL
                * len(WORKLOADS)
                * len(THREADS)
            )
            print(
                "EXISTING CAMPAIGN: VERIFIED",
                existing,
                flush=True,
            )
            return

        groups = {}

        for model in MODELS:
            groups[model] = select_placements(db, model)

        plan = []

        # Interleave models, placements, workloads and threads
        # rather than grouping all jobs by one model.
        for position in range(PLACEMENTS_PER_MODEL):
            for wid in WORKLOADS:
                for threads in THREADS:
                    for model in MODELS:
                        placement = groups[model][position]

                        rows = db.execute(
                            """
                            SELECT task_id,state
                            FROM tasks
                            WHERE placement_id=?
                              AND workload_id=?
                              AND cpu_threads=?
                              AND execution_mode='sequential'
                              AND precision='float32'
                              AND hardware_id='mac_m4'
                            """,
                            (
                                placement["placement_id"],
                                wid,
                                threads,
                            )
                        ).fetchall()

                        assert len(rows) == 1
                        row = rows[0]

                        # Already measured historical tasks are
                        # not silently rerun as new repetitions.
                        if row["state"] == "completed":
                            continue

                        assert row["state"] == "planned", (
                            row["task_id"], row["state"]
                        )

                        plan.append((
                            row["task_id"],
                            len(plan),
                            model,
                            placement["group_mapping"],
                            "T23_MAC_V1_DETERMINISTIC_STRATIFIED",
                        ))

        db.execute("BEGIN IMMEDIATE")
        try:
            for tid, order, model, bits, protocol in plan:
                db.execute(
                    """
                    INSERT INTO t23_campaign_tasks
                    VALUES (?,?,?,?,?)
                    """,
                    (tid, order, model, bits, protocol)
                )
                updated = db.execute(
                    """
                    UPDATE tasks SET state='queued'
                    WHERE task_id=? AND state='planned'
                    """,
                    (tid,)
                )
                assert updated.rowcount == 1
            db.commit()
        except BaseException:
            db.rollback()
            raise

        atomic_json(ROOT / "campaign_plan.json", {
            "protocol": "T23_MAC_V1",
            "models": list(MODELS),
            "precision": "float32",
            "execution_mode": "sequential",
            "placements_per_model": PLACEMENTS_PER_MODEL,
            "threads": THREADS,
            "workloads": WORKLOADS,
            "warmups": WARMUPS,
            "timed_rounds": TIMED,
            "selected_new_tasks": len(plan),
            "historically_completed_tasks_excluded": (
                len(MODELS)
                * PLACEMENTS_PER_MODEL
                * len(WORKLOADS)
                * len(THREADS)
                - len(plan)
            ),
            "source_db": str(SOURCE),
            "source_db_not_modified": True,
            "power_measured": False,
            "energy_measured": False,
            "pipeline_measured_in_this_campaign": False,
            "quality_reference": "Full CPU FP32 logits",
        })

        print(
            "T23 PRODUCTION TASKS QUEUED:",
            len(plan),
            flush=True
        )
    finally:
        db.close()

def resume(db):
    # Caller owns the exclusive campaign lock.
    rows = db.execute(
        """
        SELECT t.task_id,t.claim_token
        FROM tasks t
        JOIN t23_campaign_tasks c
        ON t.task_id=c.task_id
        WHERE t.state='running'
        """
    ).fetchall()

    if not rows:
        return 0

    db.execute("BEGIN IMMEDIATE")
    try:
        for row in rows:
            tid, token = row

            assert token

            db.execute(
                """
                UPDATE attempts
                SET state='interrupted',
                    finished_utc=?,
                    message='T23 recovered at restart'
                WHERE attempt_id=? AND state='running'
                """,
                (utc(), token)
            )

            db.execute(
                """
                UPDATE tasks
                SET state='queued',
                    claim_token=NULL,
                    started_utc=NULL
                WHERE task_id=? AND claim_token=?
                """,
                (tid, token)
            )
        db.commit()
    except BaseException:
        db.rollback()
        raise

    return len(rows)

def claim(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute(
            """
            SELECT t.task_id,p.model,p.group_mapping,
                   w.batch_size,w.microbatch_size,
                   t.cpu_threads,t.attempt_count
            FROM t23_campaign_tasks c
            JOIN tasks t ON c.task_id=t.task_id
            JOIN placements p ON p.placement_id=t.placement_id
            JOIN workloads w ON w.workload_id=t.workload_id
            WHERE t.state='queued'
            ORDER BY c.run_order LIMIT 1
            """
        ).fetchone()

        if row is None:
            db.commit()
            return None

        data = dict(row)
        token = uuid.uuid4().hex

        db.execute(
            """
            UPDATE tasks
            SET state='running',
                claim_token=?,
                started_utc=?,
                attempt_count=attempt_count+1
            WHERE task_id=? AND state='queued'
            """,
            (token, utc(), data["task_id"])
        )

        db.execute(
            """
            INSERT INTO attempts
            (attempt_id,task_id,state,started_utc)
            VALUES (?,?,'running',?)
            """,
            (token, data["task_id"], utc())
        )

        db.commit()
        data["token"] = token
        return data
    except BaseException:
        db.rollback()
        raise

def complete(db, task, path):
    record = json.loads(path.read_text())
    assert record["task_id"] == task["task_id"]
    assert record["attempt_id"] == task["token"]

    digest = sha(path)

    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute(
            """
            INSERT INTO t23_measurements
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                task["task_id"],
                task["batch_size"],
                task["microbatch_size"],
                task["cpu_threads"],
                task["model"],
                task["group_mapping"],
                int(record["numerical_pass"]),
                int(record["top1_agreement"]),
                record["max_abs_error"],
                record["latency_median_ms"],
                record["latency_p95_ms"],
                record["throughput_images_s"],
                record["rss_bytes"],
                record["mps_allocated_bytes"],
                digest,
                utc(),
            )
        )

        updated = db.execute(
            """
            UPDATE tasks
            SET state='completed',
                claim_token=NULL,
                finished_utc=?,
                result_path=?,
                result_sha256=?,
                last_error=NULL
            WHERE task_id=?
              AND state='running'
              AND claim_token=?
            """,
            (
                utc(),
                str(path.resolve()),
                digest,
                task["task_id"],
                task["token"],
            )
        )
        assert updated.rowcount == 1

        db.execute(
            """
            UPDATE attempts
            SET state='completed',
                finished_utc=?,
                message=?
            WHERE attempt_id=?
            """,
            (utc(), str(path), task["token"])
        )
        db.commit()
    except BaseException:
        db.rollback()
        raise

def failure(db, task, message):
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute(
            """
            UPDATE tasks
            SET state='failed',
                claim_token=NULL,
                finished_utc=?,
                last_error=?
            WHERE task_id=?
              AND state='running'
              AND claim_token=?
            """,
            (
                utc(), message[-3000:],
                task["task_id"], task["token"]
            )
        )
        db.execute(
            """
            UPDATE attempts
            SET state='failed',
                finished_utc=?,
                message=?
            WHERE attempt_id=?
            """,
            (
                utc(), message[-3000:],
                task["token"]
            )
        )
        db.commit()
    except BaseException:
        db.rollback()
        raise

def worker(task, output):
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

    name = task["model"]
    bits = task["group_mapping"]
    threads = task["cpu_threads"]
    batch = task["batch_size"]
    micro = task["microbatch_size"]

    assert batch % micro == 0
    assert name in MODELS

    torch.set_num_threads(threads)
    torch.manual_seed(2026)

    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=None).eval()
    elif name == "convnext_base":
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        model = convnext_base(weights=None).eval()
    else:
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
        model = convnext_large(weights=None).eval()

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    assert checkpoint.is_file(), (
        f"Cached weights missing: {checkpoint}"
    )
    assert sha(checkpoint) == MODELS[name][1]

    metadata = json.loads(T05.read_text())
    assert metadata["models"][name][
        "weight_sha256"
    ] == MODELS[name][1]

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    del state

    manifest_path = MANIFESTS / MODELS[name][0]
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest["groups"]) == len(bits)

    records = json.loads(HOLDOUT.read_text())
    assert len(records) >= batch

    images = []
    image_shas = []

    for item in records[:batch]:
        path = IMAGE_ROOT / item["path"]
        assert sha(path) == item["sha256"]

        with Image.open(path) as image:
            x = weights.transforms()(
                image.convert("RGB")
            )

        images.append(x)
        image_shas.append(item["sha256"])

    inputs = torch.stack(images)

    with torch.inference_mode():
        reference = model(inputs).detach()

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
            nn.Sequential(*[
                model.get_submodule(module)
                for module in group["modules"]
            ])
            for group in manifest["groups"]
        ]

    assert len(stages) == len(bits)

    devices = [
        "cpu" if bit == "C" else "mps"
        for bit in bits
    ]

    uses_mps = "mps" in devices

    for stage, device in zip(stages, devices):
        stage.to(device)

    if uses_mps:
        torch.mps.synchronize()

    def forward():
        chunks = []

        for start in range(0, batch, micro):
            x = inputs[start:start + micro]

            for stage, device in zip(stages, devices):
                if x.device.type != device:
                    x = x.to(device)
                x = stage(x)

            if x.device.type != "cpu":
                x = x.to("cpu")

            chunks.append(x)

        if uses_mps:
            torch.mps.synchronize()

        return torch.cat(chunks)

    with torch.inference_mode():
        actual = forward()

    numerical = bool(torch.allclose(
        actual, reference,
        rtol=1e-3, atol=1e-3
    ))

    agreement = bool(torch.equal(
        actual.argmax(dim=1),
        reference.argmax(dim=1)
    ))

    max_error = float(
        (actual - reference).abs().max().item()
    )

    # Preserve incorrect outputs, but do not label
    # them successful benchmark measurements.
    if not numerical or not agreement:
        raise RuntimeError(
            "NUMERICAL_MISMATCH "
            f"allclose={numerical} "
            f"top1={agreement} "
            f"max_abs_error={max_error}"
        )

    with torch.inference_mode():
        for _ in range(WARMUPS):
            forward()

    durations = []

    with torch.inference_mode():
        for _ in range(TIMED):
            if uses_mps:
                torch.mps.synchronize()

            start = time.perf_counter_ns()
            forward()
            duration = time.perf_counter_ns() - start

            assert duration > 0
            durations.append(duration / 1e6)

    duration_s = statistics.median(durations) / 1000
    throughput = batch / duration_s

    assert math.isfinite(throughput) and throughput > 0

    try:
        rss = int(subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True
        ).strip()) * 1024
    except Exception:
        rss = None

    mps_bytes = None
    if uses_mps:
        mps_bytes = int(
            torch.mps.current_allocated_memory()
        )

    sorted_ms = sorted(durations)
    p95_position = .95 * (len(sorted_ms) - 1)
    low = int(p95_position)
    high = min(low + 1, len(sorted_ms) - 1)
    p95_ms = (
        sorted_ms[low] * (high - p95_position)
        + sorted_ms[high] * (p95_position - low)
    ) if high != low else sorted_ms[low]

    atomic_json(output, {
        "protocol": "T23_MAC_FP32_SEQUENTIAL_V1",
        "created_utc": utc(),
        "task_id": task["task_id"],
        "attempt_id": task["token"],
        "model": name,
        "model_weight_sha256": sha(checkpoint),
        "manifest_sha256": sha(manifest_path),
        "image_sha256_list": image_shas,
        "hardware_id": "mac_m4",
        "backend": "cpu_mps",
        "precision": "float32",
        "execution_mode": "sequential",
        "placement_mapping": bits,
        "cpu_threads": threads,
        "batch_size": batch,
        "microbatch_size": micro,
        "microbatch_count": batch // micro,
        "warmups": WARMUPS,
        "timed_rounds": TIMED,
        "timing_samples_ms": durations,
        "latency_median_ms": statistics.median(durations),
        "latency_p95_ms": p95_ms,
        "throughput_images_s": throughput,
        "numerical_pass": numerical,
        "top1_agreement": agreement,
        "max_abs_error": max_error,
        "rss_bytes": rss,
        "mps_allocated_bytes": mps_bytes,
        "memory_counter_scope": "post_inference_snapshot",
        "power_measured": False,
        "energy_measured": False,
        "qos_measured": False,
        "hardware_kernel_overlap_proven": False,
        "torch_version": torch.__version__,
    })

    print(
        f"RESULT {name} {bits[:12]} "
        f"batch={batch} micro={micro} "
        f"threads={threads} "
        f"throughput={throughput:.3f} images/s "
        f"median={statistics.median(durations):.3f} ms",
        flush=True
    )

def run_campaign(limit):
    with (ROOT / ".campaign.lock").open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )

        if not DB.exists():
            prepare()

        db = connect()

        try:
            assert db.execute(
                "PRAGMA quick_check"
            ).fetchone()[0] == "ok"

            recovered = resume(db)
            print(
                "RECOVERED INTERRUPTED TASKS:",
                recovered,
                flush=True
            )

            print(
                "T23 MAC PRODUCTION SESSION START",
                "limit=", limit,
                "free_gib=", round(free_space(), 2),
                flush=True
            )

            stop = False

            for index in range(limit):
                free_space()
                task = claim(db)

                if task is None:
                    print("CAMPAIGN QUEUE FINISHED")
                    break

                folder = ROOT / "results" / task["task_id"]
                folder.mkdir(parents=True, exist_ok=True)
                attempt_folder = (
                    folder / ("attempt_" + task["token"])
                )
                attempt_folder.mkdir()

                result = attempt_folder / "result.json"
                stdout_path = attempt_folder / "stdout.txt"
                stderr_path = attempt_folder / "stderr.txt"
                task_path = attempt_folder / "task.json"
                atomic_json(task_path, task)

                print(
                    f"[{index+1}/{limit}] RUN",
                    task["model"],
                    task["group_mapping"][:16],
                    f"batch={task['batch_size']}",
                    f"micro={task['microbatch_size']}",
                    f"threads={task['cpu_threads']}",
                    flush=True,
                )

                with stdout_path.open("w") as out, \
                     stderr_path.open("w") as err:
                    proc = subprocess.Popen(
                        [
                            sys.executable, "-u",
                            str(Path(__file__).resolve()),
                            "worker",
                            str(task_path),
                            str(result),
                        ],
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                            "TORCH_HOME": os.environ.get(
                                "TORCH_HOME",
                                str(Path.home() / ".cache/torch")
                            ),
                        },
                    )

                    try:
                        code = proc.wait(timeout=TIMEOUT)
                    except KeyboardInterrupt:
                        print(
                            "\nCTRL+C RECEIVED: stopping "
                            "current worker, preserving database.",
                            flush=True
                        )
                        os.killpg(proc.pid, signal.SIGINT)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        stop = True
                        break
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        code = 124

                if code == 0 and result.is_file():
                    try:
                        complete(db, task, result)
                        print(
                            "COMMITTED:",
                            task["task_id"][:24],
                            stdout_path.read_text(
                                errors="replace"
                            ).strip()[-240:],
                            flush=True
                        )
                    except BaseException:
                        traceback.print_exc()
                        raise
                else:
                    error = (
                        f"exit={code}\n"
                        + stderr_path.read_text(
                            errors="replace"
                        )[-2600:]
                    )
                    failure(db, task, error)

                    print(
                        "FAILED RECORDED:",
                        task["model"],
                        f"exit={code}",
                        error[-400:],
                        flush=True
                    )

            state_rows = db.execute(
                """
                SELECT t.state, COUNT(*) AS n
                FROM t23_campaign_tasks c
                JOIN tasks t ON t.task_id=c.task_id
                GROUP BY t.state
                """
            ).fetchall()

            states = {
                r["state"]: r["n"] for r in state_rows
            }

            print("\n=== T23 MAC PRODUCTION STATUS ===")
            print("CAMPAIGN TASK STATES:", states)
            print(
                "SUCCESSFUL NEW MEASUREMENTS:",
                db.execute(
                    "SELECT COUNT(*) FROM t23_measurements"
                ).fetchone()[0]
            )
            print(
                "DISK FREE GiB:",
                round(free_space(), 2)
            )
            print("SOURCE SQLITE MODIFIED: NO")
            print("CAMPAIGN SQLITE:", DB)
            print("CAMPAIGN DIRECTORY:", ROOT)
            print(
                "SESSION:",
                "INTERRUPTED_RESUMABLE"
                if stop else "FINISHED_OR_LIMIT_REACHED"
            )
        finally:
            db.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "run", "worker")
    )
    parser.add_argument("extra", nargs="*")
    parser.add_argument("--limit", type=int, default=90)
    args = parser.parse_args()

    if args.command == "worker":
        assert len(args.extra) == 2
        task = json.loads(Path(args.extra[0]).read_text())
        worker(task, Path(args.extra[1]))
        return

    if args.command == "prepare":
        prepare()
        return

    if not DB.exists():
        prepare()

    assert args.limit >= 1
    run_campaign(args.limit)

if __name__ == "__main__":
    main()
