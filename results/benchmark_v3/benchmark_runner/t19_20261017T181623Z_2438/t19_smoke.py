import csv
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()

DB = BASE / (
    "dataset_store/"
    "t18_20261017T174740Z_2438/"
    "dataset_v3.sqlite"
)

T17 = BASE / (
    "dataset_v3_design/"
    "t17_20261017T160851Z_2438"
)

T05 = BASE / (
    "validation/"
    "t05_20261015T173344Z_2438/"
    "metadata.json"
)

IMAGE_LIST = BASE / (
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

MANIFEST_NAMES = {
    "resnet50": "resnet50__stage6.json",
    "convnext_base": "convnext_base__fine41.json",
}

SPECIFICATIONS = [
    ("resnet50", "CCCCCC"),
    ("resnet50", "AAAAAA"),
    ("resnet50", "AAAAAC"),
    ("resnet50", "CACACA"),
    ("convnext_base", "A" * 41),
    ("convnext_base", "A" * 40 + "C"),
]

WARMUPS = 3
TIMED = 8
TIMEOUT_SECONDS = 240

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )
    temp = path.with_name(
        path.name + ".tmp"
    )
    with temp.open("w", encoding="utf-8") as f:
        json.dump(
            obj, f, indent=2,
            ensure_ascii=False
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def conn_open():
    assert DB.is_file(), DB
    conn = sqlite3.connect(
        str(DB),
        timeout=30,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

@contextmanager
def single_runner():
    lockfile = DB.with_name(
        "dataset_v3.runner.lock"
    )
    with lockfile.open("a+") as handle:
        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )
        try:
            yield
        finally:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_UN
            )

def verify_sources():
    checks = json.loads(
        (T17 / "checksums.json").read_text()
    )
    for name, expected in checks.items():
        assert sha(T17 / name) == expected

    assert T05.is_file()
    assert IMAGE_LIST.is_file()

    conn = conn_open()
    try:
        assert conn.execute(
            "PRAGMA quick_check"
        ).fetchone()[0] == "ok"

        metadata = dict(
            conn.execute(
                "SELECT key,value FROM metadata"
            ).fetchall()
        )

        assert metadata[
            "t17_placement_sha256"
        ] == sha(T17 / "placement_catalog.csv")

        assert metadata[
            "t17_workload_sha256"
        ] == sha(T17 / "workload_catalog.csv")

        assert conn.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0] == 327336
    finally:
        conn.close()

def manifest_load():
    path = ROOT / "smoke_manifest.json"
    assert path.is_file(), (
        "Önce prepare çalıştırılmalı"
    )
    manifest = json.loads(path.read_text())
    assert manifest["protocol"] == (
        "T19_sequential_smoke_v1"
    )
    assert len(manifest["tasks"]) == 6
    assert len({
        t["task_id"] for t in manifest["tasks"]
    }) == 6
    return manifest

def prepare():
    verify_sources()

    target = ROOT / "smoke_manifest.json"
    assert not target.exists(), (
        "Manifest zaten var. "
        "Tekrar prepare yapmadan run kullan."
    )

    conn = conn_open()
    selected = []

    try:
        conn.execute("BEGIN IMMEDIATE")

        for model, bits in SPECIFICATIONS:
            rows = conn.execute(
                """
                SELECT t.task_id,t.state,
                       p.placement_id,
                       p.transition_count
                FROM tasks t
                JOIN placements p
                  ON t.placement_id=p.placement_id
                WHERE p.model=?
                  AND p.group_mapping=?
                  AND t.workload_id='batch1_micro1'
                  AND t.cpu_threads=4
                  AND t.execution_mode='sequential'
                  AND t.precision='float32'
                  AND t.hardware_id='mac_m4'
                """,
                (model, bits)
            ).fetchall()

            assert len(rows) == 1, (
                model, bits, len(rows)
            )

            row = rows[0]
            assert row["state"] == "planned", (
                "Konfigürasyon önceden çalıştırılmış: "
                + row["task_id"]
            )

            updated = conn.execute(
                """
                UPDATE tasks SET state='queued'
                WHERE task_id=?
                  AND state='planned'
                """,
                (row["task_id"],)
            )
            assert updated.rowcount == 1

            selected.append({
                "task_id": row["task_id"],
                "placement_id": row["placement_id"],
                "model": model,
                "group_mapping": bits,
                "transition_count":
                    row["transition_count"],
            })

        conn.commit()

        atomic_json(target, {
            "protocol": "T19_sequential_smoke_v1",
            "created_utc": utc(),
            "database": str(DB),
            "t17_catalog_sha256": sha(
                T17 / "placement_catalog.csv"
            ),
            "image_list_sha256": sha(IMAGE_LIST),
            "warmups": WARMUPS,
            "timed_calls": TIMED,
            "tasks": selected,
        })
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()

    print("T19 PILOT TASK SELECTION: PASS")
    print("QUEUED TASKS: 6")
    print("OTHER TASKS: 327330")
    print("MANIFEST:", target)

def task_from_db(conn, task_id):
    row = conn.execute(
        """
        SELECT t.*,p.model,p.group_mapping,
               p.manifest_sha256,
               p.transition_count,
               w.batch_size,w.microbatch_size
        FROM tasks t
        JOIN placements p
          ON t.placement_id=p.placement_id
        JOIN workloads w
          ON t.workload_id=w.workload_id
        WHERE t.task_id=?
        """,
        (task_id,)
    ).fetchone()

    assert row is not None
    return dict(row)

def recover(conn, manifest):
    # This is safe only with our exclusive runner lock:
    # no other T19 worker may still be active.
    recovered = 0
    conn.execute("BEGIN IMMEDIATE")

    try:
        for item in manifest["tasks"]:
            tid = item["task_id"]
            row = conn.execute(
                "SELECT state,claim_token "
                "FROM tasks WHERE task_id=?",
                (tid,)
            ).fetchone()

            if row["state"] != "running":
                continue

            token = row["claim_token"]
            assert token

            attempt = conn.execute(
                """
                UPDATE attempts
                SET state='interrupted',
                    finished_utc=?,
                    message='recovered_at_next_run'
                WHERE attempt_id=?
                  AND state='running'
                """,
                (utc(), token)
            )
            assert attempt.rowcount == 1

            result = conn.execute(
                """
                UPDATE tasks
                SET state='queued',
                    claim_token=NULL,
                    started_utc=NULL
                WHERE task_id=?
                  AND state='running'
                  AND claim_token=?
                """,
                (tid, token)
            )
            assert result.rowcount == 1
            recovered += 1

        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    return recovered

def claim(conn, tid):
    token = uuid.uuid4().hex

    conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.execute(
            """
            UPDATE tasks
            SET state='running',
                claim_token=?,
                started_utc=?,
                attempt_count=attempt_count+1
            WHERE task_id=?
              AND state='queued'
            """,
            (token, utc(), tid)
        )
        assert cursor.rowcount == 1

        conn.execute(
            """
            INSERT INTO attempts
            (attempt_id,task_id,state,started_utc)
            VALUES (?,?,'running',?)
            """,
            (token, tid, utc())
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    return token

def finish(conn, tid, token, result_path):
    data = json.loads(
        Path(result_path).read_text()
    )
    assert data["task_id"] == tid
    assert data["attempt_id"] == token
    assert data["numerical_pass"] is True
    assert data["top1_agreement"] is True

    digest = sha(result_path)
    conn.execute("BEGIN IMMEDIATE")

    try:
        cursor = conn.execute(
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
                utc(), str(Path(result_path).resolve()),
                digest, tid, token
            )
        )
        assert cursor.rowcount == 1

        cursor = conn.execute(
            """
            UPDATE attempts
            SET state='completed',
                finished_utc=?,
                message=?
            WHERE attempt_id=?
              AND state='running'
            """,
            (utc(), str(result_path), token)
        )
        assert cursor.rowcount == 1
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

def fail(conn, tid, token, message):
    conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.execute(
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
            (utc(), message[-3000:], tid, token)
        )
        assert cursor.rowcount == 1

        cursor = conn.execute(
            """
            UPDATE attempts
            SET state='failed',
                finished_utc=?,
                message=?
            WHERE attempt_id=?
              AND state='running'
            """,
            (utc(), message[-3000:], token)
        )
        assert cursor.rowcount == 1
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

def rss_bytes():
    output = subprocess.check_output(
        [
            "ps", "-o", "rss=",
            "-p", str(os.getpid())
        ],
        text=True
    )
    return int(output.strip()) * 1024

def worker(task_id, token, output_path):
    import torch
    from torch import nn
    from PIL import Image
    from torchvision.models import (
        resnet50,
        ResNet50_Weights,
        convnext_base,
        ConvNeXt_Base_Weights,
    )

    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()

    conn = conn_open()
    try:
        task = task_from_db(conn, task_id)
    finally:
        conn.close()

    assert task["state"] == "running"
    assert task["claim_token"] == token
    assert task["execution_mode"] == "sequential"
    assert task["cpu_threads"] == 4
    assert task["batch_size"] == 1
    assert task["microbatch_size"] == 1

    model_name = task["model"]
    bits = task["group_mapping"]

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    if model_name == "resnet50":
        weights = (
            ResNet50_Weights.IMAGENET1K_V2
        )
        model = resnet50(weights=None).eval()
    elif model_name == "convnext_base":
        weights = (
            ConvNeXt_Base_Weights.IMAGENET1K_V1
        )
        model = convnext_base(
            weights=None
        ).eval()
    else:
        raise ValueError(model_name)

    validation = json.loads(T05.read_text())
    holdout = json.loads(
        IMAGE_LIST.read_text()
    )
    assert len(holdout) == 16

    image_info = holdout[0]
    image_path = IMAGE_ROOT / image_info["path"]
    assert sha(image_path) == (
        image_info["sha256"]
    )

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    expected_weight = validation[
        "models"
    ][model_name]["weight_sha256"]

    assert checkpoint.is_file(), (
        f"Cached checkpoint missing: {checkpoint}"
    )
    assert sha(checkpoint) == expected_weight

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        state, strict=True
    )
    del state

    manifest_path = (
        MANIFESTS / MANIFEST_NAMES[model_name]
    )
    assert sha(manifest_path) == (
        task["manifest_sha256"]
    )
    manifest = json.loads(
        manifest_path.read_text()
    )
    assert manifest["group_count"] == len(bits)

    if model_name == "resnet50":
        stages = [
            nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
            ),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.Sequential(
                model.avgpool,
                nn.Flatten(1),
                model.fc,
            ),
        ]
    else:
        stages = []

        for group in manifest["groups"]:
            modules = [
                model.get_submodule(path)
                for path in group["modules"]
            ]
            stages.append(
                nn.Sequential(*modules)
            )

    assert len(stages) == len(bits)

    with Image.open(image_path) as image:
        x_cpu = weights.transforms()(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tuple(x_cpu.shape) == (
        1, 3, 224, 224
    )

    with torch.inference_mode():
        reference = model(
            x_cpu
        ).detach().clone()

    assert tuple(reference.shape) == (
        1, 1000
    )

    devices = [
        "cpu" if bit == "C"
        else "mps"
        for bit in bits
    ]
    uses_mps = "mps" in devices

    # Weight placement is excluded from timing.
    for stage, device in zip(
        stages, devices
    ):
        stage.to(device)

    if uses_mps:
        torch.mps.synchronize()

    def forward(record=False):
        y = x_cpu
        events = []

        for index, (
            stage, device
        ) in enumerate(
            zip(stages, devices)
        ):
            if y.device.type != device:
                if record:
                    events.append({
                        "boundary_after_group":
                            index - 1,
                        "source": y.device.type,
                        "destination": device,
                        "shape": list(y.shape),
                        "dtype": str(y.dtype),
                        "logical_bytes":
                            y.numel()
                            * y.element_size(),
                    })

                y = y.to(device)

            y = stage(y)

        if y.device.type != "cpu":
            if record:
                events.append({
                    "boundary_after_group":
                        len(stages) - 1,
                    "source": "mps",
                    "destination": "cpu",
                    "shape": list(y.shape),
                    "dtype": str(y.dtype),
                    "logical_bytes":
                        y.numel()
                        * y.element_size(),
                })

            y = y.cpu()

        if uses_mps:
            torch.mps.synchronize()

        return y, events

    with torch.inference_mode():
        actual, transfers = forward(True)

    numerical_pass = bool(
        torch.allclose(
            reference, actual,
            atol=1e-3, rtol=1e-3
        )
    )
    top1_agreement = (
        reference.argmax().item()
        == actual.argmax().item()
    )

    max_error = float(
        (reference - actual)
        .abs().max().item()
    )

    assert numerical_pass
    assert top1_agreement

    del actual

    with torch.inference_mode():
        for _ in range(WARMUPS):
            output, _ = forward()
            del output

    samples_ns = []

    with torch.inference_mode():
        for _ in range(TIMED):
            if uses_mps:
                torch.mps.synchronize()

            start = time.perf_counter_ns()
            output, _ = forward()
            elapsed = (
                time.perf_counter_ns()
                - start
            )
            del output

            assert elapsed > 0
            samples_ns.append(elapsed)

    samples_ms = [
        n / 1e6 for n in samples_ns
    ]
    assert all(
        math.isfinite(x) and x > 0
        for x in samples_ms
    )

    result = {
        "protocol": "T19_sequential_smoke_v1",
        "task_id": task_id,
        "attempt_id": token,
        "created_utc": utc(),
        "model": model_name,
        "placement_id": task["placement_id"],
        "group_mapping": bits,
        "cpu_threads": 4,
        "execution_mode": "sequential",
        "backend": "mps",
        "precision": "float32",
        "batch_size": 1,
        "microbatch_size": 1,
        "image_sha256": image_info["sha256"],
        "weight_sha256": expected_weight,
        "manifest_sha256":
            task["manifest_sha256"],
        "numerical_pass": numerical_pass,
        "top1_agreement": top1_agreement,
        "max_abs_error": max_error,
        "warmups": WARMUPS,
        "timed_calls": TIMED,
        "timings_ns": samples_ns,
        "median_latency_ms":
            statistics.median(samples_ms),
        "mean_latency_ms":
            statistics.mean(samples_ms),
        "min_latency_ms": min(samples_ms),
        "max_latency_ms": max(samples_ms),
        "transfer_events": transfers,
        "transfer_event_count":
            len(transfers),
        "logical_transfer_bytes": sum(
            item["logical_bytes"]
            for item in transfers
        ),
        "individual_transfer_latency_status":
            "not_measured",
        "ground_truth_accuracy_status":
            "not_measured",
        "steady_state_throughput_status":
            "not_measured",
        "pipeline_overlap_status":
            "not_measured",
        "power_status": "not_measured",
        "energy_status": "not_measured",
        "rss_after_inference_bytes":
            rss_bytes(),
        "runtime": {
            "torch": torch.__version__,
            "mps_fallback": "0",
            "cpu_input_output": True,
            "weight_placement_timed": False,
            "preprocessing_timed": False,
            "device_transfers_in_e2e_timing": True,
            "true_peak_memory_measured": False,
        },
    }

    atomic_json(output_path, result)

    print(
        f"WORKER PASS | {model_name} "
        f"{bits[:12]} | "
        f"median="
        f"{result['median_latency_ms']:.3f} ms "
        f"| transfers={len(transfers)}",
        flush=True
    )

def run_jobs(limit):
    assert limit >= 1
    manifest = manifest_load()
    verify_sources()

    with single_runner():
        conn = conn_open()
        failures = 0
        executed = 0

        try:
            recovered = recover(
                conn, manifest
            )
            print(
                "INTERRUPTED ATTEMPTS RECOVERED:",
                recovered,
                flush=True
            )

            for item in manifest["tasks"]:
                if executed >= limit:
                    break

                tid = item["task_id"]
                task = task_from_db(conn, tid)

                if task["state"] == "completed":
                    continue

                if task["state"] == "failed":
                    print(
                        "SKIPPING PREVIOUS FAILED TASK:",
                        item["model"],
                        item["group_mapping"][:12],
                        flush=True
                    )
                    continue

                assert task["state"] == "queued", (
                    tid, task["state"]
                )

                token = claim(conn, tid)
                output_path = (
                    ROOT / "results" / tid
                    / (token + ".json")
                )

                print(
                    f"\nSTART {executed+1}/{limit}: "
                    f"{item['model']} "
                    f"{item['group_mapping'][:12]} "
                    f"transitions="
                    f"{item['transition_count']}",
                    flush=True
                )

                try:
                    child = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            str(Path(__file__).resolve()),
                            "worker",
                            tid,
                            token,
                            str(output_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=TIMEOUT_SECONDS,
                        env={
                            **os.environ,
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                        },
                    )

                    if child.stdout:
                        print(
                            child.stdout,
                            end="",
                            flush=True
                        )

                    if child.returncode != 0:
                        raise RuntimeError(
                            child.stderr
                            or (
                                "Worker exit code: "
                                + str(child.returncode)
                            )
                        )

                    assert output_path.is_file()
                    finish(
                        conn, tid, token,
                        output_path
                    )

                    print(
                        "RESULT COMMITTED:",
                        output_path.name,
                        flush=True
                    )

                except subprocess.TimeoutExpired:
                    fail(
                        conn, tid, token,
                        "timeout_after_240_seconds"
                    )
                    failures += 1
                    print(
                        "TASK TIMEOUT: recorded",
                        flush=True
                    )

                except Exception as error:
                    fail(
                        conn, tid, token,
                        repr(error)
                    )
                    failures += 1
                    print(
                        "TASK FAILURE:",
                        repr(error),
                        flush=True
                    )

                executed += 1

        finally:
            conn.close()

        print(
            "\nRUN ATTEMPTS:", executed,
            "FAILURES:", failures
        )

        if failures:
            raise RuntimeError(
                "Some smoke tests failed. "
                "Records are preserved."
            )

def status():
    manifest = manifest_load()
    conn = conn_open()
    counts = {}
    completed = 0

    try:
        assert conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        for item in manifest["tasks"]:
            task = task_from_db(
                conn, item["task_id"]
            )
            state = task["state"]
            counts[state] = (
                counts.get(state, 0) + 1
            )

            if state == "completed":
                path = Path(
                    task["result_path"]
                )
                assert path.is_file()
                assert sha(path) == (
                    task["result_sha256"]
                )

                data = json.loads(
                    path.read_text()
                )
                assert data["task_id"] == (
                    item["task_id"]
                )
                assert data["numerical_pass"]
                assert data["top1_agreement"]
                assert len(
                    data["timings_ns"]
                ) == TIMED

                completed += 1

                print(
                    f"PASS: {item['model']:<15} "
                    f"transitions="
                    f"{item['transition_count']} "
                    f"median="
                    f"{data['median_latency_ms']:.3f} ms"
                )

        all_states = dict(
            conn.execute(
                "SELECT state,COUNT(*) "
                "FROM tasks GROUP BY state"
            ).fetchall()
        )

        assert sum(
            all_states.values()
        ) == 327336
    finally:
        conn.close()

    print("\n=== T19 QUEUE STATUS ===")
    print("SMOKE STATES:", counts)
    print("GLOBAL TASK STATES:", all_states)
    print(
        "COMPLETED ARTIFACT HASHES:",
        f"{completed}/{completed} PASS"
    )
    print(
        "T19 SMOKE COMPLETED:",
        f"{completed}/6"
    )

    if completed == 6:
        print(
            "T19 SEQUENTIAL SMOKE: PASS"
        )

def backup():
    manifest = manifest_load()
    destination = (
        ROOT / "backups"
        / "dataset_v3_after_smoke.sqlite"
    )
    assert not destination.exists()
    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    with single_runner():
        source_conn = sqlite3.connect(
            str(DB)
        )
        temp = destination.with_name(
            destination.name + ".tmp"
        )

        target_conn = sqlite3.connect(
            str(temp)
        )
        try:
            source_conn.backup(
                target_conn,
                pages=1000,
                sleep=0.05,
            )

            assert target_conn.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"

            assert target_conn.execute(
                "SELECT COUNT(*) FROM tasks"
            ).fetchone()[0] == 327336
        finally:
            target_conn.close()
            source_conn.close()

        os.replace(temp, destination)

        atomic_json(
            destination.with_suffix(
                ".sqlite.sha256.json"
            ),
            {
                "backup_sha256":
                    sha(destination),
                "bytes":
                    destination.stat().st_size,
                "created_utc": utc(),
                "task_count": 327336,
                "manifest_sha256":
                    sha(
                        ROOT / "smoke_manifest.json"
                    ),
                "manifest_tasks":
                    len(manifest["tasks"]),
            }
        )

    print("T19 SQLITE BACKUP: PASS")
    print("BACKUP:", destination)

def main():
    assert len(sys.argv) >= 2
    action = sys.argv[1]

    if action == "prepare":
        with single_runner():
            prepare()
    elif action == "run":
        limit = (
            int(sys.argv[2])
            if len(sys.argv) > 2
            else 6
        )
        run_jobs(limit)
    elif action == "status":
        status()
    elif action == "backup":
        backup()
    elif action == "worker":
        assert len(sys.argv) == 5
        worker(
            sys.argv[2],
            sys.argv[3],
            Path(sys.argv[4]).resolve(),
        )
    else:
        raise ValueError(action)

if __name__ == "__main__":
    main()
