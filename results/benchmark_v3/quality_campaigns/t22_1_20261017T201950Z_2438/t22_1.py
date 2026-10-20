import csv
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()

SOURCE_DB = BASE / (
    "pipeline_recovery/"
    "t21_1_20261017T201021Z_2438/"
    "dataset_v3_with_pipeline.sqlite"
)
INVENTORY_ROOT = BASE / (
    "model_integration/"
    "t22_20261017T201408Z_2438"
)
INVENTORY = INVENTORY_ROOT / "model_inventory.json"
IMAGE_MANIFEST = INVENTORY_ROOT / "imagenetv2_manifest.jsonl"
NEW_DB = ROOT / "dataset_v3_with_quality.sqlite"

MODELS = ("resnet50", "convnext_base", "convnext_large")
PRECISIONS = ("fp32", "fp16")
JOBS = [
    (model, precision)
    for model in MODELS
    for precision in PRECISIONS
]

BATCH_SIZE = 8
MIN_FREE_GIB = 4

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

def save_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(
            obj, f, indent=2, ensure_ascii=False
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)

def connect(path):
    db = sqlite3.connect(str(path), timeout=90)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=90000")
    db.execute("PRAGMA synchronous=FULL")
    return db

def assert_disk():
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE_GIB * 1024**3:
        raise RuntimeError(
            f"Disk guard: less than {MIN_FREE_GIB} GiB free"
        )

def load_inputs_manifest():
    inventory = json.loads(INVENTORY.read_text())

    assert inventory["imagenetv2"]["images"] == 10000
    assert inventory["imagenetv2"][
        "expected_10000_image_protocol_ready"
    ] is True

    assert sha(IMAGE_MANIFEST) == inventory[
        "imagenetv2"
    ]["manifest_sha256"]

    records = [
        json.loads(line)
        for line in IMAGE_MANIFEST.read_text().splitlines()
        if line.strip()
    ]

    assert len(records) == 10000
    paths = [r["path"] for r in records]
    assert len(set(paths)) == 10000

    labels = {}

    for record in records:
        label = record["label"]
        assert type(label) is int
        assert 0 <= label < 1000
        labels[label] = labels.get(label, 0) + 1

    assert len(labels) == 1000
    assert all(value == 10 for value in labels.values())

    return inventory, records

def verify_source():
    assert SOURCE_DB.is_file()
    assert INVENTORY.is_file()
    assert IMAGE_MANIFEST.is_file()

    inv, _ = load_inputs_manifest()

    c = sqlite3.connect(
        f"file:{SOURCE_DB}?mode=ro",
        uri=True
    )

    try:
        assert c.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        assert c.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0] == 327336

        assert c.execute(
            "SELECT COUNT(*) FROM energy_measurements"
        ).fetchone()[0] == 18

        assert c.execute(
            "SELECT COUNT(*) FROM pipeline_measurements"
        ).fetchone()[0] == 24

        assert c.execute(
            "SELECT COUNT(*) FROM tasks "
            "WHERE state='completed'"
        ).fetchone()[0] == 6

    finally:
        c.close()

    by_name = {
        r["model"]: r
        for r in inv["vision_checkpoints"]
    }

    assert set(by_name) == set(MODELS)

    for name in MODELS:
        r = by_name[name]
        assert r["matches_t05"] is True
        assert Path(r["checkpoint"]).is_file()
        assert sha(Path(r["checkpoint"])) == (
            r["checkpoint_sha256"]
        )

    return inv

def initialize():
    assert_disk()
    inv = verify_source()

    plan = {
        "protocol": "T22_1_full_imagenetv2_quality_v1",
        "source_database_sha256": sha(SOURCE_DB),
        "model_inventory_sha256": sha(INVENTORY),
        "image_manifest_sha256": sha(IMAGE_MANIFEST),
        "runner_sha256": sha(Path(__file__)),
        "models": list(MODELS),
        "precisions": list(PRECISIONS),
        "images_per_job": 10000,
        "batch_size": BATCH_SIZE,
        "device": "mps",
        "precision_mode": "model_and_input_direct_dtype",
        "checkpoint_downloads": False,
        "number_of_jobs": 6,
    }

    plan_path = ROOT / "campaign_plan.json"

    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, (
            "Resume refused: plan or source changed."
        )
    else:
        save_json(plan_path, plan)

    if NEW_DB.exists():
        db = connect(NEW_DB)
        try:
            assert db.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"
            assert db.execute(
                "SELECT COUNT(*) FROM pipeline_measurements"
            ).fetchone()[0] == 24

            stored = dict(db.execute(
                "SELECT key,value FROM vision_quality_meta"
            ).fetchall())

            assert stored == {
                "protocol":
                    "T22_1_full_imagenetv2_quality_v1",
                "image_manifest_sha256": sha(IMAGE_MANIFEST),
                "model_inventory_sha256": sha(INVENTORY),
            }

        finally:
            db.close()

        return inv

    temp = ROOT / (
        "database_build_" + uuid.uuid4().hex + ".tmp"
    )

    source = sqlite3.connect(
        f"file:{SOURCE_DB}?mode=ro",
        uri=True
    )
    destination = sqlite3.connect(str(temp))

    try:
        source.backup(destination)

        destination.executescript("""
            CREATE TABLE vision_quality_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE vision_quality_jobs (
                model TEXT NOT NULL,
                precision TEXT NOT NULL,
                status TEXT NOT NULL,
                expected_images INTEGER NOT NULL,
                completed_images INTEGER NOT NULL DEFAULT 0,
                correct_top1 INTEGER NOT NULL DEFAULT 0,
                correct_top5 INTEGER NOT NULL DEFAULT 0,
                checkpoint_sha256 TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                PRIMARY KEY(model,precision)
            );

            CREATE TABLE vision_quality_predictions (
                model TEXT NOT NULL,
                precision TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                ground_truth INTEGER NOT NULL,
                predicted_top1 INTEGER NOT NULL,
                predicted_top5_json TEXT NOT NULL,
                correct_top1 INTEGER NOT NULL,
                correct_top5 INTEGER NOT NULL,
                provenance TEXT NOT NULL,
                PRIMARY KEY(model,precision,image_path),
                FOREIGN KEY(model,precision)
                    REFERENCES vision_quality_jobs(model,precision)
            );
        """)

        destination.executemany(
            "INSERT INTO vision_quality_meta VALUES (?,?)",
            [
                (
                    "protocol",
                    "T22_1_full_imagenetv2_quality_v1"
                ),
                (
                    "image_manifest_sha256",
                    sha(IMAGE_MANIFEST)
                ),
                (
                    "model_inventory_sha256",
                    sha(INVENTORY)
                ),
            ]
        )

        destination.commit()

        assert destination.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

    finally:
        source.close()
        destination.close()

    os.replace(temp, NEW_DB)
    return inv

def worker(model_name, precision):
    import torch
    from PIL import Image
    from torchvision.models import (
        resnet50, convnext_base, convnext_large,
        ResNet50_Weights,
        ConvNeXt_Base_Weights,
        ConvNeXt_Large_Weights,
    )

    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"

    assert torch.backends.mps.is_available()

    factories = {
        "resnet50": (
            resnet50,
            ResNet50_Weights.IMAGENET1K_V2,
        ),
        "convnext_base": (
            convnext_base,
            ConvNeXt_Base_Weights.IMAGENET1K_V1,
        ),
        "convnext_large": (
            convnext_large,
            ConvNeXt_Large_Weights.IMAGENET1K_V1,
        ),
    }

    assert model_name in factories
    assert precision in PRECISIONS

    stop = {"requested": False}

    def on_stop(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    inventory, records = load_inputs_manifest()
    image_root = Path(
        inventory["imagenetv2"]["root"]
    )

    entry = next(
        x for x in inventory["vision_checkpoints"]
        if x["model"] == model_name
    )

    checkpoint = Path(entry["checkpoint"])

    assert sha(checkpoint) == entry["checkpoint_sha256"]
    assert entry["matches_t05"] is True

    factory, weights = factories[model_name]
    model = factory(weights=None).eval()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state, strict=True)
    del state

    dtype = (
        torch.float32
        if precision == "fp32"
        else torch.float16
    )

    model = model.to(
        device="mps",
        dtype=dtype
    ).eval()

    transform = weights.transforms()

    db = connect(NEW_DB)

    try:
        current = db.execute(
            """
            SELECT checkpoint_sha256
            FROM vision_quality_jobs
            WHERE model=? AND precision=?
            """,
            (model_name, precision)
        ).fetchone()

        if current is not None:
            assert current[0] == entry["checkpoint_sha256"]

        db.execute(
            """
            INSERT INTO vision_quality_jobs (
                model,precision,status,expected_images,
                checkpoint_sha256,updated_utc
            ) VALUES (?,?,?,?,?,?)
            ON CONFLICT(model,precision) DO UPDATE SET
                status='running',
                updated_utc=excluded.updated_utc
            """,
            (
                model_name, precision, "running",
                10000, entry["checkpoint_sha256"], utc()
            )
        )
        db.commit()

        existing = set(
            row[0]
            for row in db.execute(
                """
                SELECT image_path
                FROM vision_quality_predictions
                WHERE model=? AND precision=?
                """,
                (model_name, precision)
            )
        )

        pending = [
            row for row in records
            if row["path"] not in existing
        ]

        print(
            f"WORKER {model_name}/{precision}: "
            f"existing={len(existing)} "
            f"pending={len(pending)}",
            flush=True,
        )

        for offset in range(0, len(pending), BATCH_SIZE):
            if stop["requested"]:
                db.execute(
                    """
                    UPDATE vision_quality_jobs
                    SET status='interrupted',updated_utc=?
                    WHERE model=? AND precision=?
                    """,
                    (utc(), model_name, precision)
                )
                db.commit()
                print(
                    "WORKER INTERRUPTED — completed rows saved",
                    flush=True,
                )
                return 130

            assert_disk()

            chunk = pending[offset:offset + BATCH_SIZE]
            tensors = []

            for record in chunk:
                path = image_root / record["path"]

                assert path.is_file()
                assert sha(path) == record["sha256"]

                with Image.open(path) as image:
                    tensors.append(
                        transform(image.convert("RGB"))
                    )

            batch = torch.stack(tensors).to(
                device="mps",
                dtype=dtype,
            )

            with torch.inference_mode():
                logits = model(batch)
                top5 = logits.float().topk(
                    5, dim=1
                ).indices.cpu().tolist()

            values = []

            for record, choices in zip(chunk, top5):
                label = record["label"]

                values.append((
                    model_name,
                    precision,
                    record["path"],
                    record["sha256"],
                    label,
                    choices[0],
                    json.dumps(choices),
                    int(choices[0] == label),
                    int(label in choices),
                    "v3_measured",
                ))

            with db:
                db.executemany(
                    """
                    INSERT INTO vision_quality_predictions
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    values,
                )

                db.execute(
                    """
                    UPDATE vision_quality_jobs
                    SET
                        completed_images=(
                            SELECT COUNT(*)
                            FROM vision_quality_predictions
                            WHERE model=? AND precision=?
                        ),
                        correct_top1=(
                            SELECT COALESCE(SUM(correct_top1),0)
                            FROM vision_quality_predictions
                            WHERE model=? AND precision=?
                        ),
                        correct_top5=(
                            SELECT COALESCE(SUM(correct_top5),0)
                            FROM vision_quality_predictions
                            WHERE model=? AND precision=?
                        ),
                        updated_utc=?
                    WHERE model=? AND precision=?
                    """,
                    (
                        model_name, precision,
                        model_name, precision,
                        model_name, precision,
                        utc(), model_name, precision
                    )
                )

            processed = len(existing) + offset + len(chunk)

            if (
                processed % 1000 < BATCH_SIZE
                or processed == 10000
            ):
                print(
                    f"PROGRESS {model_name}/{precision}: "
                    f"{processed}/10000",
                    flush=True,
                )

            del batch, logits

        result = db.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(correct_top1),0),
                   COALESCE(SUM(correct_top5),0)
            FROM vision_quality_predictions
            WHERE model=? AND precision=?
            """,
            (model_name, precision)
        ).fetchone()

        assert result[0] == 10000
        assert 0 <= result[1] <= result[2] <= 10000

        with db:
            db.execute(
                """
                UPDATE vision_quality_jobs
                SET status='completed',
                    completed_images=?,
                    correct_top1=?,
                    correct_top5=?,
                    updated_utc=?
                WHERE model=? AND precision=?
                """,
                (
                    result[0], result[1], result[2],
                    utc(), model_name, precision
                )
            )

        print(
            f"COMPLETED {model_name}/{precision}: "
            f"Top1={100*result[1]/10000:.2f}% "
            f"Top5={100*result[2]/10000:.2f}%",
            flush=True,
        )

        return 0

    finally:
        db.close()

def status():
    db = sqlite3.connect(
        f"file:{NEW_DB}?mode=ro",
        uri=True
    )
    try:
        rows = db.execute(
            """
            SELECT model,precision,status,
                   completed_images,
                   correct_top1,correct_top5
            FROM vision_quality_jobs
            """
        ).fetchall()
        return {
            (row[0], row[1]): row[2:]
            for row in rows
        }
    finally:
        db.close()

def finish_reports():
    db = sqlite3.connect(
        f"file:{NEW_DB}?mode=ro",
        uri=True
    )

    try:
        assert db.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        rows = db.execute(
            """
            SELECT model,precision,status,
                   completed_images,correct_top1,correct_top5
            FROM vision_quality_jobs
            ORDER BY model,precision
            """
        ).fetchall()

        results = []

        for model, precision, st, count, top1, top5 in rows:
            results.append({
                "model": model,
                "precision": precision,
                "status": st,
                "images": count,
                "top1_accuracy": (
                    top1 / count if count else None
                ),
                "top5_accuracy": (
                    top5 / count if count else None
                ),
            })

        complete = (
            len(results) == 6
            and all(
                r["status"] == "completed"
                and r["images"] == 10000
                for r in results
            )
        )

        flip_rows = []

        if complete:
            for model in MODELS:
                count, disagreements = db.execute(
                    """
                    SELECT COUNT(*),
                           SUM(CASE
                             WHEN a.predicted_top1 != b.predicted_top1
                             THEN 1 ELSE 0 END)
                    FROM vision_quality_predictions a
                    JOIN vision_quality_predictions b
                      ON a.model=b.model
                     AND a.image_path=b.image_path
                    WHERE a.model=?
                      AND a.precision='fp32'
                      AND b.precision='fp16'
                    """,
                    (model,)
                ).fetchone()

                assert count == 10000

                flip_rows.append({
                    "model": model,
                    "paired_images": count,
                    "fp32_fp16_top1_disagreements":
                        disagreements,
                    "top1_disagreement_fraction":
                        disagreements / count,
                })

        assert db.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0] == 327336
        assert db.execute(
            "SELECT COUNT(*) FROM pipeline_measurements"
        ).fetchone()[0] == 24
        assert db.execute(
            "SELECT COUNT(*) FROM energy_measurements"
        ).fetchone()[0] == 18

    finally:
        db.close()

    def output_csv(path, rows):
        if not rows:
            return
        temp = path.with_name(path.name + ".tmp")
        with temp.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0])
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)

    output_csv(
        ROOT / "vision_quality_summary.csv",
        results
    )
    output_csv(
        ROOT / "vision_precision_agreement.csv",
        flip_rows
    )

    report = {
        "protocol": "T22_1_full_imagenetv2_quality_v1",
        "complete": complete,
        "completed_jobs": sum(
            r["status"] == "completed"
            for r in results
        ),
        "target_jobs": 6,
        "target_predictions": 60000,
        "actual_predictions": sum(
            r["images"] for r in results
        ),
        "source_database_unmodified": True,
        "measurement_scope": "vision_quality_only",
        "throughput_from_this_campaign_valid": False,
        "precision_method":
            "FP32 or direct FP16 weights and inputs on MPS",
        "results": results,
        "precision_agreement": flip_rows,
    }

    save_json(ROOT / "quality_report.json", report)
    return report

def coordinator():
    import fcntl

    with (ROOT / ".runner.lock").open("a+") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            raise SystemExit(
                "Another T22.1 runner is already active."
            )

        initialize()

        control = {
            "stop": False,
            "child": None,
        }

        def on_stop(_signum, _frame):
            control["stop"] = True
            child = control["child"]

            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass

        signal.signal(signal.SIGINT, on_stop)
        signal.signal(signal.SIGTERM, on_stop)

        print("=== T22.1 FULL VISION QUALITY ===")
        print("MODELS: 3")
        print("PRECISIONS: FP32, FP16")
        print("DATASET: ImageNetV2 10000 images")
        print("TARGET PREDICTIONS: 60000")
        print("DEVICE: MPS")
        print("SOURCE SQLITE: READ-ONLY")
        print("AUTO-DOWNLOAD: DISABLED")
        print("CTRL+C: safe stop after current batch")
        print("RESULT DIRECTORY:", ROOT)
        print()

        for index, (model, precision) in enumerate(JOBS, 1):
            if control["stop"]:
                break

            current = status().get((model, precision))

            if current is not None and (
                current[0] == "completed"
                and current[1] == 10000
            ):
                print(
                    f"SKIP VERIFIED: {model}/{precision}",
                    flush=True,
                )
                continue

            assert_disk()

            attempt = ROOT / "logs" / (
                f"{model}_{precision}_{uuid.uuid4().hex}"
            )
            attempt.mkdir(parents=True)

            error_file = attempt / "stderr.txt"

            print(
                f"\nJOB {index}/6: "
                f"{model}/{precision}",
                flush=True,
            )

            with error_file.open("w") as errors:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "worker",
                        model,
                        precision,
                    ],
                    stderr=errors,
                    env={
                        **os.environ,
                        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                    },
                    start_new_session=True,
                )

                control["child"] = proc
                code = proc.wait()
                control["child"] = None

            if code != 0:
                print(
                    f"WORKER EXIT CODE: {code}",
                    flush=True,
                )

                if error_file.is_file():
                    lines = error_file.read_text(
                        errors="replace"
                    ).splitlines()

                    print("WORKER STDERR (last 20 lines):")
                    for line in lines[-20:]:
                        print(line)

                if code == 130 or control["stop"]:
                    print(
                        "INTERRUPTED — saved predictions retained",
                        flush=True,
                    )
                else:
                    print(
                        "JOB FAILED — raw log retained; "
                        "do not interpret partial data as completed",
                        flush=True,
                    )
                break

        report = finish_reports()

        print("\n=== T22.1 CAMPAIGN STATUS ===")
        print(
            "COMPLETED JOBS:",
            report["completed_jobs"], "/6"
        )
        print(
            "RECORDED IMAGE PREDICTIONS:",
            report["actual_predictions"],
            "/60000"
        )
        print("SOURCE DATABASE MODIFIED: NO")

        if report["complete"]:
            print("VISION QUALITY VALIDATION: PASS")
            print("T22.1 VISION QUALITY CAMPAIGN: PASS")
        else:
            print("STATUS: INCOMPLETE — RESUMABLE")

        print("NEW SQLITE:", NEW_DB)
        print("RESULT DIRECTORY:", ROOT)

if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        coordinator()

    elif (
        len(sys.argv) == 4
        and sys.argv[1] == "worker"
    ):
        raise SystemExit(worker(
            sys.argv[2], sys.argv[3]
        ))

    else:
        raise SystemExit(
            "Usage: python t22_1.py run"
        )
