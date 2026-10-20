import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()
BASE = Path("results/benchmark_v3").resolve()

DB = BASE / (
    "dataset_store/t18_20261017T174740Z_2438/"
    "dataset_v3.sqlite"
)
T17 = BASE / (
    "dataset_v3_design/t17_20261017T160851Z_2438"
)
T19 = BASE / (
    "benchmark_runner/t19_20261017T181623Z_2438"
)

T05 = BASE / (
    "validation/t05_20261015T173344Z_2438/"
    "metadata.json"
)
T12 = BASE / (
    "holdout_thread_confirmation/"
    "t12_20261015T192923Z_2438/"
    "holdout_images.json"
)
MANIFEST_DIR = Path(
    "configs/partition_manifests_v2"
).resolve()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, data):
    path.parent.mkdir(
        parents=True, exist_ok=True
    )
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=2,
            ensure_ascii=False
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

def copy_verified(source, relative, expected=None):
    source = Path(source)
    assert source.is_file(), str(source)

    original_sha = sha(source)
    if expected is not None:
        assert original_sha == expected, (
            f"Source SHA mismatch: {source}"
        )

    destination = OUT / relative
    destination.parent.mkdir(
        parents=True, exist_ok=True
    )
    assert not destination.exists()

    temporary = destination.with_name(
        destination.name + ".tmp"
    )
    assert not temporary.exists()

    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())

    os.replace(temporary, destination)
    assert sha(destination) == original_sha

    return {
        "relative_path": str(relative),
        "sha256": original_sha,
        "bytes": destination.stat().st_size,
    }

def check_sqlite(conn):
    assert conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    assert conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall() == []

    assert conn.execute(
        "SELECT COUNT(*) FROM tasks"
    ).fetchone()[0] == 327336

def main():
    assert DB.is_file()
    assert T17.is_dir()
    assert T19.is_dir()

    # Use the same lock as T19. Do not snapshot
    # while the T19 runner is active.
    lock_path = DB.with_name(
        "dataset_v3.runner.lock"
    )

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError:
            raise RuntimeError(
                "Benchmark runner active; "
                "stop it before archiving."
            )

        try:
            build_archive()
        finally:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_UN
            )

def build_archive():
    source = sqlite3.connect(str(DB))
    source.execute("PRAGMA foreign_keys=ON")

    check_sqlite(source)

    rows = source.execute(
        """
        SELECT task_id, result_path, result_sha256
        FROM tasks
        WHERE state='completed'
        ORDER BY task_id
        """
    ).fetchall()

    assert len(rows) == 6

    counts = dict(source.execute(
        "SELECT state,COUNT(*) FROM tasks GROUP BY state"
    ).fetchall())

    assert counts == {
        "completed": 6,
        "planned": 327330,
    }

    smoke_manifest = json.loads(
        (T19 / "smoke_manifest.json").read_text()
    )
    smoke_ids = {
        entry["task_id"]
        for entry in smoke_manifest["tasks"]
    }

    assert len(smoke_ids) == 6
    assert {row[0] for row in rows} == smoke_ids

    print("SOURCE DATABASE INTEGRITY: PASS")
    print("SIX REAL COMPLETED TASKS: PASS")

    snapshot_path = OUT / "dataset_v3.sqlite"
    temp_path = OUT / "dataset_v3.sqlite.tmp"

    destination = sqlite3.connect(str(temp_path))

    try:
        source.backup(
            destination, pages=1000,
            sleep=0.05
        )
        check_sqlite(destination)
    finally:
        destination.close()
        source.close()

    os.replace(temp_path, snapshot_path)

    inventory = [{
        "relative_path": "dataset_v3.sqlite",
        "sha256": sha(snapshot_path),
        "bytes": snapshot_path.stat().st_size,
    }]

    # Archive the actual result artifacts,
    # not merely their absolute paths in SQLite.
    archived_results = []

    for task_id, path_text, expected in rows:
        assert re.fullmatch(
            r"task_[0-9a-f]{64}",
            task_id
        )

        original = Path(path_text)
        assert original.is_file()
        assert sha(original) == expected

        data = json.loads(
            original.read_text()
        )
        assert data["task_id"] == task_id
        assert data["numerical_pass"] is True
        assert data["top1_agreement"] is True
        assert len(data["timings_ns"]) == 8

        relative = Path(
            "artifacts"
        ) / f"{task_id}.json"

        item = copy_verified(
            original, relative, expected
        )
        inventory.append(item)

        archived_results.append({
            "task_id": task_id,
            "original_path": str(original),
            "archive_path": str(relative),
            "sha256": expected,
        })

    print("RESULT ARTIFACTS: 6/6 COPIED")
    print("RESULT ARTIFACT SHA256: 6/6 PASS")

    # Preserve experiment design and provenance.
    t17_checks = json.loads(
        (T17 / "checksums.json").read_text()
    )

    for filename, expected in sorted(
        t17_checks.items()
    ):
        inventory.append(
            copy_verified(
                T17 / filename,
                Path("protocol/t17") / filename,
                expected
            )
        )

    inventory.append(
        copy_verified(
            T17 / "checksums.json",
            Path("protocol/t17/checksums.json")
        )
    )

    source_files = [
        (
            T19 / "t19_smoke.py",
            "protocol/t19/t19_smoke.py"
        ),
        (
            T19 / "smoke_manifest.json",
            "protocol/t19/smoke_manifest.json"
        ),
        (
            T05,
            "protocol/t05/validation_metadata.json"
        ),
        (
            T12,
            "protocol/t12/holdout_images.json"
        ),
        (
            MANIFEST_DIR / "resnet50__stage6.json",
            "protocol/manifests/resnet50__stage6.json"
        ),
        (
            MANIFEST_DIR / "convnext_base__fine41.json",
            "protocol/manifests/convnext_base__fine41.json"
        ),
        (
            Path(__file__).resolve(),
            "protocol/archive_t19.py"
        ),
    ]

    for source_path, relative in source_files:
        inventory.append(
            copy_verified(
                source_path, Path(relative)
            )
        )

    print("EXPERIMENT PROTOCOL COPIES: PASS")

    # Disaster-recovery exercise. Restore into a
    # separate in-memory DB and relink artifact
    # paths there. Never edit the source DB.
    backup_conn = sqlite3.connect(
        str(snapshot_path)
    )
    restored = sqlite3.connect(":memory:")
    restored.execute("PRAGMA foreign_keys=ON")

    try:
        backup_conn.backup(restored)

        restored.execute("BEGIN IMMEDIATE")

        for item in archived_results:
            archived_path = (
                OUT / item["archive_path"]
            ).resolve()

            cursor = restored.execute(
                """
                UPDATE tasks
                SET result_path=?
                WHERE task_id=?
                  AND state='completed'
                  AND result_sha256=?
                """,
                (
                    str(archived_path),
                    item["task_id"],
                    item["sha256"],
                )
            )
            assert cursor.rowcount == 1

        restored.commit()
        check_sqlite(restored)

        recovered = restored.execute(
            """
            SELECT task_id,result_path,result_sha256
            FROM tasks
            WHERE state='completed'
            ORDER BY task_id
            """
        ).fetchall()

        assert len(recovered) == 6

        for task_id, path_text, expected in recovered:
            path = Path(path_text)
            assert path.is_file()
            assert sha(path) == expected
            assert json.loads(
                path.read_text()
            )["task_id"] == task_id

    finally:
        restored.close()
        backup_conn.close()

    print("ISOLATED RESTORE TEST: PASS")
    print("RESTORED RESULT LINKS: 6/6 PASS")

    recovery_path = OUT / "recovery_selftest.json"
    write_json(recovery_path, {
        "status": "PASS",
        "method": "isolated_in_memory_sqlite_restore",
        "restored_tasks": 327336,
        "relinked_completed_artifacts": 6,
        "source_database_modified": False,
        "note": (
            "Tests archive recoverability. "
            "Does not copy model weights or "
            "ImageNetV2 image contents."
        ),
    })

    inventory.append({
        "relative_path": recovery_path.name,
        "sha256": sha(recovery_path),
        "bytes": recovery_path.stat().st_size,
    })

    manifest_path = OUT / "bundle_manifest.json"
    write_json(manifest_path, {
        "protocol": "T19_2_portable_backup_v1",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "original_database": str(DB),
        "database_snapshot": "dataset_v3.sqlite",
        "database_sha256": sha(snapshot_path),
        "tasks_total": 327336,
        "completed_tasks": 6,
        "planned_tasks": 327330,
        "results": archived_results,
        "source_protocol_included": True,
        "original_source_files_modified": False,
        "model_weights_included": False,
        "imagenetv2_images_included": False,
        "full_benchmark_environment_clone": False,
    })

    inventory.append({
        "relative_path": manifest_path.name,
        "sha256": sha(manifest_path),
        "bytes": manifest_path.stat().st_size,
    })

    checks_path = OUT / "checksums.json"
    write_json(checks_path, {
        row["relative_path"]: row["sha256"]
        for row in inventory
    })

    assert len(inventory) == len({
        row["relative_path"]
        for row in inventory
    })

    for relative, expected in json.loads(
        checks_path.read_text()
    ).items():
        assert sha(OUT / relative) == expected

    print("ARCHIVE FILE CHECKSUMS: PASS")
    print("\nT19 PORTABLE DATA BACKUP: PASS")
    print("SOURCE DATABASE TASKS: 327336")
    print("ARCHIVED RESULT ARTIFACTS: 6")
    print("RESTORE VERIFICATION: PASS")
    print("BACKUP DIRECTORY:", OUT)

if __name__ == "__main__":
    main()
