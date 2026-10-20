import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()

ORIGINAL = BASE / (
    "pipeline_campaigns/"
    "t21_20261017T200320Z_2438"
)

ORIGINAL_SCRIPT = ORIGINAL / "t21.py"
PATCHED_SCRIPT = ROOT / "t21_single_mps_owner.py"

SOURCE_DB = BASE / (
    "dataset_extensions/"
    "t20_9_20261017T195103Z_2438/"
    "dataset_v3_with_energy.sqlite"
)

SAFE_NAMES = ("mps_cpu_cut3", "mps_cpu_cut5")
REPAIR_NAME = "cpu_mps_cut3"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def atomic_bytes(path, data):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def atomic_json(path, data):
    raw = (
        json.dumps(
            data, indent=2, ensure_ascii=False
        ) + "\n"
    ).encode()
    atomic_bytes(path, raw)

def patch_script():
    assert ORIGINAL_SCRIPT.is_file()
    source = ORIGINAL_SCRIPT.read_text()

    changes = [
        (
            """    for i, stage in enumerate(stages):
        stage.to(dev1 if i < cut else dev2)

    first = stages[:cut]
""",
            """    for i, stage in enumerate(stages):
        stage.to(dev1 if i < cut else dev2)

    # Complete model placement before starting worker threads.
    torch.mps.synchronize()

    first = stages[:cut]
"""
        ),
        (
            """        transfer_start = compute_end

        if y.device.type != dev2:
            y = y.to(dev2)

        if dev2 == "mps":
            torch.mps.synchronize()

        transfer_end = time.monotonic_ns()
""",
            """        transfer_start = compute_end

        if dev2 == "mps":
            # CPU->MPS: producer handles CPU only.
            # GPU transfer belongs to the consumer thread.
            transfer_end = transfer_start
        else:
            if y.device.type != dev2:
                y = y.to(dev2)
            transfer_end = time.monotonic_ns()
"""
        ),
        (
            """    def second_stage(index, y, first_event):
        start = time.monotonic_ns()

        for layer in second:
            y = layer(y)
""",
            """    def second_stage(index, y, first_event):
        # Only the consumer submits GPU work in CPU->MPS mode.
        if dev2 == "mps" and y.device.type != "mps":
            transfer_start = time.monotonic_ns()
            y = y.to("mps")
            torch.mps.synchronize()
            transfer_end = time.monotonic_ns()

            first_event = dict(first_event)
            first_event["transfer_start_ns"] = transfer_start
            first_event["transfer_end_ns"] = transfer_end

        # Stage-2 compute starts after the transfer.
        start = time.monotonic_ns()

        for layer in second:
            y = layer(y)
"""
        ),
    ]

    for old, new in changes:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                "Expected source pattern not found uniquely. "
                f"Matches={count}; refusing unsafe patch."
            )
        source = source.replace(old, new, 1)

    source = source.replace(
        "T21_real_two_thread_pipeline_v1",
        "T21_single_owner_pipeline_v2",
    )

    rendered = source.encode("utf-8")

    if PATCHED_SCRIPT.exists():
        assert PATCHED_SCRIPT.read_bytes() == rendered, (
            "Patched script changed; refusing resume."
        )
    else:
        atomic_bytes(PATCHED_SCRIPT, rendered)

    compile(source, str(PATCHED_SCRIPT), "exec")

def load_engine():
    spec = importlib.util.spec_from_file_location(
        "t21_single_owner_engine",
        PATCHED_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def validate_source_database():
    db = sqlite3.connect(
        f"file:{SOURCE_DB}?mode=ro",
        uri=True
    )
    try:
        assert db.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        assert db.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0] == 327336

        assert db.execute(
            "SELECT COUNT(*) FROM energy_measurements"
        ).fetchone()[0] == 18

        assert db.execute(
            "SELECT COUNT(*) FROM tasks "
            "WHERE state='completed'"
        ).fetchone()[0] == 6
    finally:
        db.close()

def copy_valid_successes(engine):
    recovered = 0

    for repeat in (1, 2):
        for name in SAFE_NAMES:
            for mb in (1, 2):
                relative = (
                    f"repeat_{repeat}/"
                    f"{name}_micro{mb}"
                )
                source = ORIGINAL / relative
                destination = ROOT / relative

                assert (
                    source / "summary.json"
                ).is_file(), relative

                if not destination.exists():
                    shutil.copytree(
                        source, destination
                    )

                summary = engine.verify_completed(
                    destination,
                    name,
                    mb,
                    repeat,
                )

                assert summary is not None
                recovered += 1

    return recovered

def show_failure_log(name, mb, repeat):
    folder = ROOT / (
        f"repeat_{repeat}/{name}_micro{mb}"
    )
    attempts = sorted(
        folder.glob("attempt_*"),
        key=lambda p: p.stat().st_mtime_ns,
    )
    if not attempts:
        return

    latest = attempts[-1]
    error_file = latest / "stderr.txt"

    print("FAILED ATTEMPT:", latest)

    if error_file.exists():
        lines = error_file.read_text(
            errors="replace"
        ).splitlines()

        print("STDERR (last 30 lines):")
        for line in lines[-30:]:
            print(" ", line)

def main():
    lock_path = ROOT / ".runner.lock"

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(
                lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            raise SystemExit(
                "Another recovery runner is active."
            )

        assert ORIGINAL_SCRIPT.is_file()
        assert SOURCE_DB.is_file()
        assert shutil.disk_usage(ROOT).free > (
            5 * 1024**3
        ), "Disk guard: less than 5 GiB free"

        validate_source_database()
        patch_script()

        plan = {
            "protocol": "T21_1_single_mps_owner_recovery",
            "original_t21_script_sha256":
                digest(ORIGINAL_SCRIPT),
            "patched_script_sha256":
                digest(PATCHED_SCRIPT),
            "source_database_sha256":
                digest(SOURCE_DB),
            "preserved_mps_to_cpu_cases": 8,
            "remeasured_cpu_to_mps_cases": 4,
            "old_cpu_to_mps_micro2_results":
                "preserved_in_original_campaign_not_mixed",
        }

        plan_path = ROOT / "recovery_plan.json"

        if plan_path.exists():
            assert json.loads(
                plan_path.read_text()
            ) == plan, (
                "Source changed since first run; "
                "refusing resume."
            )
        else:
            atomic_json(plan_path, plan)

        engine = load_engine()

        preserved = copy_valid_successes(engine)

        print("=== T21.1 METAL-SAFE PIPELINE RECOVERY ===")
        print("ORIGINAL T21 RESULTS: PRESERVED")
        print("VERIFIED MPS->CPU CASES:", preserved)
        print("CPU->MPS CASES TO VALIDATE: 4")
        print("GPU THREAD OWNERSHIP: SINGLE CONSUMER")
        print("OLD SQLITE: READ-ONLY")
        print("RESULT DIRECTORY:", ROOT)
        print()

        failures = []

        for repeat in (1, 2):
            for mb in (1, 2):
                print(
                    f"CPU->MPS microbatch={mb} "
                    f"repeat={repeat}",
                    flush=True,
                )

                result = engine.run_case(
                    REPAIR_NAME, mb, repeat
                )

                if result is None:
                    failures.append((mb, repeat))
                    show_failure_log(
                        REPAIR_NAME, mb, repeat
                    )

                    # Do not repeatedly submit GPU work
                    # after a native Metal assertion.
                    break

            if failures:
                break

        rows = engine.gather()
        engine.export_csv(rows)

        print("\n=== T21.1 PROGRESS ===")
        print("VALIDATED CASES:", len(rows), "/12")

        if failures:
            print(
                "METAL-SAFE RECOVERY: NEEDS INVESTIGATION"
            )
            print(
                "Original successful measurements retained."
            )
            return 2

        if len(rows) != 12:
            print("RECOVERY INCOMPLETE — CAN RESUME")
            return 2

        assert sum(
            row["mapping"] == "cpu_mps_cut3"
            for row in rows
        ) == 4

        engine.create_timeline(rows)

        db_path = engine.integrate_dataset(rows)

        check = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True
        )

        try:
            assert check.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"

            assert check.execute(
                "SELECT COUNT(*) FROM pipeline_measurements"
            ).fetchone()[0] == 24

            assert check.execute(
                "SELECT COUNT(*) FROM energy_measurements"
            ).fetchone()[0] == 18

            assert check.execute(
                "SELECT COUNT(*) FROM tasks"
            ).fetchone()[0] == 327336

            assert check.execute(
                "SELECT COUNT(*) FROM tasks "
                "WHERE state='completed'"
            ).fetchone()[0] == 6
        finally:
            check.close()

        # Keep provenance for both original and revised
        # worker protocols.
        report = {
            "protocol":
                "T21_1_single_mps_owner_recovery",
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "preserved_original_cases": 8,
            "remeasured_cpu_to_mps_cases": 4,
            "total_cases": 12,
            "sequential_pipeline_evaluations": 24,
            "new_database": str(db_path),
            "source_database_modified": False,
            "metal_crash_cause":
                "suspected_cross_thread_command_buffer_use",
            "kernel_level_overlap_proven": False,
            "cpu_wallclock_stage_overlap_measured": True,
            "protocol_variants": [
                "T21_real_two_thread_pipeline_v1",
                "T21_single_owner_pipeline_v2",
            ],
        }

        atomic_json(
            ROOT / "recovery_report.json",
            report,
        )

        files = sorted(
            p for p in ROOT.rglob("*")
            if p.is_file()
            and p.name not in (
                "checksums.json",
                ".runner.lock",
            )
            and not p.name.endswith(".tmp")
        )

        checksums = {
            str(path.relative_to(ROOT)): digest(path)
            for path in files
        }

        atomic_json(
            ROOT / "checksums.json",
            checksums,
        )

        for relative, expected in checksums.items():
            assert digest(ROOT / relative) == expected

        print("\n=== T21.1 FINAL VALIDATION ===")
        print("TOTAL CASES: 12/12 PASS")
        print("MODE EVALUATIONS: 24")
        print("ORIGINAL SUCCESSFUL CASES PRESERVED: 8")
        print("CPU->MPS NEW VALIDATED CASES: 4")
        print("ENERGY MEASUREMENTS PRESERVED: 18")
        print("TASK CATALOG PRESERVED: 327336")
        print("SQLITE INTEGRITY: PASS")
        print("CHECKSUMS: PASS")
        print("OLD SQLITE MODIFIED: NO")
        print("T21.1 PIPELINE RECOVERY: PASS")
        print("NEW SQLITE:", db_path)
        print("RESULT DIRECTORY:", ROOT)

        return 0

if __name__ == "__main__":
    if sys.argv[1:] != ["run"]:
        raise SystemExit(
            "Usage: python recover_t21.py run"
        )

    raise SystemExit(main())
