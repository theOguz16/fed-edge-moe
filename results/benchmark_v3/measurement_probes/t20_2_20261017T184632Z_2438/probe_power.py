import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()

DB = Path(
    "results/benchmark_v3/dataset_extensions/"
    "t19_3_20261017T183826Z_2438/"
    "dataset_v3_extended.sqlite"
).resolve()

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(part)
    return h.hexdigest()

def atomic_write(path, content):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

assert DB.is_file()

db = sqlite3.connect(
    f"file:{DB}?mode=ro",
    uri=True
)

try:
    assert db.execute(
        "PRAGMA quick_check"
    ).fetchone()[0] == "ok"

    states = dict(db.execute(
        "SELECT state, COUNT(*) "
        "FROM tasks GROUP BY state"
    ).fetchall())

    assert states == {
        "completed": 6,
        "planned": 327330,
    }
finally:
    db.close()

pm = shutil.which("powermetrics")
assert pm is not None

command = [
    pm,
    "-i", "1000",
    "-n", "3",
    "-s", "cpu_power,gpu_power",
]

started = time.monotonic()

try:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        errors="replace",
    )
    rc = result.returncode
    stdout = result.stdout
    stderr = result.stderr
    timed_out = False

except subprocess.TimeoutExpired as e:
    rc = None
    stdout = (
        e.stdout.decode(errors="replace")
        if isinstance(e.stdout, bytes)
        else (e.stdout or "")
    )
    stderr = (
        e.stderr.decode(errors="replace")
        if isinstance(e.stderr, bytes)
        else (e.stderr or "")
    )
    timed_out = True

elapsed = time.monotonic() - started

stdout_path = OUT / "powermetrics_stdout.txt"
stderr_path = OUT / "powermetrics_stderr.txt"

atomic_write(stdout_path, stdout)
atomic_write(stderr_path, stderr)

power_lines = [
    line.strip()
    for line in stdout.splitlines()
    if re.search(
        r"(cpu|gpu|combined|package).*power|"
        r"power.*(cpu|gpu)",
        line,
        flags=re.IGNORECASE,
    )
]

combined_text = (stdout + "\n" + stderr).lower()

if timed_out:
    status = "timeout"
elif rc != 0 and any(
    token in combined_text
    for token in (
        "root",
        "permission",
        "privilege",
        "not permitted",
        "authorization",
    )
):
    status = "administrator_permission_required"
elif rc != 0:
    status = "command_failed"
elif not power_lines:
    status = "command_succeeded_no_recognized_power_lines"
else:
    status = "candidate_power_output_observed"

report = {
    "protocol": "T20_2_unprivileged_power_probe",
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "source_database": str(DB),
    "source_task_states": states,
    "command": command,
    "sudo_used": False,
    "samples_requested": 3,
    "sample_interval_ms": 1000,
    "elapsed_wall_seconds": round(elapsed, 3),
    "exit_code": rc,
    "status": status,
    "power_lines_found": len(power_lines),
    "power_lines_preview": power_lines[:30],
    "stdout_sha256": sha(stdout_path),
    "stderr_sha256": sha(stderr_path),
    "measurement_boundary_validated": False,
    "energy_per_inference_measured": False,
    "new_model_inferences": 0,
    "existing_database_modified": False,
}

report_path = OUT / "power_probe_report.json"
atomic_write(
    report_path,
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ) + "\n"
)

print("=== T20.2 POWER ACCESS TEST ===")
print("SQLITE READ-ONLY CHECK: PASS")
print("SUDO USED: NO")
print("COMMAND EXIT CODE:", rc)
print("ELAPSED SECONDS:", round(elapsed, 3))
print("ACCESS STATUS:", status)
print("POWER LINES FOUND:", len(power_lines))

if power_lines:
    print("\nPOWER OUTPUT PREVIEW:")
    for line in power_lines[:15]:
        print(" ", line)

if rc != 0:
    print("\nERROR OUTPUT:")
    for line in stderr.splitlines()[:15]:
        print(" ", line)

print("\nENERGY PER INFERENCE: NOT MEASURED")
print("DATABASE MODIFIED: NO")
print("REPORT SHA256:", sha(report_path))
print("REPORT:", report_path)
print("T20.2 ACCESS PROBE: COMPLETE")
