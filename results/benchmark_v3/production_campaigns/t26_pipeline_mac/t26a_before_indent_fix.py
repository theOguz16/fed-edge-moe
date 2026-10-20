import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()
SOURCE = BASE / (
    "pipeline_recovery/"
    "t21_1_20261017T201021Z_2438/"
    "t21_single_mps_owner.py"
)
WORKER = ROOT / "t26a_worker.py"
DB = BASE / (
    "production_campaigns/t23_mac_v1/"
    "dataset_v3_t23_mac.sqlite"
)
PROTOCOL = "T26A_RESNET_PIPELINE_ENERGY_V1"

MAPPINGS = {
    "mps_cpu_cut3": "AAACCC",
    "mps_cpu_cut5": "AAAAAC",
    "cpu_mps_cut3": "CCCAAA",
}
QOS = (
    "background",
    "utility",
    "default",
    "user_initiated",
)
MICROBATCHES = (1, 2)
REPEATS = (1, 2)

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1048576), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(
            data, f, indent=2, ensure_ascii=False
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def disk_guard():
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < 5:
        raise RuntimeError(
            f"Disk guard: {free:.2f} GiB"
        )

def build_worker():
    assert SOURCE.is_file(), SOURCE
    source = SOURCE.read_text(encoding="utf-8")

    helper = '''
def t26a_set_qos(name):
    import ctypes
    values = {
        "background": 0x09,
        "utility": 0x11,
        "default": 0x15,
        "user_initiated": 0x19,
    }
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    setter = lib.pthread_set_qos_class_self_np
    getter = lib.qos_class_self
    setter.argtypes = [ctypes.c_uint, ctypes.c_int]
    setter.restype = ctypes.c_int
    getter.restype = ctypes.c_uint
    code = setter(values[name], 0)
    if code != 0 or getter() != values[name]:
        raise RuntimeError("QOS_VERIFICATION_FAILED")

def t26a_power_pairs(raw):
    import re
    import math
    cpu = list(re.finditer(
        r"CPU Power:\\s*([\\d.]+)\\s*mW", raw, re.I
    ))
    pairs = []
    for i, match in enumerate(cpu):
        stop = (
            cpu[i+1].start()
            if i+1 < len(cpu) else len(raw)
        )
        gpu = re.search(
            r"GPU Power:\\s*([\\d.]+)\\s*mW",
            raw[match.end():stop], re.I
        )
        if gpu:
            a = float(match.group(1))/1000
            b = float(gpu.group(1))/1000
            if (
                math.isfinite(a) and math.isfinite(b)
                and a >= 0 and b >= 0
            ):
                pairs.append((a,b))
    return pairs

'''

    anchor = "def worker(mapping_name, mb_size, repeat, result_path):"
    assert source.count(anchor) == 1
    source = source.replace(
        anchor, helper + anchor, 1
    )

    anchor = (
        "    torch.set_num_threads(4)\n"
        "    torch.manual_seed(2026)"
    )
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        '    t26a_set_qos(os.environ["T26A_QOS"])\n'
        + anchor,
        1
    )

    start_marker = (
        "        rounds = []\n\n"
        "        for round_index in range(TIMED_ROUNDS):\n"
    )
    end_marker = "        measurements[mode] = {\n"
    assert source.count(start_marker) == 1

    begin = source.index(start_marker)
    end = source.index(end_marker, begin)

    # Preserve the previously validated numerical checks,
    # per-microbatch events and throughput calculations.
    original_body = source[
        begin + len(start_marker):end
    ].rstrip("\n")

    nested_body = "\n".join(
        "    " + line if line else ""
        for line in original_body.splitlines()
    )

    replacement = (
        '''        rounds = []
        power_path = result_path.with_name(
            "powermetrics_" + mode + ".txt"
        )
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
            mode_started = time.monotonic()
            try:
                while (
                    len(rounds) < TIMED_ROUNDS
                    or time.monotonic() - mode_started < 10.0
                ):
                    round_index = len(rounds)
'''
        + nested_body
        + '''
            finally:
                power_window_s = (
                    time.monotonic() - mode_started
                )
                if sensor.poll() is None:
                    sensor.send_signal(signal.SIGINT)
                try:
                    sensor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    sensor.kill()
                    sensor.wait()

        pairs = t26a_power_pairs(
            power_path.read_text(errors="replace")
        )
        if len(pairs) < 2:
            raise RuntimeError(
                "POWER_MISSING "
                f"mode={mode} samples={len(pairs)} "
                f"sensor_exit={sensor.returncode}"
            )

        cpu_w = statistics.mean(x[0] for x in pairs)
        gpu_w = statistics.mean(x[1] for x in pairs)
        sustained_ips = (
            BATCH * len(rounds) / power_window_s
        )
        assert sustained_ips > 0

'''
    )
    source = (
        source[:begin] + replacement + source[end:]
    )

    anchor = (
        '        measurements[mode] = {\n'
        '            "rounds": rounds,\n'
    )
    assert source.count(anchor) == 1

    addition = (
        '        measurements[mode] = {\n'
        '            "rounds": rounds,\n'
        '            "power_cpu_w": cpu_w,\n'
        '            "power_gpu_w": gpu_w,\n'
        '            "power_sample_count": len(pairs),\n'
        '            "power_window_s": power_window_s,\n'
        '            "power_raw_sha256": sha(power_path),\n'
        '            "sustained_throughput_images_s": sustained_ips,\n'
        '            "gross_energy_j_image": '
        '(cpu_w+gpu_w)/sustained_ips,\n'
    )
    source = source.replace(anchor, addition, 1)

    power_flag = (
        '        "power_measured": False,\n'
        '        "energy_measured": False,'
    )
    assert source.count(power_flag) == 1
    source = source.replace(
        power_flag,
        '        "power_measured": True,\n'
        '        "energy_measured": True,',
        1
    )

    source = source.replace(
        "T21_single_owner_pipeline_v2",
        PROTOCOL
    )

    compile(source, str(WORKER), "exec")
    output = source.encode("utf-8")

    if WORKER.exists():
        if WORKER.read_bytes() != output:
            raise RuntimeError(
                "Existing T26A worker differs; "
                "refusing unsafe overwrite"
            )
    else:
        with WORKER.open("xb") as f:
            f.write(output)

    return sha(SOURCE), sha(WORKER)

def connect():
    assert DB.is_file()
    db = sqlite3.connect(
        str(DB), timeout=90,
        isolation_level=None
    )
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=90000")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db

def setup(db, source_sha, worker_sha):
    assert db.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    for table, expected in (
        ("t23_measurements", 90),
        ("t24r_measurements", 203),
        ("t24_1_results", 13),
        ("t25_results", 256),
    ):
        count = db.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]
        assert count == expected, (table, count)

    db.execute("""
        CREATE TABLE IF NOT EXISTS t26a_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS t26a_cases (
            case_id TEXT PRIMARY KEY,
            ordinal INTEGER UNIQUE NOT NULL,
            mapping TEXT NOT NULL,
            mapping_bits TEXT NOT NULL,
            microbatch INTEGER NOT NULL,
            qos TEXT NOT NULL,
            repeat INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'planned',
            current_attempt TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS t26a_attempts (
            attempt_id TEXT PRIMARY KEY,
            case_id TEXT NOT NULL,
            started_utc TEXT NOT NULL,
            finished_utc TEXT,
            status TEXT NOT NULL,
            exit_code INTEGER,
            error TEXT,
            folder TEXT NOT NULL,
            FOREIGN KEY(case_id)
                REFERENCES t26a_cases(case_id)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS t26a_results (
            case_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            sustained_images_s REAL NOT NULL,
            median_round_images_s REAL NOT NULL,
            batch_wall_median_ms REAL NOT NULL,
            stage_overlap_median_ms REAL NOT NULL,
            cpu_power_w REAL NOT NULL,
            gpu_power_w REAL NOT NULL,
            gross_energy_j_image REAL NOT NULL,
            power_sample_count INTEGER NOT NULL,
            power_window_s REAL NOT NULL,
            result_path TEXT NOT NULL,
            result_sha256 TEXT NOT NULL,
            power_raw_sha256 TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            PRIMARY KEY(case_id,mode),
            FOREIGN KEY(case_id)
                REFERENCES t26a_cases(case_id)
        )
    """)

    metadata = {
        "protocol": PROTOCOL,
        "source_t21_sha256": source_sha,
        "derived_worker_sha256": worker_sha,
        "model": "resnet50",
        "precision": "fp32",
        "batch_size": "8",
        "cpu_threads": "4",
        "mode_comparison": "sequential_vs_real_pipeline",
        "power_boundary": "estimated_cpu_plus_gpu_soc",
        "kernel_overlap_proven": "false",
    }

    for key, value in metadata.items():
        row = db.execute(
            "SELECT value FROM t26a_metadata WHERE key=?",
            (key,)
        ).fetchone()
        if row:
            assert row[0] == value, key
        else:
            db.execute(
                "INSERT INTO t26a_metadata VALUES (?,?)",
                (key, value)
            )

    cases = []
    for qos in QOS:
        for repeat in REPEATS:
            for mapping, bits in MAPPINGS.items():
                for mb in MICROBATCHES:
                    name = (
                        f"{PROTOCOL}|{qos}|{repeat}|"
                        f"{mapping}|{mb}"
                    )
                    case_id = (
                        "t26a_"
                        + hashlib.sha256(
                            name.encode()
                        ).hexdigest()
                    )
                    cases.append((
                        case_id, len(cases), mapping,
                        bits, mb, qos, repeat
                    ))

    assert len(cases) == 48

    db.execute("BEGIN IMMEDIATE")
    try:
        for case in cases:
            db.execute("""
                INSERT OR IGNORE INTO t26a_cases (
                    case_id,ordinal,mapping,
                    mapping_bits,microbatch,qos,repeat
                ) VALUES (?,?,?,?,?,?,?)
            """, case)

        assert db.execute(
            "SELECT COUNT(*) FROM t26a_cases"
        ).fetchone()[0] == 48

        for row in db.execute("""
            SELECT case_id,current_attempt
            FROM t26a_cases
            WHERE status='running'
        """).fetchall():
            db.execute("""
                UPDATE t26a_attempts
                SET status='interrupted',
                    finished_utc=?
                WHERE attempt_id=?
                  AND status='running'
            """, (utc(), row["current_attempt"]))
            db.execute("""
                UPDATE t26a_cases
                SET status='planned',
                    current_attempt=NULL
                WHERE case_id=?
            """, (row["case_id"],))

        db.commit()
    except BaseException:
        db.rollback()
        raise

    for row in db.execute("""
        SELECT result_path,result_sha256
        FROM t26a_results
    """):
        assert Path(row[0]).is_file()
        assert sha(row[0]) == row[1]

def claim(db):
    db.execute("BEGIN IMMEDIATE")
    try:
        row = db.execute("""
            SELECT * FROM t26a_cases
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

        db.execute("""
            UPDATE t26a_cases
            SET status='running',current_attempt=?
            WHERE case_id=?
        """, (token, case["case_id"]))

        db.execute("""
            INSERT INTO t26a_attempts (
                attempt_id,case_id,started_utc,
                status,folder
            ) VALUES (?,?,?,'running',?)
        """, (
            token, case["case_id"], utc(), str(folder)
        ))
        db.commit()
        return case, token, folder

    except BaseException:
        db.rollback()
        raise

def record_failure(db, case, token, status, code, message):
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("""
            UPDATE t26a_attempts
            SET status=?,finished_utc=?,
                exit_code=?,error=?
            WHERE attempt_id=?
        """, (
            status, utc(), code,
            message[-3000:], token
        ))
        db.execute("""
            UPDATE t26a_cases
            SET status=?,current_attempt=NULL
            WHERE case_id=?
        """, (
            "planned" if status == "interrupted"
            else "failed",
            case["case_id"]
        ))
        db.commit()
    except BaseException:
        db.rollback()
        raise

def commit(db, case, token, folder):
    path = folder / "result.json"
    data = json.loads(path.read_text())

    assert data["protocol"] == PROTOCOL
    assert data["model"] == "resnet50"
    assert data["precision"] == "fp32"
    assert data["mapping"] == case["mapping"]
    assert data["mapping_bits"] == case["mapping_bits"]
    assert data["microbatch_size"] == case["microbatch"]
    assert data["independent_process_repeat"] == case["repeat"]
    assert data["status"] == "numerical_validation_pass"
    assert data["power_measured"] is True
    assert data["energy_measured"] is True
    assert data["pipeline_has_actual_separate_threads"] is True
    assert data["hardware_kernel_overlap_proven"] is False

    records = []

    for mode in ("sequential", "pipeline"):
        metric = data["measurements"][mode]
        power_path = folder / (
            "powermetrics_" + mode + ".txt"
        )

        assert power_path.is_file()
        assert sha(power_path) == metric["power_raw_sha256"]
        assert metric["power_sample_count"] >= 2
        assert len(metric["rounds"]) >= 10

        throughput = metric[
            "sustained_throughput_images_s"
        ]
        energy = metric["gross_energy_j_image"]
        cpu = metric["power_cpu_w"]
        gpu = metric["power_gpu_w"]

        for value in (
            throughput, energy, cpu, gpu,
            metric["power_window_s"],
        ):
            assert math.isfinite(value)
            assert value >= 0

        assert throughput > 0
        assert math.isclose(
            energy, (cpu+gpu)/throughput,
            rel_tol=1e-9, abs_tol=1e-12
        )

        records.append((
            case["case_id"], mode,
            throughput,
            metric["throughput_median_images_s"],
            metric["batch_wall_median_ms"],
            metric["stage_overlap_median_ms"],
            cpu, gpu, energy,
            metric["power_sample_count"],
            metric["power_window_s"],
            str(path.resolve()),
            sha(path),
            sha(power_path),
            token, utc()
        ))

    db.execute("BEGIN IMMEDIATE")
    try:
        for values in records:
            assert len(values) == 16
            db.execute(
                "INSERT INTO t26a_results VALUES ("
                + ",".join("?" for _ in values)
                + ")",
                values
            )

        changed = db.execute("""
            UPDATE t26a_cases
            SET status='completed',
                current_attempt=NULL
            WHERE case_id=? AND current_attempt=?
        """, (case["case_id"], token))
        assert changed.rowcount == 1

        db.execute("""
            UPDATE t26a_attempts
            SET status='completed',
                finished_utc=?,exit_code=0
            WHERE attempt_id=?
        """, (utc(), token))
        db.commit()

    except BaseException:
        db.rollback()
        raise

    seq = data["measurements"]["sequential"]
    pipe = data["measurements"]["pipeline"]

    print(
        "COMMITTED",
        case["mapping"],
        "micro=", case["microbatch"],
        "qos=", case["qos"],
        "repeat=", case["repeat"],
        "speedup=",
        f"{pipe['sustained_throughput_images_s'] / seq['sustained_throughput_images_s']:.3f}x",
        "seq_J/image=",
        f"{seq['gross_energy_j_image']:.6f}",
        "pipe_J/image=",
        f"{pipe['gross_energy_j_image']:.6f}",
        flush=True
    )

def report(db):
    print("\n=== T26A PIPELINE STATUS ===")
    for row in db.execute("""
        SELECT status,COUNT(*)
        FROM t26a_cases
        GROUP BY status
    """):
        print(row[0], row[1])

    count = db.execute(
        "SELECT COUNT(*) FROM t26a_results"
    ).fetchone()[0]

    print("POWER+PERFORMANCE MODE MEASUREMENTS:", count, "/96")
    print("MODEL: ResNet50 FP32")
    print("OTHER FIVE MODELS: NOT YET MEASURED IN T26")
    print("DATABASE:", DB)

def stop_group(proc):
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

def run(limit):
    disk_guard()
    source_sha, worker_sha = build_worker()

    lock_path = BASE / (
        "production_campaigns/t23_mac_v1/"
        ".campaign.lock"
    )

    with lock_path.open("a+") as lock:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_EX | fcntl.LOCK_NB
        )

        db = connect()
        try:
            setup(db, source_sha, worker_sha)
            print("T26A PROTOCOL VERIFIED", flush=True)

            for index in range(limit):
                disk_guard()
                claimed = claim(db)

                if claimed is None:
                    print("NO REMAINING T26A CASES")
                    break

                case, token, folder = claimed
                result = folder / "result.json"
                stdout = folder / "stdout.txt"
                stderr = folder / "stderr.txt"

                print(
                    f"[{index+1}/{limit}]",
                    case["mapping"],
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
                            str(WORKER), "worker",
                            case["mapping"],
                            str(case["microbatch"]),
                            str(case["repeat"]),
                            str(result),
                        ],
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "T26A_QOS": case["qos"],
                            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                        }
                    )
                    try:
                        code = proc.wait(timeout=900)
                    except KeyboardInterrupt:
                        interrupted = True
                        stop_group(proc)
                        code = proc.returncode
                    except subprocess.TimeoutExpired:
                        timeout = True
                        stop_group(proc)
                        code = proc.returncode

                error = stderr.read_text(
                    errors="replace"
                )

                if interrupted:
                    record_failure(
                        db, case, token,
                        "interrupted", code, error
                    )
                    print("INTERRUPTED — RESUMABLE")
                    break

                if timeout or code != 0:
                    # Clean up any powermetrics child left
                    # after a native MPS worker crash.
                    stop_group(proc)
                    status = (
                        "timeout" if timeout
                        else "failed_runtime"
                    )
                    record_failure(
                        db, case, token,
                        status, code, error
                    )
                    print(
                        "FAILED — CAMPAIGN PAUSED",
                        status, "exit=", code,
                        error[-1300:],
                        flush=True
                    )
                    break

                if not result.is_file():
                    record_failure(
                        db, case, token,
                        "missing_result", code,
                        "Worker returned 0 without JSON"
                    )
                    print("MISSING RESULT — PAUSED")
                    break

                try:
                    commit(db, case, token, folder)
                except Exception as exc:
                    record_failure(
                        db, case, token,
                        "validation_failed",
                        code, repr(exc)
                    )
                    print(
                        "RESULT VALIDATION FAILED:",
                        repr(exc), flush=True
                    )
                    break

            report(db)
        finally:
            db.close()

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "status":
        db = connect()
        try:
            report(db)
        finally:
            db.close()
        return

    if len(sys.argv) != 3 or sys.argv[1] != "run":
        raise SystemExit(
            "Usage: python t26a.py run 48"
        )

    run(int(sys.argv[2]))

if __name__ == "__main__":
    main()
