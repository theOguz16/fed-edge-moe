import fcntl
import hashlib
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
G4 = ROOT / "results/benchmark_v3/production_campaigns/g4_campaign_mac_v1"
WORKER = G4 / "g4_perf_worker_v2.py"
DB = G4 / "g4_long_campaign_v2/performance.sqlite"
OUT = G4 / "g4_metal_stress_v1"
LOCK = G4 / "g4_long_campaign_v2/.campaign.lock"

CASES = [
    ("resnet50", 2, 1),
    ("resnet50", 2, 2),
    ("resnet50", 2, 3),
    ("convnext_base", 20, 1),
    ("convnext_large", 20, 1),
    ("distilgpt2", 1, 1),
    ("qwen2_5_1_5b", 10, 1),
    ("qwen3_1_7b", 10, 1),
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stop(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()

def validate(path, case):
    obj = json.loads(path.read_text())
    assert obj["protocol"] == "G4_LONG_PERFORMANCE_V2"
    for key, value in case.items():
        assert obj["case"][key] == value, (key, obj["case"])
    assert set(obj["modes"]) == {"sequential", "pipeline"}
    for mode in obj["modes"].values():
        assert mode["numerical_status"] == "numerical_pass"
        assert math.isfinite(mode["throughput_per_second"])
        assert mode["throughput_per_second"] > 0
        assert mode["power_measured"] is False
    assert math.isfinite(obj["pipeline_speedup"])
    assert obj["pipeline_speedup"] > 0
    return obj

assert WORKER.is_file() and DB.is_file()
assert LOCK.is_file()

with LOCK.open("a+") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        registered = con.execute(
            "SELECT value FROM metadata "
            "WHERE key='perf_worker_sha256'"
        ).fetchone()
        assert registered is not None
        assert registered[0] == sha(WORKER), (
            "Performance worker changed since plan"
        )

    OUT.mkdir(exist_ok=True)
    passed = 0

    for number, (model, cut, repeat) in enumerate(CASES, 1):
        case = {
            "model": model,
            "cut": cut,
            "direction": "mps_cpu",
            "precision": "fp32",
            "microbatch": 2,
            "qos": "default",
            "repeat": repeat,
        }

        name = f"{model}_cut{cut}_repeat{repeat}"
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        result = folder / "result.json"

        print(f"[{number}/{len(CASES)}] {name}", flush=True)

        if result.exists():
            obj = validate(result, case)
            print("SKIP VERIFIED:", name, flush=True)
            passed += 1
            continue

        attempt = 1
        while (folder / f"attempt_{attempt:03d}").exists():
            attempt += 1

        run = folder / f"attempt_{attempt:03d}"
        run.mkdir()

        config = run / "case.json"
        candidate = run / "result.json"

        with config.open("x") as f:
            json.dump(case, f, indent=2)
            f.write("\n")

        env = {
            **os.environ,
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }

        with (run / "stdout.txt").open("x") as out, \
             (run / "stderr.txt").open("x") as err:
            proc = subprocess.Popen(
                [
                    sys.executable, "-u",
                    str(WORKER), "worker",
                    str(config), str(candidate)
                ],
                stdout=out,
                stderr=err,
                env=env,
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                stop(proc)
                rc = 124
            except KeyboardInterrupt:
                stop(proc)
                raise

        if rc != 0 or not candidate.is_file():
            print("STRESS FAILED:", name, "EXIT:", rc)
            print((run / "stderr.txt").read_text(
                errors="replace"
            )[-1800:])
            raise SystemExit(1)

        obj = validate(candidate, case)

        # Hard-link preserves the exact successful result
        # without overwriting any previous result.
        os.link(candidate, result)

        passed += 1
        print(
            "PASS:",
            name,
            "SPEEDUP:",
            round(obj["pipeline_speedup"], 4),
            flush=True,
        )

    print("\nG4 METAL STRESS: PASS")
    print("PASSED:", passed, "/", len(CASES))
    print("DIRECTION: mps_cpu")
    print("NUMERICAL OUTPUT CHECKS: PASS")
    print("PRODUCTION SQLITE MODIFIED: NO")
    print("POWER TESTED IN THIS STRESS: NO")
    print("OUTPUT:", OUT)
