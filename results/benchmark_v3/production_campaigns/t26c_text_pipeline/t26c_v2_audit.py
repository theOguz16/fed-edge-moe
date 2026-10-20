import hashlib
import json
import math
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(
    "results/benchmark_v3/production_campaigns/"
    "t26c_text_pipeline"
).resolve()
DB = ROOT.parent / "t23_mac_v1/dataset_v3_t23_mac.sqlite"
REPORT = ROOT / "t26c_v2_independent_audit.json"
WORKER = ROOT / "t26c_unified_worker.py"

errors = []
warnings = []

def check(condition, message):
    if not condition:
        errors.append(message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def close(a, b):
    return (
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and math.isclose(
            float(a), float(b),
            rel_tol=1e-8,
            abs_tol=1e-12,
        )
    )

def power_pairs(raw):
    cpus = list(re.finditer(
        r"CPU Power:\s*([\d.]+)\s*mW",
        raw, re.I
    ))
    pairs = []
    for i, match in enumerate(cpus):
        stop = (
            cpus[i + 1].start()
            if i + 1 < len(cpus)
            else len(raw)
        )
        gpu = re.search(
            r"GPU Power:\s*([\d.]+)\s*mW",
            raw[match.end():stop],
            re.I,
        )
        if gpu:
            pairs.append((
                float(match.group(1)) / 1000,
                float(gpu.group(1)) / 1000,
            ))
    return pairs

assert DB.is_file(), DB

uri = f"file:{DB}?mode=ro"
with sqlite3.connect(uri, uri=True) as db:
    db.row_factory = sqlite3.Row

    integrity = db.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]
    check(integrity == "ok", f"SQLITE: {integrity}")

    state_counts = dict(db.execute("""
        SELECT status,COUNT(*)
        FROM t26c_v2_cases
        GROUP BY status
    """))
    check(
        state_counts == {
            "completed": 286,
            "legacy_linked": 2,
        },
        f"Unexpected case counts: {state_counts}",
    )

    metadata = dict(db.execute(
        "SELECT key,value FROM t26c_v2_metadata"
    ))
    check(
        metadata.get("protocol")
        == "T26C_UNIFIED_TEXT_PIPELINE_FP32_V2",
        "Protocol mismatch",
    )

    recorded_worker_sha = metadata.get("worker_sha256")
    check(
        bool(recorded_worker_sha),
        "Missing recorded worker SHA",
    )
    check(
        WORKER.is_file()
        and sha(WORKER) == recorded_worker_sha,
        "Current worker differs from committed worker",
    )

    cases = list(db.execute(
        "SELECT * FROM t26c_v2_cases ORDER BY ordinal"
    ))
    results = list(db.execute(
        "SELECT * FROM t26c_v2_results ORDER BY case_id,mode"
    ))
    attempts = list(db.execute(
        "SELECT * FROM t26c_v2_attempts"
    ))
    legacy_cases = list(db.execute(
        "SELECT * FROM t26c_cases ORDER BY repeat"
    ))
    legacy_results = list(db.execute(
        "SELECT * FROM t26c_results"
    ))

    case_by_id = {
        r["case_id"]: r for r in cases
    }
    attempt_by_id = {
        r["attempt_id"]: r for r in attempts
    }
    mode_pairs = defaultdict(dict)
    numerical = Counter()
    model_stats = defaultdict(list)
    inspected = 0

    check(len(cases) == 288, "Case count != 288")
    check(len(results) == 572, "New results != 572")
    check(len(legacy_cases) == 2, "Legacy cases != 2")
    check(len(legacy_results) == 4, "Legacy modes != 4")

    legacy_by_id = {
        r["case_id"]: r for r in legacy_cases
    }
    legacy_result_pairs = defaultdict(set)

    for r in legacy_results:
        cid = r["case_id"]
        legacy_result_pairs[cid].add(r["mode"])
        check(
            r["numerical_status"] == "numerical_pass",
            f"Legacy numerical invalid: {cid}",
        )
        path = Path(r["result_path"])
        check(
            path.is_file()
            and sha(path) == r["result_sha256"],
            f"Legacy JSON SHA mismatch: {cid}",
        )
        if path.is_file():
            power = path.with_name(
                f"powermetrics_{r['mode']}_power_pilot.txt"
            )
            check(
                power.is_file()
                and sha(power) == r["power_sha256"],
                f"Legacy power SHA mismatch: {cid}",
            )

    for case in cases:
        cid = case["case_id"]
        if case["status"] == "legacy_linked":
            old = case["legacy_case_id"]
            check(
                old in legacy_by_id,
                f"Missing legacy link: {cid}",
            )
            if old in legacy_by_id:
                check(
                    case["repeat"] == legacy_by_id[old]["repeat"],
                    f"Legacy repeat mismatch: {cid}",
                )
                check(
                    legacy_result_pairs[old]
                    == {"sequential", "pipeline"},
                    f"Incomplete legacy modes: {cid}",
                )
        elif case["status"] == "completed":
            check(
                case["legacy_case_id"] is None,
                f"Unexpected legacy link: {cid}",
            )
        else:
            errors.append(f"Unfinished case: {cid}")

    for r in results:
        cid = r["case_id"]
        mode = r["mode"]
        tag = f"{cid}/{mode}"
        try:
            case = case_by_id[cid]
            assert case["status"] == "completed"
            assert mode in ("sequential", "pipeline")
            assert mode not in mode_pairs[cid]
            mode_pairs[cid][mode] = r

            attempt = attempt_by_id[r["attempt_id"]]
            assert attempt["case_id"] == cid
            assert attempt["status"] == "completed"
            assert attempt["worker_sha256"] == recorded_worker_sha

            result_path = Path(r["result_path"])
            power_path = Path(r["power_path"])
            assert result_path.is_file()
            assert power_path.is_file()

            assert sha(result_path) == r["result_sha256"]
            assert sha(power_path) == r["power_sha256"]

            obj = json.loads(result_path.read_text())
            assert obj["protocol"] == metadata["protocol"]
            assert obj["worker_sha256"] == recorded_worker_sha
            assert obj["case"]["case_id"] == cid

            for field in (
                "model", "precision", "direction",
                "cut", "batch_size", "context_tokens",
                "microbatch_size", "qos", "repeat",
                "cpu_threads",
            ):
                assert obj["case"][field] == case[field], field

            assert obj["power_scope"] == (
                "estimated_cpu_gpu_soc_not_wall"
            )
            assert obj["gpu_kernel_overlap_proven"] is False
            assert obj["pipeline_thread_qos_verified"] is True

            expected_order = (
                ["sequential", "pipeline"]
                if case["repeat"] == 1
                else ["pipeline", "sequential"]
            )
            assert obj["mode_order"] == expected_order

            m = obj["measurements"][mode]
            assert m["numerical_status"] == "numerical_pass"
            assert m["token_agreement"] == 1.0
            assert math.isfinite(m["max_abs_error"])
            assert m["max_abs_error"] >= 0
            assert m["power_file_sha256"] == r["power_sha256"]

            timings = m["timing_samples_ms"]
            rounds = m["timed_round_count"]
            assert rounds == len(timings)
            assert rounds >= 3
            assert all(
                math.isfinite(t) and t > 0
                for t in timings
            )
            assert close(
                statistics.median(timings),
                m["median_wall_ms"],
            )
            assert close(
                r["median_wall_ms"],
                m["median_wall_ms"],
            )

            pairs = power_pairs(
                power_path.read_text(errors="replace")
            )
            assert len(pairs) >= 3
            assert len(pairs) == m["power_sample_count"]
            assert len(pairs) == r["power_samples"]

            elapsed = m["power_window_s"]
            assert elapsed >= 10
            assert close(elapsed, r["window_seconds"])

            cpu = statistics.mean(x[0] for x in pairs)
            gpu = statistics.mean(x[1] for x in pairs)
            throughput = rounds * 4 * 128 / elapsed
            energy = (cpu + gpu) / throughput

            for actual, expected in (
                (cpu, m["cpu_power_w"]),
                (gpu, m["gpu_power_w"]),
                (
                    throughput,
                    m["sustained_prefill_tokens_s"],
                ),
                (energy, m["gross_energy_j_token"]),
                (cpu, r["cpu_power_w"]),
                (gpu, r["gpu_power_w"]),
                (throughput, r["throughput_tokens_s"]),
                (energy, r["energy_j_token"]),
            ):
                assert close(actual, expected)

            numerical[m["numerical_status"]] += 1
            inspected += 1

        except Exception as exc:
            errors.append(f"{tag}: {type(exc).__name__}: {exc}")

    for case in cases:
        if case["status"] != "completed":
            continue
        cid = case["case_id"]
        modes = mode_pairs[cid]
        check(
            set(modes) == {"sequential", "pipeline"},
            f"Incomplete mode pair: {cid}",
        )
        if set(modes) == {"sequential", "pipeline"}:
            seq = modes["sequential"]
            pipe = modes["pipeline"]
            speedup = (
                pipe["throughput_tokens_s"]
                / seq["throughput_tokens_s"]
            )
            energy_ratio = (
                pipe["energy_j_token"]
                / seq["energy_j_token"]
            )
            check(
                math.isfinite(speedup)
                and math.isfinite(energy_ratio)
                and speedup > 0
                and energy_ratio > 0,
                f"Invalid ratios: {cid}",
            )
            model_stats[case["model"]].append(
                (speedup, energy_ratio)
            )

    # All cases should have exactly two mode results,
    # either in V2 or through a verified legacy link.
    check(
        len(mode_pairs) == 286,
        f"New paired cases: {len(mode_pairs)}",
    )

    report = {
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": metadata.get("protocol"),
        "sqlite_integrity": integrity,
        "case_states": state_counts,
        "new_mode_rows": len(results),
        "new_mode_rows_verified": inspected,
        "legacy_mode_rows": len(legacy_results),
        "numerical_new": dict(numerical),
        "worker_sha256": recorded_worker_sha,
        "models": {},
        "warnings": warnings,
        "error_count": len(errors),
        "errors": errors[:50],
        "read_only_sqlite": True,
    }

    for model, values in sorted(model_stats.items()):
        speeds = [x[0] for x in values]
        energies = [x[1] for x in values]
        report["models"][model] = {
            "new_paired_cases": len(values),
            "median_speedup": statistics.median(speeds),
            "median_energy_ratio": statistics.median(energies),
            "faster_cases": sum(x > 1 for x in speeds),
            "energy_saving_cases": sum(x < 1 for x in energies),
        }

    print("SQLITE:", integrity)
    print("CASE STATES:", state_counts)
    print(
        "NEW MODE FILES VERIFIED:",
        f"{inspected}/{len(results)}",
    )
    print("LEGACY MODE FILES:", len(legacy_results))
    print("NUMERICAL:", dict(numerical))

    for model, data in report["models"].items():
        print(
            "MODEL:", model,
            "PAIRS:", data["new_paired_cases"],
            "MEDIAN SPEEDUP:",
            f"{data['median_speedup']:.3f}x",
            "MEDIAN ENERGY RATIO:",
            f"{data['median_energy_ratio']:.3f}x",
        )

    print("AUDIT ERRORS:", len(errors))
    if errors:
        for item in errors[:15]:
            print("ERROR:", item)

REPORT.write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
    + "\n"
)

print(
    "T26C V2 INDEPENDENT AUDIT:",
    "PASS" if not errors else "FAIL",
)
print("REPORT:", REPORT)
print("SQLITE MODIFIED: NO")
print("BENCHMARK RERUN: NO")

if errors:
    raise SystemExit(1)
