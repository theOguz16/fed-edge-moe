import hashlib
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3/production_campaigns"
G4 = BASE / "g4_campaign_mac_v1"

GRAPH = BASE / "g3_graph_dataset_mac_v2/graph_dataset.sqlite"
PERF = G4 / "g4_long_campaign_v2/performance.sqlite"
NUM = G4 / "numerical_v1/numerical.sqlite"
OUT = G4 / "g4_graph_merged_mac_v1.sqlite"
TMP = OUT.with_suffix(".sqlite.building")

def open_source(path):
    assert path.is_file(), path
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    return db

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p

def verified_json(path, expected_sha):
    p = resolve(path)
    assert p.is_file(), p
    assert sha(p) == expected_sha, f"SHA mismatch: {p}"
    return json.loads(p.read_text())

def insert(db, table, data):
    cols = list(data)
    names = ", ".join(f'"{name}"' for name in cols)
    placeholders = ", ".join("?" for _ in cols)
    db.execute(
        f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})',
        tuple(data.values()),
    )

if OUT.exists():
    raise SystemExit(
        "Refusing to overwrite an existing graph database: " + str(OUT)
    )
if TMP.exists():
    raise SystemExit(
        "Leftover building file found; inspect before proceeding: " + str(TMP)
    )

print("=== G4.18 GRAPH MERGE V2 ===", flush=True)

original = performance = numerical = merged = None

try:
    original = open_source(GRAPH)
    performance = open_source(PERF)
    numerical = open_source(NUM)

    fd = os.open(str(TMP), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)

    merged = sqlite3.connect(str(TMP))
    merged.row_factory = sqlite3.Row

    original.backup(merged)
    merged.execute("PRAGMA foreign_keys=ON")

    edges = {
        (row["model_id"], row["cut_after_stage"]): row["edge_index"]
        for row in merged.execute("""
            SELECT model_id, cut_after_stage, edge_index
            FROM cut_boundaries
        """)
    }
    assert len(edges) == 144

    merged.executescript("""
        CREATE TABLE g4_import_provenance (
            name TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE g4_perf_cases (
            case_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            cut_after_stage INTEGER NOT NULL,
            edge_index INTEGER NOT NULL,
            direction TEXT NOT NULL,
            repeat_index INTEGER NOT NULL,
            precision TEXT NOT NULL,
            microbatch INTEGER NOT NULL,
            qos TEXT NOT NULL,
            status TEXT NOT NULL,
            protocol TEXT NOT NULL,
            power_scope TEXT,
            pipeline_speedup REAL,
            pipeline_energy_ratio REAL,
            publication_ready INTEGER NOT NULL,
            physical_transfer_bytes_measured INTEGER NOT NULL,
            gpu_kernel_overlap_proven INTEGER NOT NULL,
            power_window_exactly_aligned INTEGER NOT NULL,
            result_path TEXT NOT NULL,
            result_sha256 TEXT NOT NULL,
            model_provenance_json TEXT NOT NULL,
            UNIQUE(model_id, cut_after_stage, direction, repeat_index)
        );

        CREATE TABLE g4_perf_modes (
            case_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            window_start_monotonic_ns INTEGER NOT NULL,
            window_end_monotonic_ns INTEGER NOT NULL,
            wall_seconds REAL NOT NULL,
            rounds INTEGER NOT NULL,
            total_work_items INTEGER NOT NULL,
            work_unit TEXT NOT NULL,
            throughput_per_second REAL NOT NULL,
            compute_round_throughput_per_second REAL NOT NULL,
            stage1_median_ms REAL NOT NULL,
            transfer_median_ms REAL NOT NULL,
            stage2_median_ms REAL NOT NULL,
            logical_activation_bytes_per_microbatch INTEGER NOT NULL,
            activation_shape_json TEXT NOT NULL,
            numerical_status TEXT NOT NULL,
            max_abs_error REAL,
            cpu_power_w REAL,
            gpu_power_w REAL,
            power_sample_count INTEGER,
            estimated_soc_energy_per_work_item_j REAL,
            rss_peak_sampled_bytes INTEGER,
            mps_current_prepost_snapshot_max_bytes INTEGER,
            mps_driver_prepost_snapshot_max_bytes INTEGER,
            mps_peak_measured INTEGER NOT NULL,
            peak_memory_measured INTEGER NOT NULL,
            power_sample_sha256 TEXT,
            PRIMARY KEY(case_id, mode),
            FOREIGN KEY(case_id) REFERENCES g4_perf_cases(case_id)
        );

        CREATE TABLE g4_numeric_cases (
            case_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            cut_after_stage INTEGER NOT NULL,
            edge_index INTEGER NOT NULL,
            direction TEXT NOT NULL,
            precision TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            note TEXT,
            result_path TEXT,
            result_sha256 TEXT,
            UNIQUE(model_id, cut_after_stage, direction, precision)
        );

        CREATE INDEX idx_g4_perf_edge
            ON g4_perf_cases(model_id, edge_index);

        CREATE INDEX idx_g4_numeric_edge
            ON g4_numeric_cases(model_id, edge_index);
    """)

    provenance = {
        "import_protocol": "G4_GRAPH_MERGE_V2",
        "source_graph": str(GRAPH.relative_to(ROOT)),
        "source_performance": str(PERF.relative_to(ROOT)),
        "source_numerical": str(NUM.relative_to(ROOT)),
        "source_graph_sha256": sha(GRAPH),
        "source_performance_sha256": sha(PERF),
        "source_numerical_sha256": sha(NUM),
        "performance_protocol": "G4_LONG_PERFORMANCE_V2",
        "hardware_id": "mac_m4",
        "power_scope": "estimated_cpu_gpu_soc_not_wall",
        "mps_peak_memory_measured": "false",
        "physical_transfer_bytes_measured": "false",
        "gpu_kernel_overlap_proven": "false",
        "task_quality_status_imported": "false",
        "old_datasets_modified": "false",
    }
    for key, value in provenance.items():
        insert(
            merged, "g4_import_provenance",
            {"name": key, "value": value},
        )

    perf_count = 0
    mode_count = 0

    for row in performance.execute("""
        SELECT * FROM perf_cases
        ORDER BY model, cut, direction, repeat
    """):
        key = (row["model"], row["cut"])
        assert key in edges, ("Unknown performance cut", key)
        assert row["status"] == "completed"

        obj = verified_json(row["result_path"], row["result_sha256"])
        case = obj["case"]

        assert obj["protocol"] == "G4_LONG_PERFORMANCE_V2"
        assert (
            case["model"], case["cut"],
            case["direction"], case["repeat"],
        ) == (
            row["model"], row["cut"],
            row["direction"], row["repeat"],
        )
        assert case["precision"] == "fp32"
        assert case["microbatch"] == 2
        assert set(obj["modes"]) == {"sequential", "pipeline"}

        insert(merged, "g4_perf_cases", {
            "case_id": row["case_id"],
            "model_id": row["model"],
            "cut_after_stage": row["cut"],
            "edge_index": edges[key],
            "direction": row["direction"],
            "repeat_index": row["repeat"],
            "precision": case["precision"],
            "microbatch": case["microbatch"],
            "qos": case["qos"],
            "status": row["status"],
            "protocol": obj["protocol"],
            "power_scope": obj.get("power_scope"),
            "pipeline_speedup": obj["pipeline_speedup"],
            "pipeline_energy_ratio": obj["pipeline_energy_ratio"],
            "publication_ready": int(obj["publication_ready"]),
            "physical_transfer_bytes_measured": int(
                obj["physical_transfer_bytes_measured"]
            ),
            "gpu_kernel_overlap_proven": int(
                obj["gpu_kernel_overlap_proven"]
            ),
            "power_window_exactly_aligned": int(
                obj["power_window_exactly_aligned"]
            ),
            "result_path": row["result_path"],
            "result_sha256": row["result_sha256"],
            "model_provenance_json": json.dumps(
                obj["model_provenance"], sort_keys=True
            ),
        })
        perf_count += 1

        for mode_name in ("sequential", "pipeline"):
            m = obj["modes"][mode_name]
            insert(merged, "g4_perf_modes", {
                "case_id": row["case_id"],
                "mode": mode_name,
                "window_start_monotonic_ns": m["window_start_monotonic_ns"],
                "window_end_monotonic_ns": m["window_end_monotonic_ns"],
                "wall_seconds": m["wall_seconds"],
                "rounds": m["rounds"],
                "total_work_items": m["total_work_items"],
                "work_unit": m["work_unit"],
                "throughput_per_second": m["throughput_per_second"],
                "compute_round_throughput_per_second": m[
                    "compute_round_throughput_per_second"
                ],
                "stage1_median_ms": m["stage1_median_ms"],
                "transfer_median_ms": m["transfer_median_ms"],
                "stage2_median_ms": m["stage2_median_ms"],
                "logical_activation_bytes_per_microbatch": m[
                    "logical_activation_bytes_per_microbatch"
                ],
                "activation_shape_json": json.dumps(m["activation_shape"]),
                "numerical_status": m["numerical_status"],
                "max_abs_error": m.get("max_abs_error"),
                "cpu_power_w": m.get("cpu_power_w"),
                "gpu_power_w": m.get("gpu_power_w"),
                "power_sample_count": m.get("power_sample_count"),
                "estimated_soc_energy_per_work_item_j": m.get(
                    "estimated_soc_energy_per_work_item_j"
                ),
                "rss_peak_sampled_bytes": m.get("rss_peak_sampled_bytes"),
                "mps_current_prepost_snapshot_max_bytes": m.get(
                    "mps_current_prepost_snapshot_max_bytes"
                ),
                "mps_driver_prepost_snapshot_max_bytes": m.get(
                    "mps_driver_prepost_snapshot_max_bytes"
                ),
                "mps_peak_measured": int(m["mps_peak_measured"]),
                "peak_memory_measured": int(m["peak_memory_measured"]),
                "power_sample_sha256": m.get("power_sample_sha256"),
            })
            mode_count += 1

    print("Performance imported:", perf_count, flush=True)
    print("Performance modes imported:", mode_count, flush=True)

    num_count = 0
    for row in numerical.execute("""
        SELECT * FROM cases
        ORDER BY model, cut_after_stage, direction, precision
    """):
        key = (row["model"], row["cut_after_stage"])
        assert key in edges, ("Unknown numerical cut", key)

        if row["status"] == "excluded_not_run":
            assert not row["result_path"]
            assert not row["result_sha256"]
        else:
            verified_json(row["result_path"], row["result_sha256"])

        insert(merged, "g4_numeric_cases", {
            "case_id": row["case_id"],
            "model_id": row["model"],
            "cut_after_stage": row["cut_after_stage"],
            "edge_index": edges[key],
            "direction": row["direction"],
            "precision": row["precision"],
            "status": row["status"],
            "attempts": row["attempts"],
            "note": row["note"],
            "result_path": row["result_path"],
            "result_sha256": row["result_sha256"],
        })
        num_count += 1

    assert (perf_count, mode_count, num_count) == (576, 1152, 576)

    for table, expected in (
        ("stage_nodes", 150),
        ("graph_edges", 144),
        ("cut_boundaries", 144),
        ("g4_perf_cases", 576),
        ("g4_perf_modes", 1152),
        ("g4_numeric_cases", 576),
    ):
        actual = merged.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]
        assert actual == expected, (table, actual, expected)

    for table in ("g4_perf_cases", "g4_numeric_cases"):
        count = merged.execute(f"""
            SELECT COUNT(*)
            FROM {table} AS g
            JOIN cut_boundaries AS c
              ON c.model_id = g.model_id
             AND c.cut_after_stage = g.cut_after_stage
             AND c.edge_index = g.edge_index
        """).fetchone()[0]
        assert count == 576, (table, count)

    assert merged.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT model_id, cut_after_stage
            FROM g4_perf_cases
        )
    """).fetchone()[0] == 144

    merged.commit()

    assert merged.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert merged.execute("PRAGMA foreign_key_check").fetchall() == []

    merged.close()
    merged = None

    original.close()
    performance.close()
    numerical.close()
    original = performance = numerical = None

    os.rename(TMP, OUT)

except BaseException:
    for db in (merged, original, performance, numerical):
        if db is not None:
            db.close()
    if TMP.exists():
        TMP.unlink()
    raise

print("\n=== G4.18 GRAPH MERGE V2 ===")
print("MERGE: PASS")
print("OUTPUT:", OUT)
print("GRAPH CUTS:", 144)
print("PERFORMANCE CASES:", perf_count)
print("PERFORMANCE MODES:", mode_count)
print("NUMERICAL CASES:", num_count)
print("SOURCE DATABASES MODIFIED: NO")
print("SOURCE RESULTS MODIFIED: NO")
print("FP16 STATUSES RECLASSIFIED: NO")
