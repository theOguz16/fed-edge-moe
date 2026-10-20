import csv
import hashlib
import math
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"

G3 = BASE / "production_campaigns"
SOURCE_DB = (
    G3 / "g3_graph_dataset_mac_v1/graph_dataset.sqlite"
)
DEST_DIR = G3 / "g3_graph_dataset_mac_v2"
DEST_DB = DEST_DIR / "graph_dataset.sqlite"
TEMP_DB = DEST_DIR / ".graph_dataset_building.sqlite"

EXPORT = (
    BASE / "pipeline_dataset_exports"
    / "t26_merged_20261019T093658472630Z"
)
CLEAN = EXPORT / "publication_cleaned_v2"

PATHS = {
    "raw_pairs": EXPORT / "pipeline_pairs_528.csv",
    "raw_modes": EXPORT / "pipeline_modes_1056.csv",
    "clean_pairs": CLEAN / "pipeline_pairs_cleaned_528.csv",
    "clean_modes": CLEAN / "pipeline_modes_cleaned_1056.csv",
    "cleaning_manifest": CLEAN / "cleaning_manifest.json",
    "export_manifest": EXPORT / "manifest.json",
    "t26b_worker": G3 / "t26b_convnext_mac/t26b.py",
}

assert SOURCE_DB.is_file(), SOURCE_DB
assert not DEST_DB.exists(), "G3 v2 exists; refusing overwrite"
assert not TEMP_DB.exists(), "Unfinished temporary DB exists"
for name, path in PATHS.items():
    assert path.is_file(), (name, path)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def keyed(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        assert key not in result, ("duplicate", key)
        result[key] = row
    return result

def integer(value):
    assert value not in ("", None, "not_applicable")
    return int(value)

def optional_int(value):
    if value in ("", None, "not_applicable"):
        return None
    return int(value)

def number(value):
    result = float(value)
    assert math.isfinite(result)
    assert result >= 0
    return result

def boolean(value):
    assert value in ("True", "False"), value
    return int(value == "True")

def insert(conn, table, data):
    keys = list(data)
    query = (
        f"INSERT INTO {table} ("
        + ", ".join(keys)
        + ") VALUES ("
        + ", ".join(":" + k for k in keys)
        + ")"
    )
    conn.execute(query, data)

def normalize_direction(raw):
    direction = raw["direction"]
    match = re.fullmatch(
        r"(cpu_mps|mps_cpu)_cut([1-5])",
        direction,
    )
    if match:
        assert raw["campaign"] == "t26a"
        return match.group(1), int(match.group(2))

    assert direction in ("cpu_mps", "mps_cpu")
    return direction, None

def canonical_cut(raw, cleaned):
    model = raw["model"]
    direction, direction_cut = normalize_direction(raw)
    assert cleaned["direction"] == direction

    legacy = (
        direction_cut
        if direction_cut is not None
        else integer(raw["cut"])
    )

    if model.startswith("convnext_"):
        assert raw["campaign"] == "t26b"
        assert legacy in (1, 3, 5)
        fine_cut = {1: 1, 3: 8, 5: 40}[legacy]
        assert integer(cleaned["cut"]) == legacy
    else:
        fine_cut = legacy
        assert integer(cleaned["cut"]) == fine_cut

    if raw["campaign"] == "t26a":
        assert raw["cut"] == ""
        assert direction_cut == legacy

    first = "cpu" if direction == "cpu_mps" else "mps"
    second = "mps" if direction == "cpu_mps" else "cpu"

    bits = raw.get("mapping_bits") or None

    if bits is not None:
        assert len(bits) == 6
        a = "C" if first == "cpu" else "A"
        b = "C" if second == "cpu" else "A"
        assert bits == a * legacy + b * (6 - legacy)

    return fine_cut, legacy, direction, first, second, bits

# Verify that the T26B worker really used these macro stages.
worker = re.sub(
    r"\s+", "", PATHS["t26b_worker"].read_text()
)
for fragment in (
    "model.features[0]",
    "model.features[1]",
    "model.features[2],model.features[3]",
    "model.features[4],model.features[5]",
    "model.features[6],model.features[7]",
    "model.avgpool,model.classifier",
):
    assert fragment in worker, (
        "T26B stage definition changed", fragment
    )

raw_pairs = keyed(
    read_csv(PATHS["raw_pairs"]), ("case_id",)
)
clean_pairs = keyed(
    read_csv(PATHS["clean_pairs"]), ("case_id",)
)
raw_modes = keyed(
    read_csv(PATHS["raw_modes"]), ("case_id", "mode")
)
clean_modes = keyed(
    read_csv(PATHS["clean_modes"]), ("case_id", "mode")
)

assert len(raw_pairs) == len(clean_pairs) == 528
assert len(raw_modes) == len(clean_modes) == 1056
assert set(raw_pairs) == set(clean_pairs)
assert set(raw_modes) == set(clean_modes)

# Require all original quantitative measurements and source
# pointers to survive cleaning without alteration.
common_pair_fields = (
    "campaign", "model", "family", "microbatch",
    "qos", "repeat", "precision", "power_scope",
    "gpu_kernel_overlap_proven",
    "speedup_pipeline_over_sequential",
    "energy_pipeline_over_sequential",
    "pipeline_faster", "pipeline_energy_lower",
    "throughput_unit", "energy_unit",
)

common_mode_fields = (
    "campaign", "model", "family", "microbatch",
    "qos", "repeat", "precision", "power_scope",
    "gpu_kernel_overlap_proven",
    "throughput_per_second", "throughput_unit",
    "energy_per_item", "energy_unit",
    "cpu_power_w", "gpu_power_w", "median_wall_ms",
    "numerical_status", "result_path",
    "result_sha256", "power_sha256",
)

for key, clean in clean_pairs.items():
    raw = raw_pairs[key]
    for field in common_pair_fields:
        assert raw[field] == clean[field], (
            "pair field changed", key, field
        )

for key, clean in clean_modes.items():
    raw = raw_modes[key]
    for field in common_mode_fields:
        assert raw[field] == clean[field], (
            "mode field changed", key, field
        )

    assert clean["mode"] in ("sequential", "pipeline")

for (case_id,), pair in clean_pairs.items():
    assert (case_id, "sequential") in clean_modes
    assert (case_id, "pipeline") in clean_modes

# Snapshot existing v1 database; never update it in place.
DEST_DIR.mkdir(parents=True, exist_ok=True)

src = sqlite3.connect(
    f"file:{SOURCE_DB}?mode=ro", uri=True
)
dst = sqlite3.connect(TEMP_DB)

try:
    assert src.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"
    src.backup(dst)
finally:
    src.close()
    dst.close()

conn = sqlite3.connect(TEMP_DB)
conn.execute("PRAGMA foreign_keys=ON")

try:
    conn.executescript("""
    CREATE TABLE t26_pairs (
        case_id TEXT PRIMARY KEY,
        campaign TEXT NOT NULL,
        model_id TEXT NOT NULL,
        family TEXT NOT NULL,
        direction TEXT NOT NULL,
        first_device TEXT NOT NULL,
        second_device TEXT NOT NULL,
        cut_after_stage INTEGER NOT NULL,
        source_cut INTEGER NOT NULL,
        mapping_bits TEXT,
        microbatch INTEGER NOT NULL,
        qos TEXT NOT NULL,
        repeat_index INTEGER NOT NULL,
        precision TEXT NOT NULL,
        batch_size INTEGER NOT NULL,
        context_tokens INTEGER,
        cpu_threads INTEGER,
        power_scope TEXT NOT NULL,
        gpu_kernel_overlap_proven INTEGER NOT NULL,
        speedup_pipeline_over_sequential REAL NOT NULL,
        energy_pipeline_over_sequential REAL NOT NULL,
        pipeline_faster INTEGER NOT NULL,
        pipeline_energy_lower INTEGER NOT NULL,
        throughput_unit TEXT NOT NULL,
        energy_unit TEXT NOT NULL,
        FOREIGN KEY (model_id, cut_after_stage)
          REFERENCES cut_boundaries(model_id, cut_after_stage)
    );

    CREATE TABLE t26_modes (
        case_id TEXT NOT NULL,
        mode TEXT NOT NULL
          CHECK(mode IN ('sequential', 'pipeline')),
        throughput_per_second REAL NOT NULL,
        throughput_unit TEXT NOT NULL,
        energy_per_item REAL NOT NULL,
        energy_unit TEXT NOT NULL,
        cpu_power_w REAL NOT NULL,
        gpu_power_w REAL NOT NULL,
        median_wall_ms REAL NOT NULL,
        numerical_status TEXT NOT NULL,
        result_path TEXT NOT NULL,
        reported_result_sha256 TEXT NOT NULL,
        reported_power_sha256 TEXT NOT NULL,
        PRIMARY KEY (case_id, mode),
        FOREIGN KEY (case_id) REFERENCES t26_pairs(case_id)
    );

    CREATE TABLE t26_stage_assignments (
        case_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        stage_index INTEGER NOT NULL,
        device TEXT NOT NULL
          CHECK(device IN ('cpu', 'mps')),
        PRIMARY KEY (case_id, stage_index),
        FOREIGN KEY (case_id) REFERENCES t26_pairs(case_id),
        FOREIGN KEY (model_id, stage_index)
          REFERENCES stage_nodes(model_id, stage_index)
    );

    CREATE INDEX idx_t26_graph_cut
      ON t26_pairs(model_id, cut_after_stage, direction);

    CREATE INDEX idx_t26_config
      ON t26_pairs(model_id, precision, qos, microbatch);

    CREATE INDEX idx_t26_mode
      ON t26_modes(mode);
    """)

    stage_counts = dict(conn.execute(
        "SELECT model_id, stage_count FROM models"
    ))

    imported_cuts = set()
    model_counts = Counter()

    for (case_id,), cleaned in clean_pairs.items():
        raw = raw_pairs[(case_id,)]

        model = cleaned["model"]
        assert model in stage_counts

        cut, legacy, direction, first, second, bits = (
            canonical_cut(raw, cleaned)
        )

        assert 1 <= cut < stage_counts[model]
        assert cleaned["precision"] == "fp32"
        assert cleaned["power_scope"] == (
            "estimated_cpu_gpu_soc_not_wall"
        )
        assert boolean(
            cleaned["gpu_kernel_overlap_proven"]
        ) == 0

        batch = integer(cleaned["batch_size"])
        context = optional_int(cleaned["context_tokens"])

        assert batch == (
            8 if cleaned["family"] == "vision" else 4
        )
        if cleaned["family"] == "text":
            assert context == 128
        else:
            assert context is None

        coverage = conn.execute("""
            SELECT prior_t26_coverage
            FROM cut_boundaries
            WHERE model_id=? AND cut_after_stage=?
        """, (model, cut)).fetchone()

        assert coverage is not None
        assert coverage[0] == "measured", (
            "T26 cut is not registered as measured",
            model, cut
        )

        insert(conn, "t26_pairs", {
            "case_id": case_id,
            "campaign": cleaned["campaign"],
            "model_id": model,
            "family": cleaned["family"],
            "direction": direction,
            "first_device": first,
            "second_device": second,
            "cut_after_stage": cut,
            "source_cut": legacy,
            "mapping_bits": bits,
            "microbatch": integer(cleaned["microbatch"]),
            "qos": cleaned["qos"],
            "repeat_index": integer(cleaned["repeat"]),
            "precision": cleaned["precision"],
            "batch_size": batch,
            "context_tokens": context,
            "cpu_threads": optional_int(
                cleaned.get("cpu_threads")
            ),
            "power_scope": cleaned["power_scope"],
            "gpu_kernel_overlap_proven": 0,
            "speedup_pipeline_over_sequential": number(
                cleaned["speedup_pipeline_over_sequential"]
            ),
            "energy_pipeline_over_sequential": number(
                cleaned["energy_pipeline_over_sequential"]
            ),
            "pipeline_faster": boolean(
                cleaned["pipeline_faster"]
            ),
            "pipeline_energy_lower": boolean(
                cleaned["pipeline_energy_lower"]
            ),
            "throughput_unit": cleaned["throughput_unit"],
            "energy_unit": cleaned["energy_unit"],
        })

        for stage_index in range(
            1, stage_counts[model] + 1
        ):
            insert(conn, "t26_stage_assignments", {
                "case_id": case_id,
                "model_id": model,
                "stage_index": stage_index,
                "device": (
                    first if stage_index <= cut else second
                ),
            })

        imported_cuts.add((model, cut))
        model_counts[model] += 1

    for (case_id, mode), row in clean_modes.items():
        assert (case_id,) in clean_pairs

        pair = clean_pairs[(case_id,)]
        assert row["model"] == pair["model"]
        assert row["precision"] == pair["precision"]
        assert row["qos"] == pair["qos"]
        assert row["microbatch"] == pair["microbatch"]
        assert row["repeat"] == pair["repeat"]

        insert(conn, "t26_modes", {
            "case_id": case_id,
            "mode": mode,
            "throughput_per_second": number(
                row["throughput_per_second"]
            ),
            "throughput_unit": row["throughput_unit"],
            "energy_per_item": number(
                row["energy_per_item"]
            ),
            "energy_unit": row["energy_unit"],
            "cpu_power_w": number(row["cpu_power_w"]),
            "gpu_power_w": number(row["gpu_power_w"]),
            "median_wall_ms": number(row["median_wall_ms"]),
            "numerical_status": row["numerical_status"],
            "result_path": row["result_path"],
            "reported_result_sha256": row["result_sha256"],
            "reported_power_sha256": row["power_sha256"],
        })

    # Register exact import sources and their independently
    # calculated CSV / worker SHA-256 digests.
    for role, path in PATHS.items():
        insert(conn, "source_files", {
            "path": str(path.relative_to(ROOT)),
            "role": "g3_v2_" + role,
            "sha256": sha(path),
            "size_bytes": path.stat().st_size,
        })

    conn.execute(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        ("t26_import_version", "g3_t26_import_v2")
    )
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        ("t26_import_status", "raw_and_cleaned_cross_checked")
    )
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        (
            "t26_cut_semantics",
            "convnext_macro_1_3_5_to_fine41_1_8_40"
        )
    )
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        (
            "t26_result_hash_status",
            "reported_in_source_csv_not_independently_verified"
        )
    )

    assert len(imported_cuts) == 17, imported_cuts
    assert model_counts == {
        "resnet50": 48,
        "convnext_base": 96,
        "convnext_large": 96,
        "distilgpt2": 96,
        "qwen2_5_1_5b": 96,
        "qwen3_1_7b": 96,
    }, model_counts

    counts = {}
    for table in (
        "models",
        "stage_nodes",
        "graph_edges",
        "cut_boundaries",
        "pilot_results",
        "t26_pairs",
        "t26_modes",
        "t26_stage_assignments",
    ):
        counts[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

    assert counts["models"] == 6
    assert counts["stage_nodes"] == 150
    assert counts["graph_edges"] == 144
    assert counts["cut_boundaries"] == 144
    assert counts["pilot_results"] == 12
    assert counts["t26_pairs"] == 528
    assert counts["t26_modes"] == 1056
    assert counts["t26_stage_assignments"] == 14112

    assert conn.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT case_id
            FROM t26_modes
            GROUP BY case_id
            HAVING COUNT(*) != 2
        )
    """).fetchone()[0] == 0

    assert conn.execute(
        "PRAGMA foreign_key_check"
    ).fetchall() == []

    conn.commit()

    assert conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    print("\nG3 V2 IMPORT: PASS")
    for table, count in counts.items():
        print(table, count)

    print("MEASURED MODEL-CUT BOUNDARIES:", len(imported_cuts))
    print("\n=== CANONICAL CUT COVERAGE ===")

    for model, cut in sorted(imported_cuts):
        print(model, "fine_cut=", cut)

finally:
    conn.close()

# Atomic publication without replacing an existing database.
os.link(TEMP_DB, DEST_DB)
TEMP_DB.unlink()

print("\nOUTPUT:", DEST_DB)
print("SQLITE INTEGRITY: ok")
print("G3 V1 MODIFIED: NO")
print("OLD BENCHMARKS MODIFIED: NO")
print("BENCHMARK RERUN: NO")
