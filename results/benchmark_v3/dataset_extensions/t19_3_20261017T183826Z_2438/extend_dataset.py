import csv
import fcntl
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()
BASE = Path("results/benchmark_v3").resolve()

SOURCE_DB = BASE / (
    "dataset_store/"
    "t18_20261017T174740Z_2438/"
    "dataset_v3.sqlite"
)

T17 = BASE / (
    "dataset_v3_design/"
    "t17_20261017T160851Z_2438"
)

NEW_DB = OUT / "dataset_v3_extended.sqlite"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def save_json(path, data):
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

def source_state(conn):
    states = dict(conn.execute(
        "SELECT state,COUNT(*) "
        "FROM tasks GROUP BY state"
    ).fetchall())

    results = conn.execute(
        "SELECT task_id,result_path,result_sha256 "
        "FROM tasks WHERE state='completed' "
        "ORDER BY task_id"
    ).fetchall()

    return states, results

assert SOURCE_DB.is_file()
assert (T17 / "model_catalog.csv").is_file()

checks = json.loads(
    (T17 / "checksums.json").read_text()
)
for name, expected in checks.items():
    assert sha(T17 / name) == expected, name

models_csv = {}
with (T17 / "model_catalog.csv").open(
    newline="", encoding="utf-8"
) as f:
    for row in csv.DictReader(f):
        models_csv[row["model"]] = row

assert len(models_csv) == 3

lock_path = SOURCE_DB.with_name(
    "dataset_v3.runner.lock"
)

with lock_path.open("a+") as lock:
    fcntl.flock(
        lock.fileno(),
        fcntl.LOCK_EX | fcntl.LOCK_NB
    )

    try:
        source = sqlite3.connect(
            f"file:{SOURCE_DB}?mode=ro",
            uri=True
        )

        assert source.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        before_states, before_results = (
            source_state(source)
        )

        assert before_states == {
            "completed": 6,
            "planned": 327330,
        }

        for task_id, path, expected in before_results:
            assert Path(path).is_file(), task_id
            assert sha(path) == expected, task_id

        temporary = NEW_DB.with_name(
            NEW_DB.name + ".tmp"
        )
        assert not temporary.exists()
        assert not NEW_DB.exists()

        destination = sqlite3.connect(
            str(temporary)
        )
        try:
            source.backup(
                destination,
                pages=1000,
                sleep=0.05
            )
            assert destination.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"
        finally:
            destination.close()
            source.close()

        os.replace(temporary, NEW_DB)

    finally:
        fcntl.flock(
            lock.fileno(),
            fcntl.LOCK_UN
        )

db = sqlite3.connect(str(NEW_DB))
db.execute("PRAGMA foreign_keys=ON")
db.execute("PRAGMA synchronous=FULL")

db.executescript("""
CREATE TABLE IF NOT EXISTS research_models (
    model_id TEXT PRIMARY KEY,
    modality TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    checkpoint_status TEXT NOT NULL,
    partition_manifest_status TEXT NOT NULL,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_dimensions (
    dimension_id TEXT PRIMARY KEY,
    values_json TEXT NOT NULL,
    applicability TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    notes TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS measurement_registry (
    metric_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    unit TEXT NOT NULL,
    interpretation TEXT NOT NULL,
    availability TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS figure_registry (
    figure_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    required_metrics_json TEXT NOT NULL,
    grouping_json TEXT NOT NULL,
    status TEXT NOT NULL,
    caution TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_notes (
    note_id TEXT PRIMARY KEY,
    source_pages TEXT NOT NULL,
    statement TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    new_measurement INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_extensions (
    extension_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    original_database TEXT NOT NULL,
    source_paper TEXT NOT NULL,
    scheduler_in_scope INTEGER NOT NULL
);
""")

models = [
    ("resnet50", "vision", "easy",
     "verified_T17", "verified_T17",
     "T17_and_paper_page_2"),
    ("convnext_base", "vision", "medium",
     "verified_T17", "verified_T17",
     "T17_and_paper_page_2"),
    ("convnext_large", "vision", "hard",
     "verified_T17", "verified_T17",
     "T17_and_paper_page_2"),
    ("distilgpt2", "text", "easy",
     "exact_checkpoint_pending",
     "layer_manifest_pending",
     "paper_page_2"),
    ("qwen2_5_1_5b", "text", "medium",
     "exact_checkpoint_pending",
     "layer_manifest_pending",
     "paper_page_2"),
    ("qwen3_1_7b", "text", "hard",
     "exact_checkpoint_pending",
     "layer_manifest_pending",
     "paper_page_2"),
]

dimensions = [
    (
        "precision",
        ["fp32", "fp16", "bf16"],
        "vision_and_text",
        "paper_and_extension_plan",
        "support_validation_pending",
        "BF16 is an additional proposed option; "
        "do not assume support on every backend."
    ),
    (
        "mac_cpu_threads",
        [1, 2, 4, 6, 8, 10],
        "apple_m4",
        "T17",
        "catalogued",
        "Existing six-thread design."
    ),
    (
        "macos_qos",
        [
            "background", "utility",
            "default", "user_initiated"
        ],
        "apple_m4",
        "paper_pages_3_4",
        "planned",
        "Record actual OS policy and invocation method."
    ),
    (
        "windows_qos",
        ["system", "eco", "high"],
        "msi_windows",
        "paper_pages_3_4",
        "planned",
        "Windows categories are not equivalent to macOS QoS."
    ),
    (
        "cpu_affinity",
        [
            "unrestricted",
            "four_physical_cores",
            "two_physical_cores_with_smt"
        ],
        "msi_windows",
        "paper_pages_3_4",
        "planned",
        "Verify physical cores and logical CPU mapping."
    ),
    (
        "vision_resolution",
        [224, 320],
        "vision",
        "paper_page_2",
        "planned",
        "Record exact resolution, batch and preprocessing."
    ),
    (
        "execution_mode",
        ["sequential", "overlapped_pipeline"],
        "partitioned_models",
        "paper_pages_3_5",
        "planned",
        "Pipeline must demonstrate observed stage overlap."
    ),
    (
        "text_workload",
        [
            "prompt_token_length",
            "generated_token_length",
            "evaluation_token_count",
            "generation_sampling_config"
        ],
        "text",
        "paper_pages_2_4",
        "values_pending",
        "Exact lengths and tokenizer configurations "
        "must be frozen before measurement."
    ),
    (
        "measurement_window",
        [
            "short_characterization",
            "sustained_power",
            "quality_evaluation"
        ],
        "all_models",
        "paper_page_2",
        "planned",
        "Different protocols must not be pooled blindly."
    ),
    (
        "platform",
        [
            "apple_m4_cpu_mps",
            "intel_i7_11800h_cpu",
            "nvidia_rtx3050_laptop_cuda"
        ],
        "all_models",
        "paper_page_2",
        "second_platform_pending",
        "RTX 3050 Laptop GPU reported with 4GB VRAM."
    ),
    (
        "energy_measurement_boundary",
        [
            "apple_combined_soc",
            "intel_cpu_package",
            "nvidia_gpu_board",
            "intel_package_plus_gpu_board"
        ],
        "power_energy",
        "paper_pages_2_5",
        "sensor_validation_pending",
        "These boundaries must not be treated "
        "as equivalent whole-system energy."
    ),
    (
        "quality_evaluation_scope",
        [
            "dataset",
            "tokenizer",
            "precision",
            "input_shape",
            "preprocessing",
            "sequence_length",
            "evaluated_token_count"
        ],
        "vision_and_text",
        "paper_pages_3_7",
        "planned",
        "Only compare quality evidence with matching scope."
    ),
]

metrics = [
    ("task_completion", "fitness", "boolean",
     "Actual inference completion and validation", "planned"),
    ("numerical_max_abs_error", "quality", "absolute",
     "Reference-to-placement numerical difference", "existing_T19"),
    ("vision_top1", "quality", "percent",
     "Ground-truth Top-1 accuracy", "planned"),
    ("vision_top5", "quality", "percent",
     "Ground-truth Top-5 accuracy", "planned"),
    ("perplexity", "text_quality", "unitless",
     "Evaluation-corpus language-model perplexity", "planned"),
    ("negative_log_likelihood", "text_quality", "nats_per_token",
     "Teacher-forced NLL on a fixed token set", "planned"),
    ("kl_fp32_to_fp16", "text_quality", "nats",
     "Matched-context token distribution divergence", "planned"),
    ("token_agreement", "text_quality", "fraction",
     "Matched-context next-token argmax agreement", "planned"),
    ("selfcheck_nli", "text_quality", "score",
     "Sampling-based consistency; not hallucination rate", "planned"),
    ("latency_median", "latency", "ms",
     "End-to-end inference median", "existing_T19"),
    ("latency_p95", "latency", "ms",
     "End-to-end distribution 95th percentile", "planned"),
    ("latency_p99", "latency", "ms",
     "End-to-end distribution 99th percentile", "planned"),
    ("throughput_images", "throughput", "images_per_second",
     "Measured sustained vision processing rate", "planned"),
    ("throughput_tokens", "throughput", "tokens_per_second",
     "Measured language decoding throughput", "planned"),
    ("time_to_first_token", "latency", "ms",
     "Language generation first-token latency", "planned"),
    ("inter_token_latency", "latency", "ms",
     "Language generation token-to-token timing", "planned"),
    ("average_power", "energy", "watts",
     "Measured power with declared sensor boundary", "planned"),
    ("energy_per_item", "energy", "joules_per_item",
     "Same-window energy/item with compatible scope", "planned"),
    ("rss", "memory", "bytes",
     "Process resident memory; not total physical RAM", "existing_T19"),
    ("peak_memory", "memory", "bytes",
     "Measured peak with an explicitly defined scope", "planned"),
    ("accelerator_allocated", "memory", "bytes",
     "MPS/CUDA allocator-specific memory", "planned"),
    ("transfer_logical_bytes", "transfer", "bytes",
     "Logical transfer volume; not elapsed time", "existing_T19"),
    ("transfer_latency", "transfer", "nanoseconds",
     "Synchronized per-boundary transfer duration", "planned"),
    ("tensor_roundtrip_error", "transfer", "absolute",
     "Numerical effect of transfer-only roundtrip", "planned"),
    ("pipeline_overlap", "pipeline", "nanoseconds",
     "Observed inter-stage overlapping execution", "planned"),
    ("stage_runtime", "pipeline", "nanoseconds",
     "Per-stage execution time", "planned"),
    ("os_qos_policy", "environment", "category",
     "Observed scheduler policy", "planned"),
    ("cpu_affinity_mask", "environment", "cpu_list",
     "Actual allowed CPU cores", "planned"),
]

figures = [
    (
        "F01_thread_scaling",
        "CPU thread count versus throughput",
        ["throughput_images"],
        ["platform", "model", "cpu_threads"],
        "Repeated observations, not assumed confidence intervals."
    ),
    (
        "F02_os_qos",
        "Operating-system QoS versus throughput",
        ["throughput_images", "os_qos_policy"],
        ["platform", "model", "qos_policy"],
        "Mac and Windows QoS categories are not equivalent."
    ),
    (
        "F03_affinity_smt",
        "CPU core placement versus throughput and energy",
        ["throughput_images", "energy_per_item"],
        ["platform", "core_placement"],
        "Do not combine incompatible energy boundaries."
    ),
    (
        "F04_pipeline_tradeoff",
        "Pipeline throughput versus energy",
        ["throughput_images", "energy_per_item",
         "pipeline_overlap"],
        ["platform", "model", "partition", "execution_mode"],
        "Overlap and energy measurement scope must be verified."
    ),
    (
        "F05_partition_heatmap",
        "Partition and thread performance heatmap",
        ["latency_median", "transfer_logical_bytes"],
        ["model", "group_mapping", "cpu_threads"],
        "Show slow configurations and failures."
    ),
    (
        "F06_precision_quality",
        "FP32/FP16 vision accuracy and performance",
        ["vision_top1", "vision_top5", "latency_median"],
        ["model", "device", "precision"],
        "Accuracy deltas are percentage points."
    ),
    (
        "F07_language_quality",
        "Language quality and precision sensitivity",
        ["perplexity", "kl_fp32_to_fp16",
         "token_agreement", "selfcheck_nli"],
        ["model", "precision", "evaluation_scope"],
        "Do not interpret SelfCheck as direct hallucination rate."
    ),
    (
        "F08_energy_memory",
        "Energy, throughput and memory resource trade-offs",
        ["energy_per_item", "throughput_images",
         "peak_memory"],
        ["model", "platform", "placement"],
        "Compare only compatible energy observation boundaries."
    ),
]

notes = [
    (
        "N01_models", "2",
        "Three vision and three language workloads, "
        "with easy/medium/hard groupings.",
        "paper_protocol", 0
    ),
    (
        "N02_vision_quality", "2-3",
        "Paper describes ImageNetV2 evaluation using "
        "10000 images per model/platform. "
        "Not a new Dataset V3 measurement.",
        "paper_reported_protocol", 0
    ),
    (
        "N03_language_quality", "3-4",
        "Perplexity, FP32-to-FP16 KL, token agreement "
        "and SelfCheck NLI are distinct evidence types.",
        "paper_reported_protocol", 0
    ),
    (
        "N04_power_boundaries", "2,5",
        "Apple SoC, Intel CPU package and NVIDIA GPU "
        "board power boundaries are not directly equivalent.",
        "paper_reported_limitation", 0
    ),
    (
        "N05_pipeline", "3,5",
        "Paper reports sequential and overlapped pipeline "
        "measurements; not yet present as validated "
        "T19 Dataset V3 pipeline measurements.",
        "paper_reported_experiment", 0
    ),
    (
        "N06_scheduler", "4-8",
        "Historical offline scheduler study remains "
        "outside current Dataset V3 construction scope.",
        "scope_boundary", 0
    ),
    (
        "N07_fitness", "user_specification",
        "u1 completion > u2 energy > u3 throughput > "
        "u4 memory recommendation; weights sum to one. "
        "Metric normalization and memory semantics pending. "
        "This is user-provided and is not attributed "
        "to the paper's separate QoS selection equations.",
        "user_defined_objective", 0
    ),
]

now = datetime.now(
    timezone.utc
).isoformat()

try:
    db.execute("BEGIN IMMEDIATE")

    db.executemany(
        "INSERT INTO research_models VALUES (?,?,?,?,?,?)",
        models
    )

    db.executemany(
        "INSERT INTO experiment_dimensions VALUES (?,?,?,?,?,?)",
        [
            (key, json.dumps(values), applies,
             source, status, note)
            for key, values, applies, source, status, note
            in dimensions
        ]
    )

    db.executemany(
        "INSERT INTO measurement_registry VALUES (?,?,?,?,?)",
        metrics
    )

    db.executemany(
        "INSERT INTO figure_registry VALUES (?,?,?,?,?,?)",
        [
            (key, title, json.dumps(required),
             json.dumps(groups), "planned", caution)
            for key, title, required, groups, caution
            in figures
        ]
    )

    db.executemany(
        "INSERT INTO evidence_notes VALUES (?,?,?,?,?)",
        notes
    )

    db.execute(
        "INSERT INTO schema_extensions VALUES (?,?,?,?,?)",
        (
            "T19_3_paper_dimensions_v1",
            now,
            str(SOURCE_DB),
            "Evidence-Aware and QoS-Constrained "
            "Resource Selection for Heterogeneous "
            "Edge AI Inference",
            0
        )
    )

    db.commit()

except BaseException:
    db.rollback()
    db.close()
    raise

assert db.execute(
    "PRAGMA integrity_check"
).fetchone()[0] == "ok"

assert db.execute(
    "PRAGMA foreign_key_check"
).fetchall() == []

after_states, after_results = source_state(db)

assert after_states == before_states
assert after_results == before_results

assert db.execute(
    "SELECT COUNT(*) FROM research_models"
).fetchone()[0] == 6

assert db.execute(
    "SELECT COUNT(*) FROM figure_registry"
).fetchone()[0] == 8

assert db.execute(
    "SELECT COUNT(*) FROM measurement_registry"
).fetchone()[0] == len(metrics)

assert db.execute(
    "SELECT COUNT(*) FROM experiment_dimensions"
).fetchone()[0] == len(dimensions)

for task_id, path, expected in after_results:
    assert sha(path) == expected, task_id

db.close()

# The independent snapshot is stored as one complete file.
db = sqlite3.connect(str(NEW_DB))
mode = db.execute(
    "PRAGMA journal_mode=DELETE"
).fetchone()[0]
assert mode == "delete"
db.close()

# Reopen and verify the stored snapshot.
db = sqlite3.connect(
    f"file:{NEW_DB}?mode=ro",
    uri=True
)
assert db.execute(
    "PRAGMA integrity_check"
).fetchone()[0] == "ok"
assert source_state(db) == (
    before_states, before_results
)
db.close()

summary = {
    "protocol": "T19_3_paper_schema_extension_v1",
    "created_utc": now,
    "source_database": str(SOURCE_DB),
    "extended_database": str(NEW_DB),
    "vision_models": 3,
    "text_models": 3,
    "experiment_dimensions": len(dimensions),
    "registered_metrics": len(metrics),
    "planned_figures": len(figures),
    "paper_evidence_notes": len(notes),
    "existing_tasks_preserved": 327336,
    "completed_tasks_preserved": 6,
    "planned_tasks_preserved": 327330,
    "additional_benchmark_runs": 0,
    "new_metrics_measured": 0,
    "text_layer_manifests_verified": False,
    "precision_support_validated": False,
    "scheduler_in_scope": False,
    "user_fitness": {
        "components": [
            "task_completion",
            "energy",
            "throughput",
            "memory_recommendation"
        ],
        "weight_order": "u1>u2>u3>u4>0",
        "weight_sum": 1,
        "weights_selected": False,
        "normalization_selected": False
    },
    "uploaded_paper_pdf_included": False,
    "source_database_modified": False,
    "runner_repointed_to_new_database": False,
}

save_json(OUT / "extension_summary.json", summary)

checksums = {
    "dataset_v3_extended.sqlite": sha(NEW_DB),
    "extension_summary.json":
        sha(OUT / "extension_summary.json"),
    "extend_dataset.py": sha(__file__),
}
save_json(OUT / "checksums.json", checksums)

for name, expected in checksums.items():
    assert sha(OUT / name) == expected

print("=== T19.3 DATASET EXTENSION ===")
print("SOURCE DATA PRESERVED: PASS")
print("SQLITE INTEGRITY: PASS")
print("RESULT SHA256: 6/6 PASS")
print("VISION MODELS:", 3)
print("TEXT MODELS:", 3)
print("EXPERIMENT DIMENSIONS:", len(dimensions))
print("MEASUREMENT METRICS:", len(metrics))
print("GRAPH DEFINITIONS:", len(figures))
print("PAPER EVIDENCE NOTES:", len(notes))
print("ORIGINAL TASKS PRESERVED: 327336")
print("COMPLETED RESULTS PRESERVED: 6")
print("NEW INFERENCE MEASUREMENTS: 0")
print("CHECKSUMS: PASS")
print()
print("T19.3 PAPER DATASET EXTENSION: PASS")
print("EXTENDED DATABASE:", NEW_DB)
print("RESULT DIRECTORY:", OUT)
