import csv
import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(sys.argv[1]).resolve()

BASE = Path("results/benchmark_v3")
VALIDATION = BASE / (
    "validation/t05_20261015T173344Z_2438/metadata.json"
)
INVENTORY = BASE / (
    "inventory/inv_20261015T175122Z_2438/"
    "structural_inventory.csv"
)
MANIFEST_DIR = Path("configs/partition_manifests_v2")

MODELS = {
    "resnet50": {
        "manifest": "resnet50__stage6.json",
        "groups": 6,
        "expected_sha":
            "39137a95b178a2a4b7d59babaa6326c8d6e06f95ff8c9a0a801cb42aa51fa7b8",
        "stage_lengths": [1, 1, 1, 1, 1, 1],
    },
    "convnext_base": {
        "manifest": "convnext_base__fine41.json",
        "groups": 41,
        "expected_sha":
            "f67aa6303f8f9e48367561083a8d0ce8665892ef92d0fea49c122a8f0614506e",
        "stage_lengths": [1, 3, 4, 28, 4, 1],
    },
    "convnext_large": {
        "manifest": "convnext_large__fine41.json",
        "groups": 41,
        "expected_sha":
            "9741ded73c174f4e4e032077728629ba7dbe331d6987eb3ce0c2bb118e0dda4c",
        "stage_lengths": [1, 3, 4, 28, 4, 1],
    },
}

THREADS = [1, 2, 4, 6, 8, 10]
BATCH_SIZES = [1, 2, 4, 8]
MICROBATCH_SIZES = [1, 2, 4, 8]

EXPECTED_WEIGHTS = {
    "resnet50":
        "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
    "convnext_base":
        "6075fbad31c688f9ed013922a59daf0fc2f4dc12e6e43bc01fa840c7bd1ca2cd",
    "convnext_large":
        "ea097f822cdb2cab40905cec3949f4bdf6d6bf1d546c3ec5e43f0b1c6275e144",
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(part)
    return h.hexdigest()

def stable_id(prefix, obj):
    encoded = json.dumps(
        obj, sort_keys=True,
        separators=(",", ":")
    ).encode()
    return prefix + hashlib.sha256(
        encoded
    ).hexdigest()

def write_json(path, data):
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )
    with temp.open("w", encoding="utf-8") as f:
        json.dump(
            data, f, indent=2,
            ensure_ascii=False
        )
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def write_csv(path, fields, rows):
    temp = path.with_suffix(
        path.suffix + ".tmp"
    )
    with temp.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def read_csv(path):
    with Path(path).open(
        newline="", encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))

def coarse_bits():
    return (
        "".join(x)
        for x in itertools.product(
            "CA", repeat=6
        )
    )

def expand_coarse(bits, lengths):
    assert len(bits) == 6
    return "".join(
        bit * count
        for bit, count in zip(
            bits, lengths
        )
    )

def fine_bits_at_most_two(groups):
    boundaries = list(range(1, groups))

    for number_of_cuts in (0, 1, 2):
        for cut_positions in itertools.combinations(
            boundaries, number_of_cuts
        ):
            for starting_device in "CA":
                bits = []
                current = starting_device
                cuts = set(cut_positions)

                for group_number in range(
                    1, groups + 1
                ):
                    if (
                        group_number > 1
                        and group_number - 1 in cuts
                    ):
                        current = (
                            "A" if current == "C"
                            else "C"
                        )
                    bits.append(current)

                yield "".join(bits)

def transitions(bits):
    result = []

    for boundary in range(
        1, len(bits)
    ):
        left = bits[boundary - 1]
        right = bits[boundary]

        if left != right:
            result.append({
                "after_group": boundary,
                "direction": (
                    ("cpu" if left == "C"
                     else "accelerator")
                    + "->"
                    + ("cpu" if right == "C"
                       else "accelerator")
                )
            })

    return result

def source_validation():
    assert VALIDATION.is_file()
    assert INVENTORY.is_file()

    validation = json.loads(
        VALIDATION.read_text()
    )
    inventory = read_csv(INVENTORY)

    expected_fine_modules = (
        [["features.0"]]
        + [
            [f"features.1.{i}"]
            for i in range(3)
        ]
        + [["features.2"]]
        + [
            [f"features.3.{i}"]
            for i in range(3)
        ]
        + [["features.4"]]
        + [
            [f"features.5.{i}"]
            for i in range(27)
        ]
        + [["features.6"]]
        + [
            [f"features.7.{i}"]
            for i in range(3)
        ]
        + [["avgpool", "classifier"]]
    )

    assert len(expected_fine_modules) == 41

    for model, spec in MODELS.items():
        manifest_path = (
            MANIFEST_DIR / spec["manifest"]
        )

        assert manifest_path.is_file(), (
            f"Missing: {manifest_path}"
        )

        actual_sha = sha(manifest_path)
        assert actual_sha == spec[
            "expected_sha"
        ], f"Manifest changed: {model}"

        manifest = json.loads(
            manifest_path.read_text()
        )

        assert manifest["group_count"] == (
            spec["groups"]
        )
        assert len(manifest["groups"]) == (
            spec["groups"]
        )
        assert sum(spec["stage_lengths"]) == (
            spec["groups"]
        )

        if model.startswith("convnext"):
            actual_modules = [
                group["modules"]
                for group in manifest["groups"]
            ]
            assert actual_modules == (
                expected_fine_modules
            ), f"Fine group mismatch: {model}"

        weight_sha = validation[
            "models"
        ][model]["weight_sha256"]

        assert weight_sha == (
            EXPECTED_WEIGHTS[model]
        ), f"Weight SHA mismatch: {model}"

        stages = [
            row for row in inventory
            if row["model"] == model
        ]

        assert len(stages) == 6
        assert {
            row["stage_id"]
            for row in stages
        } == {
            f"S{i}" for i in range(6)
        }

        assert all(
            row["manifest_sha256"]
            == actual_sha
            for row in stages
        )

    return validation

print(
    "=== T17 SOURCE VALIDATION ===",
    flush=True
)
validation = source_validation()
print(
    "MODEL WEIGHTS, MANIFESTS, "
    "AND STAGE INVENTORY: PASS"
)

placement_rows = []
model_rows = []
global_ids = set()

for model, spec in MODELS.items():
    groups = spec["groups"]
    lengths = spec["stage_lengths"]

    coarse = {
        expand_coarse(bits, lengths)
        for bits in coarse_bits()
    }

    assert len(coarse) == 64

    if groups == 41:
        fine = set(
            fine_bits_at_most_two(groups)
        )
        assert len(fine) == 1642
        all_bits = coarse | fine
        assert len(coarse & fine) == 32
        assert len(all_bits) == 1674
    else:
        fine = set()
        all_bits = coarse
        assert len(all_bits) == 64

    for bits in sorted(all_bits):
        cuts = transitions(bits)
        in_coarse = bits in coarse
        in_fine = bits in fine

        coarse_alias = ""

        if in_coarse:
            stage_alias = []

            position = 0
            for length in lengths:
                segment = bits[
                    position:position + length
                ]
                assert len(set(segment)) == 1
                stage_alias.append(segment[0])
                position += length

            coarse_alias = "".join(stage_alias)
            assert len(coarse_alias) == 6

        if in_coarse and in_fine:
            origin = "coarse_and_fine"
        elif in_coarse:
            origin = "coarse_only"
        else:
            origin = "fine_only"

        placement_id = stable_id(
            "place_",
            {
                "model": model,
                "group_mapping": bits,
                "manifest_sha256":
                    spec["expected_sha"],
            }
        )

        assert placement_id not in global_ids
        global_ids.add(placement_id)

        placement_rows.append({
            "placement_id": placement_id,
            "model": model,
            "group_count": groups,
            "group_mapping": bits,
            "coarse_6stage_mapping":
                coarse_alias,
            "source_family": origin,
            "transition_count": len(cuts),
            "cut_after_groups": json.dumps(
                [
                    c["after_group"]
                    for c in cuts
                ]
            ),
            "transfer_directions": json.dumps(
                [
                    c["direction"]
                    for c in cuts
                ]
            ),
            "uses_cpu": int(
                "C" in bits
            ),
            "uses_accelerator": int(
                "A" in bits
            ),
            "manifest_sha256":
                spec["expected_sha"],
            "execution_validated": 0,
            "measured": 0,
        })

    model_rows.append({
        "model": model,
        "modality": "vision",
        "model_identity_status": "verified",
        "stage_count": 6,
        "group_count": groups,
        "manifest_path": str(
            MANIFEST_DIR / spec["manifest"]
        ),
        "manifest_sha256":
            spec["expected_sha"],
        "weight_sha256":
            EXPECTED_WEIGHTS[model],
        "placement_candidates": len(all_bits),
        "execution_backend_mac": "mps",
        "execution_backend_msi":
            "cuda_pending_validation",
    })

    print(
        f"{model}: {len(all_bits)} "
        "unique placement candidates",
        flush=True
    )

assert len(placement_rows) == 3412

counts = Counter(
    row["transition_count"]
    for row in placement_rows
)

assert dict(sorted(counts.items())) == {
    0: 6,
    1: 170,
    2: 3140,
    3: 60,
    4: 30,
    5: 6,
}

print("\nPLACEMENT DEDUPLICATION: PASS")
print("UNIQUE VISION PLACEMENTS: 3412")

workloads = []

for batch in BATCH_SIZES:
    for microbatch in MICROBATCH_SIZES:
        if microbatch > batch:
            continue

        if batch % microbatch != 0:
            continue

        count = batch // microbatch

        workloads.append({
            "workload_id":
                f"batch{batch}_micro{microbatch}",
            "batch_size": batch,
            "microbatch_size": microbatch,
            "microbatch_count": count,
            "sequential_candidate": 1,
            "pipeline_candidate": int(
                count >= 2
            ),
            "workload_data_status":
                "input_selection_pending",
        })

assert len(workloads) == 10
assert sum(
    int(r["pipeline_candidate"])
    for r in workloads
) == 6

heterogeneous = sum(
    row["uses_cpu"] == 1
    and row["uses_accelerator"] == 1
    for row in placement_rows
)

assert heterogeneous == 3406

sequential_candidates = (
    len(placement_rows)
    * len(THREADS)
    * len(workloads)
)

pipeline_candidates = (
    heterogeneous
    * len(THREADS)
    * sum(
        int(row["pipeline_candidate"])
        for row in workloads
    )
)

assert sequential_candidates == 204720
assert pipeline_candidates == 122616

schema = {
    "schema_version": "dataset_v3_t17_draft_1",
    "purpose": (
        "Reproducible multi-objective benchmark "
        "dataset, not a scheduler algorithm"
    ),
    "logical_devices": {
        "C": "cpu",
        "A": "accelerator",
        "mac_accelerator": "mps",
        "msi_accelerator":
            "cuda_pending_verification",
    },
    "fitness_objectives": {
        "order": [
            "task_completion",
            "energy",
            "throughput",
            "memory_recommendation",
        ],
        "weight_constraints": [
            "u1 > u2 > u3 > u4 > 0",
            "u1 + u2 + u3 + u4 = 1",
        ],
        "actual_weight_values":
            None,
        "normalization_method":
            None,
        "memory_recommendation_definition":
            "not_yet_finalized",
        "fitness_score_calculated":
            False,
    },
    "measurement_families": {
        "task_and_quality": [
            "run_success",
            "failure_reason",
            "numerical_allclose",
            "max_abs_logit_error",
            "top1_agreement",
            "top1_accuracy",
            "top5_accuracy",
            "text_task_quality_metric",
        ],
        "latency": [
            "latency_ns_per_call",
            "latency_median_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "first_result_latency_ms",
        ],
        "throughput": [
            "images_per_second",
            "requests_per_second",
            "tokens_per_second",
            "measurement_window_seconds",
        ],
        "power_and_energy": [
            "power_watts",
            "energy_joules",
            "energy_per_item_joules",
            "idle_baseline_watts",
            "measurement_scope",
            "sensor_name",
            "sensor_permission_status",
        ],
        "memory": [
            "rss_bytes",
            "mps_allocated_bytes",
            "mps_driver_bytes",
            "cuda_allocated_bytes",
            "cuda_reserved_bytes",
            "peak_memory_bytes",
            "memory_capacity_bytes",
            "memory_budget_bytes",
        ],
        "transfer": [
            "source_device",
            "destination_device",
            "boundary_after_group",
            "tensor_shape",
            "tensor_dtype",
            "logical_bytes",
            "transfer_latency_ns",
            "synchronization_method",
            "tensor_roundtrip_max_abs_error",
        ],
        "pipeline": [
            "execution_mode",
            "microbatch_count",
            "microbatch_start_ns",
            "microbatch_end_ns",
            "stage_start_ns",
            "stage_end_ns",
            "observed_overlap_ns",
            "pipeline_throughput",
        ],
        "provenance": [
            "model_id",
            "placement_id",
            "workload_id",
            "run_id",
            "process_uuid",
            "device_id",
            "backend",
            "cpu_threads",
            "precision",
            "model_weight_sha256",
            "manifest_sha256",
            "software_versions",
            "sample_id",
            "sample_sha256",
            "timestamp_utc",
        ],
    },
    "metric_status_values": [
        "measured",
        "not_measured",
        "unsupported",
        "permission_denied",
        "failed",
        "not_applicable",
    ],
    "run_status_values": [
        "pending",
        "running",
        "completed",
        "failed",
        "timeout",
        "out_of_memory",
        "numerical_mismatch",
        "unsupported",
        "interrupted",
    ],
    "rules": [
        "Never impute a missing measurement as zero",
        "Retain slow and failed configurations",
        "Preserve raw timing and transfer samples",
        "Record units, power sensor and scope",
        "Do not add RSS and MPS memory counters",
        "Do not claim transfer bytes are measured duration",
        "Do not label a theoretical pipeline as observed overlap",
        "Keep quality and task completion separate",
        "Store fitness inputs before any weighted score",
        "Record independent process count separately "
        "from timed repetitions",
    ],
    "storage_plan_for_T18": {
        "authoritative_store": "SQLite",
        "journaling": "WAL",
        "durability_and_recovery":
            "transactional checkpoint and resume",
        "export": ["CSV", "Parquet", "XLSX"],
        "integrity": "SHA256 and database checks",
        "state": "planned_not_yet_implemented",
    },
}

try:
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    git_head = None

planning = {
    "protocol": "T17_design_only",
    "vision_models": 3,
    "text_model_slots": 3,
    "text_models_verified": 0,
    "text_model_names_and_manifests":
        "awaiting_confirmation",
    "model_specific_vision_placements": {
        r["model"]:
            r["placement_candidates"]
        for r in model_rows
    },
    "unique_vision_placements": 3412,
    "transition_histogram": {
        str(k): v
        for k, v in sorted(counts.items())
    },
    "cpu_thread_values": THREADS,
    "batch_sizes": BATCH_SIZES,
    "microbatch_sizes": MICROBATCH_SIZES,
    "valid_batch_microbatch_pairs":
        len(workloads),
    "pairs_with_multiple_microbatches": 6,
    "logical_sequential_candidates":
        sequential_candidates,
    "logical_pipeline_candidates":
        pipeline_candidates,
    "logical_total_candidates":
        sequential_candidates
        + pipeline_candidates,
    "precision_in_count": ["float32"],
    "independent_repeats_in_count": 1,
    "image_datasets_in_count": 1,
    "limitations": [
        "Candidate counts are not completed benchmarks",
        "Text models are excluded pending exact identities",
        "Runtime viability has not been established",
        "Pipeline overlap has not been implemented or validated",
        "Power and energy sensor availability remains unverified",
        "The count does not include dataset sample multiplicity",
        "No accelerator memory limit has been enforced",
        "No extra fine-group placements with >2 transitions",
    ],
}

placement_path = OUT / "placement_catalog.csv"
workload_path = OUT / "workload_catalog.csv"
model_path = OUT / "model_catalog.csv"
schema_path = OUT / "dataset_schema.json"
plan_path = OUT / "planning_summary.json"
metadata_path = OUT / "metadata.json"

write_csv(
    placement_path,
    list(placement_rows[0]),
    placement_rows,
)
write_csv(
    workload_path,
    list(workloads[0]),
    workloads,
)
write_csv(
    model_path,
    list(model_rows[0]),
    model_rows,
)
write_json(schema_path, schema)
write_json(plan_path, planning)

write_json(metadata_path, {
    "protocol": "T17_design_only_v1",
    "generated_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "git_head": git_head,
    "script_sha256": sha(__file__),
    "validation_metadata_sha256":
        sha(VALIDATION),
    "inventory_sha256": sha(INVENTORY),
    "manifest_sha256": {
        name: spec["expected_sha"]
        for name, spec in MODELS.items()
    },
    "no_inference_performed": True,
    "no_energy_measurements_performed": True,
    "no_legacy_files_modified": True,
})

artifacts = [
    Path(__file__),
    placement_path,
    workload_path,
    model_path,
    schema_path,
    plan_path,
    metadata_path,
]

write_json(
    OUT / "checksums.json",
    {
        file.name: sha(file)
        for file in artifacts
    },
)

for filename, expected in json.loads(
    (OUT / "checksums.json").read_text()
).items():
    assert sha(OUT / filename) == expected

assert len(read_csv(placement_path)) == 3412
assert len(read_csv(workload_path)) == 10
assert len(read_csv(model_path)) == 3

assert json.loads(
    plan_path.read_text()
)["logical_total_candidates"] == 327336

print("\n=== T17 DESIGN SUMMARY ===")
print("VISION MODELS VERIFIED: 3")
print("TEXT MODEL SLOTS PENDING: 3")
print("UNIQUE VISION PLACEMENTS: 3412")
print("VALID BATCH/MICROBATCH PAIRS: 10")
print("CPU THREAD SETTINGS: 6")
print(
    "SEQUENTIAL CANDIDATES:",
    sequential_candidates
)
print(
    "PIPELINE CANDIDATES:",
    pipeline_candidates
)
print(
    "TOTAL LOGICAL CANDIDATES:",
    sequential_candidates + pipeline_candidates
)
print("ACTUAL BENCHMARK RUNS: 0")
print("ENERGY MEASUREMENTS: 0")
print("CATALOG CHECKSUMS: PASS")
print("\nT17 EXPERIMENT DESIGN: PASS")
print("RESULT DIRECTORY:", OUT)
