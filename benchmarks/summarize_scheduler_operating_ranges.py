import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

SRC = Path(
    "results/scheduler_candidate_registry_memory_enriched.csv"
)
CONFIG = Path("configs/workload_profiles.json")
OUT = Path("results/scheduler_operating_ranges.csv")

with SRC.open(newline="", encoding="utf-8") as f:
    registry = list(csv.DictReader(f))

profiles = json.loads(CONFIG.read_text(encoding="utf-8"))

def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}

def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None

def integer(value):
    x = number(value)
    if x is None:
        return None
    if not x.is_integer():
        raise ValueError(f"Non-integer shape: {value}")
    return int(x)

def model_id(value):
    name = value.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]", "", name)

def shape_profile(row):
    config_key = f"{row['modality']}_{row['difficulty']}"
    config = profiles.get(config_key)

    if config is None:
        return "unmapped"

    if model_id(row["model"]) != model_id(config["model"]):
        return "unmapped"

    batch = integer(row["batch"])

    if row["modality"] == "vision":
        actual = (integer(row["resolution"]), batch)

        if None in actual:
            return "unknown_shape"

        for name, p in config["profiles"].items():
            if actual == (int(p["resolution"]), int(p["batch"])):
                return name

    else:
        actual = (
            integer(row["context_tokens"]),
            integer(row["output_tokens"]),
            batch,
        )

        if None in actual:
            return "unknown_shape"

        for name, p in config["profiles"].items():
            expected = (
                int(p["context"]),
                int(p["output"]),
                int(p["batch"]),
            )
            if actual == expected:
                return name

    return "noncanonical"

def normalized_shape(value):
    n = integer(value)
    return "" if n is None else str(n)

def metric_stats(rows, column):
    vals = [
        x for row in rows
        if (x := number(row.get(column))) is not None
    ]

    if not vals:
        return len(vals), "", "", ""

    return (
        len(vals),
        min(vals),
        statistics.median(vals),
        max(vals),
    )

MEMORY_FIELDS = [
    "memory_allocated_mb",
    "memory_peak_mb",
    "device_memory_used_mb",
    "process_peak_rss_mib_max",
    "macos_peak_footprint_mib_max",
    "windows_peak_working_set_mib_max",
    "windows_sampled_private_commit_mib_max",
]

GROUP_FIELDS = [
    "modality",
    "difficulty",
    "model",
    "shape_profile",
    "resolution",
    "context_tokens",
    "output_tokens",
    "batch",
    "device",
    "backend",
    "precision",
    "quantization",
    "throughput_unit",
    "energy_boundary",
]

core = [
    row for row in registry
    if is_true(row["scheduler_core"])
]

groups = defaultdict(list)

for row in core:
    group_data = dict(row)
    group_data["shape_profile"] = shape_profile(row)

    for field in (
        "resolution",
        "context_tokens",
        "output_tokens",
        "batch",
    ):
        group_data[field] = normalized_shape(row[field])

    key = tuple(group_data[field] for field in GROUP_FIELDS)
    groups[key].append(group_data)

out_rows = []

for key, members in sorted(groups.items()):
    result = dict(zip(GROUP_FIELDS, key))

    result["candidate_count"] = len(members)
    result["threads"] = ";".join(
        sorted({
            normalized_shape(row["threads"])
            for row in members
            if row["threads"]
        }, key=int)
    )

    metrics = [
        ("throughput", "throughput"),
        ("latency", "latency_sec"),
        ("energy", "energy_per_item_j"),
    ]

    for label, source in metrics:
        count, low, med, high = metric_stats(members, source)

        result[f"{label}_count"] = count
        result[f"{label}_min"] = low
        result[f"{label}_median"] = med
        result[f"{label}_max"] = high

    result["memory_any_count"] = sum(
        is_true(row["has_any_memory_observation"])
        for row in members
    )

    for field in MEMORY_FIELDS:
        result[f"{field}_count"] = sum(
            number(row.get(field)) is not None
            for row in members
        )

    out_rows.append(result)

assert sum(
    row["candidate_count"] for row in out_rows
) == len(core), "Candidate count mismatch"

fieldnames = (
    GROUP_FIELDS
    + [
        "threads",
        "candidate_count",
        "throughput_count",
        "throughput_min",
        "throughput_median",
        "throughput_max",
        "latency_count",
        "latency_min",
        "latency_median",
        "latency_max",
        "energy_count",
        "energy_min",
        "energy_median",
        "energy_max",
        "memory_any_count",
    ]
    + [f"{field}_count" for field in MEMORY_FIELDS]
)

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, lineterminator="\n", fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

coverage = defaultdict(lambda: [0, 0, 0])

for row in core:
    key = (row["model"], shape_profile(row))
    item = coverage[key]

    item[0] += 1
    item[1] += bool(row["latency_sec"])
    item[2] += is_true(row["has_any_memory_observation"])

print("=== OPERATING RANGES: SHAPE-AWARE ===")
print("Core candidates       :", len(core))
print("Execution groups      :", len(out_rows))
print(
    "Core with memory      :",
    sum(is_true(r["has_any_memory_observation"]) for r in core),
)
print(
    "Unknown-shape core    :",
    sum(shape_profile(r) == "unknown_shape" for r in core),
)

print("\nPROFILE COVERAGE (counts only)")
print(
    f"{'MODEL':28s} {'PROFILE':13s} "
    f"{'N':>4s} {'LAT':>5s} {'MEM':>5s}"
)

for (model, profile), (n, latency, memory) in sorted(
    coverage.items()
):
    print(
        f"{model:28s} {profile:13s} "
        f"{n:4d} {latency:5d} {memory:5d}"
    )

print(f"\nSaved: {OUT}")
