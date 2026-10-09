import csv
import statistics
from collections import defaultdict
from pathlib import Path

SRC = Path("results/unified_characterization.csv")
OUT = Path("results/scheduler_candidate_registry.csv")


def valid(v):
    return str(v).strip() not in ("", "nan", "None")


def fnum(v):
    try:
        return float(v)
    except Exception:
        return None


def median_numeric(values):
    vals = [fnum(v) for v in values if fnum(v) is not None]
    return statistics.median(vals) if vals else ""


def max_numeric(values):
    vals = [fnum(v) for v in values if fnum(v) is not None]
    return max(vals) if vals else ""


with SRC.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))


# ------------------------------------------------------------------
# 1. Recover missing workload shapes.
#
# Example:
# Qwen3 CPU power rows contain workload=long_single/batch but may not
# contain context/output/batch explicitly. Scaling rows contain both.
# ------------------------------------------------------------------

shape_map = defaultdict(set)

for r in rows:
    workload = r.get("workload", "")

    if not valid(workload):
        continue

    resolution = r.get("resolution", "")
    context = r.get("context_tokens", "")
    output = r.get("output_tokens", "")
    batch = r.get("batch", "")

    has_vision_shape = valid(resolution) and valid(batch)
    has_text_shape = (
        valid(context)
        and valid(output)
        and valid(batch)
    )

    if not (has_vision_shape or has_text_shape):
        continue

    key = (
        r.get("model", ""),
        r.get("device", ""),
        r.get("backend", ""),
        r.get("precision", ""),
        r.get("quantization", ""),
        workload,
    )

    shape_map[key].add(
        (resolution, context, output, batch)
    )


for r in rows:
    workload = r.get("workload", "")

    if not valid(workload):
        continue

    key = (
        r.get("model", ""),
        r.get("device", ""),
        r.get("backend", ""),
        r.get("precision", ""),
        r.get("quantization", ""),
        workload,
    )

    shapes = shape_map.get(key, set())

    # Only fill automatically when mapping is unambiguous.
    if len(shapes) != 1:
        continue

    resolution, context, output, batch = next(iter(shapes))

    if not valid(r.get("resolution", "")):
        r["resolution"] = resolution

    if not valid(r.get("context_tokens", "")):
        r["context_tokens"] = context

    if not valid(r.get("output_tokens", "")):
        r["output_tokens"] = output

    if not valid(r.get("batch", "")):
        r["batch"] = batch



# ------------------------------------------------------------------
# CONVNEXT_MPS_SHAPE_REPAIR
#
# Canonical sustained-power rows lost resolution and batch during
# normalization. Recover them from the original 18-row power results.
# Use existing sweep rows' numeric serialization to ensure that
# corresponding power and latency records share the same identity.
# ------------------------------------------------------------------

raw_path = Path(
    "results/convnext_large_vision_hard_mac_mps_power.csv"
)

raw_shapes = defaultdict(set)
raw_counts = defaultdict(int)

with raw_path.open(newline="", encoding="utf-8") as f:
    for source in csv.DictReader(f):
        identity = (source["precision"], source["profile"])
        raw_shapes[identity].add((
            int(source["resolution"]),
            int(source["batch"]),
        ))
        raw_counts[identity] += 1

if (
    len(raw_shapes) != 6
    or any(len(shapes) != 1 for shapes in raw_shapes.values())
    or any(count != 3 for count in raw_counts.values())
):
    raise RuntimeError("Unexpected ConvNeXt MPS raw profile shapes")

sweep_shapes = defaultdict(set)

for r in rows:
    if (
        r.get("source_file")
        != "convnext_large_vision_hard_mac_mps_sweep.csv"
        or r.get("model") != "ConvNeXt-Large"
        or r.get("device") != "Apple M4"
        or r.get("backend") != "mps"
    ):
        continue

    if not valid(r.get("resolution")) or not valid(r.get("batch")):
        continue

    shape_identity = (
        r["precision"],
        int(float(r["resolution"])),
        int(float(r["batch"])),
    )

    sweep_shapes[shape_identity].add((
        r["resolution"],
        r["batch"],
    ))

recovered = 0

for r in rows:
    if r.get("source_file") != (
        "convnext_large_vision_hard_mac_mps_power_canonical.csv"
    ):
        continue

    if (
        r.get("model") != "ConvNeXt-Large"
        or r.get("device") != "Apple M4"
        or r.get("backend") != "mps"
    ):
        raise RuntimeError("Unexpected canonical source identity")

    profile_key = (r["precision"], r["profile"])

    if profile_key not in raw_shapes:
        raise RuntimeError(f"Missing raw profile: {profile_key}")

    resolution, batch = next(iter(raw_shapes[profile_key]))

    lookup = (r["precision"], resolution, batch)
    serialized = sweep_shapes.get(lookup, set())

    if len(serialized) != 1:
        raise RuntimeError(
            f"Cannot uniquely match canonical profile: {lookup}"
        )

    sweep_resolution, sweep_batch = next(iter(serialized))

    for field, expected in (
        ("resolution", resolution),
        ("batch", batch),
    ):
        existing = r.get(field, "")
        if valid(existing) and float(existing) != expected:
            raise RuntimeError(
                f"Conflicting {field} for {profile_key}"
            )

    r["resolution"] = sweep_resolution
    r["batch"] = sweep_batch
    recovered += 1

if recovered != 6:
    raise RuntimeError(
        f"Expected 6 recovered profiles; found {recovered}"
    )

print(f"ConvNeXt MPS canonical shape recovery: {recovered}/6")

# ------------------------------------------------------------------
# 2. Physical configuration identity.
#
# profile/workload/measurement_mode are NOT identity fields.
# They describe how a configuration was measured.
# ------------------------------------------------------------------

KEYS = [
    "modality",
    "difficulty",
    "model",
    "device",
    "backend",
    "precision",
    "quantization",
    "resolution",
    "context_tokens",
    "output_tokens",
    "batch",
    "threads",
]

groups = defaultdict(list)

for r in rows:
    key = tuple(r.get(k, "") for k in KEYS)
    groups[key].append(r)


def sustained_rows(group):
    return [
        r for r in group
        if r.get("is_sustained") == "1"
    ]


def rows_with(group, metric):
    return [
        r for r in group
        if valid(r.get(metric, ""))
    ]


out_rows = []

for key, group in groups.items():
    merged = dict(zip(KEYS, key))

    sustained = sustained_rows(group)

    # Prefer sustained measurements for throughput where available.
    throughput_group = rows_with(sustained, "throughput")
    if not throughput_group:
        throughput_group = rows_with(group, "throughput")

    merged["throughput"] = median_numeric(
        [r["throughput"] for r in throughput_group]
    )

    units = {
        r.get("throughput_unit", "")
        for r in throughput_group
        if valid(r.get("throughput_unit", ""))
    }
    merged["throughput_unit"] = (
        next(iter(units)) if len(units) == 1 else ""
    )

    # Latency uses direct measured values only.
    latency_group = rows_with(group, "latency_sec")
    merged["latency_sec"] = median_numeric(
        [r["latency_sec"] for r in latency_group]
    )

    # Power/energy must come from sustained measurements if possible.
    power_group = rows_with(sustained, "power_w")
    if not power_group:
        power_group = rows_with(group, "power_w")

    energy_group = rows_with(
        sustained,
        "energy_per_item_j",
    )
    if not energy_group:
        energy_group = rows_with(
            group,
            "energy_per_item_j",
        )

    merged["power_w"] = median_numeric(
        [r["power_w"] for r in power_group]
    )

    merged["energy_per_item_j"] = median_numeric(
        [
            r["energy_per_item_j"]
            for r in energy_group
        ]
    )

    merged["energy_per_run_j"] = median_numeric(
        [
            r["energy_per_run_j"]
            for r in sustained
            if valid(r.get("energy_per_run_j", ""))
        ]
    )

    # Conservative memory constraint:
    # use the largest measured footprint for same physical config.
    merged["memory_allocated_mb"] = max_numeric(
        [
            r["memory_allocated_mb"]
            for r in group
            if valid(r.get("memory_allocated_mb", ""))
        ]
    )

    merged["memory_peak_mb"] = max_numeric(
        [
            r["memory_peak_mb"]
            for r in group
            if valid(r.get("memory_peak_mb", ""))
        ]
    )

    merged["device_memory_used_mb"] = max_numeric(
        [
            r["device_memory_used_mb"]
            for r in group
            if valid(r.get("device_memory_used_mb", ""))
        ]
    )

    boundaries = {
        r.get("energy_boundary", "")
        for r in energy_group
        if valid(r.get("energy_boundary", ""))
    }

    merged["energy_boundary"] = (
        next(iter(boundaries))
        if len(boundaries) == 1
        else ";".join(sorted(boundaries))
    )

    merged["has_latency"] = int(
        valid(merged["latency_sec"])
    )

    merged["has_throughput"] = int(
        valid(merged["throughput"])
    )

    merged["has_power"] = int(
        valid(merged["power_w"])
    )

    merged["has_energy"] = int(
        valid(merged["energy_per_item_j"])
    )

    merged["has_memory"] = int(
        valid(merged["memory_allocated_mb"])
        or valid(merged["memory_peak_mb"])
        or valid(merged["device_memory_used_mb"])
    )

    merged["scheduler_core"] = int(
        merged["has_throughput"]
        and merged["has_energy"]
    )

    merged["workloads"] = ";".join(
        sorted({
            r.get("workload", "")
            for r in group
            if valid(r.get("workload", ""))
        })
    )

    merged["profiles"] = ";".join(
        sorted({
            r.get("profile", "")
            for r in group
            if valid(r.get("profile", ""))
        })
    )

    merged["measurement_modes"] = ";".join(
        sorted({
            r.get("measurement_mode", "")
            for r in group
            if valid(r.get("measurement_mode", ""))
        })
    )

    merged["source_files"] = ";".join(
        sorted({
            r.get("source_file", "")
            for r in group
            if valid(r.get("source_file", ""))
        })
    )

    out_rows.append(merged)


FIELDS = (
    KEYS
    + [
        "throughput",
        "throughput_unit",
        "latency_sec",
        "power_w",
        "energy_per_run_j",
        "energy_per_item_j",
        "memory_allocated_mb",
        "memory_peak_mb",
        "device_memory_used_mb",
        "energy_boundary",
        "has_latency",
        "has_throughput",
        "has_power",
        "has_energy",
        "has_memory",
        "scheduler_core",
        "workloads",
        "profiles",
        "measurement_modes",
        "source_files",
    ]
)

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, lineterminator="\n", fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(out_rows)


core = [
    r for r in out_rows
    if r["scheduler_core"] == 1
]

print(f"All unique configs : {len(out_rows)}")
print(f"Scheduler core     : {len(core)}")
print(f"Saved              : {OUT}")

summary = defaultdict(
    lambda: {
        "core": 0,
        "memory": 0,
        "latency": 0,
    }
)

for r in core:
    k = (
        r["modality"],
        r["difficulty"],
        r["model"],
    )

    summary[k]["core"] += 1
    summary[k]["memory"] += r["has_memory"]
    summary[k]["latency"] += r["has_latency"]

print("\nCORE COVERAGE")

for key, s in sorted(summary.items()):
    print(
        f"{key[0]:8s} | "
        f"{key[1]:8s} | "
        f"{key[2]:28s} | "
        f"core={s['core']:3d} | "
        f"lat={s['latency']:3d} | "
        f"mem={s['memory']:3d}"
    )
