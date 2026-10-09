import csv
import statistics
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

BASE = Path("results/scheduler_candidate_registry.csv")
OUT = Path("results/scheduler_candidate_registry_memory_enriched.csv")

SOURCES = [
    ("results/qwen3_q4_mac_cpu_memory.csv", "Apple M4", "cpu", "mac"),
    ("results/qwen3_q4_mac_cpu_memory_recheck.csv", "Apple M4", "cpu", "mac"),
    ("results/qwen3_q4_mac_metal_memory.csv", "Apple M4", "llama.cpp/metal", "mac"),
    ("results/qwen3_q4_msi_cpu_memory.csv", "Intel i7-11800H", "cpu", "windows"),
]

CPU_SHAPES = {
    "long_single": (128, 128, 1),
    "batch": (128, 64, 4),
}

METAL_SHAPES = {
    "light": (128, 64, 1),
    "medium": (512, 128, 2),
    "heavy": (1024, 128, 4),
}

NEW_COLUMNS = [
    "process_peak_rss_mib_median",
    "process_peak_rss_mib_max",
    "macos_peak_footprint_mib_max",
    "windows_peak_working_set_mib_median",
    "windows_peak_working_set_mib_max",
    "windows_sampled_private_commit_mib_max",
    "memory_observation_count",
    "memory_source_files",
    "has_any_memory_observation",
]


def as_int(value):
    value = str(value).strip()
    if not value:
        return None
    number = Decimal(value)
    if number != int(number):
        raise ValueError(f"Non-integer configuration value: {value}")
    return int(number)


def shape(row):
    return tuple(
        as_int(row.get(field, ""))
        for field in (
            "context_tokens",
            "output_tokens",
            "batch",
            "threads",
        )
    )


def key(device, backend, row):
    return (
        device,
        backend,
        "qwen3-1.7b",
        "q4_k_m",
        *shape(row),
    )


observations = defaultdict(list)

for filename, device, backend, platform in SOURCES:
    path = Path(filename)

    with path.open(newline="", encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f))

    if not source_rows:
        raise RuntimeError(f"Empty source: {path}")

    for row in source_rows:
        source_backend = row["backend"].lower()

        expected_backend = (
            "metal" if backend == "llama.cpp/metal" else "cpu"
        )

        if source_backend != expected_backend:
            raise RuntimeError(f"Backend mismatch: {path}")

        quant = (
            row.get("precision")
            or row.get("quantization")
            or ""
        ).lower()

        if quant != "q4_k_m":
            raise RuntimeError(f"Quantization mismatch: {path}")

        if platform == "windows":
            if "qwen3-1.7b" not in row["model"].lower():
                raise RuntimeError(f"Model mismatch: {path}")

            if row["exit_code"] != "0":
                raise RuntimeError(f"Failed experiment row: {path}")

        workload = (
            row.get("profile", "")
            if source_backend == "metal"
            else row.get("workload", "")
        )

        expected_shapes = (
            METAL_SHAPES
            if source_backend == "metal"
            else CPU_SHAPES
        )

        actual_shape = shape(row)

        if workload not in expected_shapes:
            raise RuntimeError(
                f"Unknown workload {workload}: {path}"
            )

        if actual_shape[:3] != expected_shapes[workload]:
            raise RuntimeError(
                f"Shape mismatch for {workload}: {path}"
            )

        if source_backend == "metal" and actual_shape[3] is not None:
            raise RuntimeError(f"Unexpected Metal thread value: {path}")

        if source_backend == "cpu" and actual_shape[3] is None:
            raise RuntimeError(f"Missing CPU thread value: {path}")

        observations[key(device, backend, row)].append({
            "platform": platform,
            "source": filename,
            "rss": row.get("peak_rss_mib", ""),
            "footprint": row.get("peak_footprint_mib", ""),
            "working_set": row.get("peak_working_set_mib", ""),
            "private_commit": row.get(
                "max_sampled_private_commit_mib", ""
            ),
        })


with BASE.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    original_columns = reader.fieldnames
    rows = list(reader)

if not original_columns:
    raise RuntimeError("Registry has no header")

if any(column in original_columns for column in NEW_COLUMNS):
    raise RuntimeError("Registry already contains enrichment columns")

matched_keys = set()

for row in rows:
    for column in NEW_COLUMNS:
        row[column] = ""

    is_qwen3_q4 = (
        row["model"].lower() == "qwen3-1.7b"
        and row["precision"].lower() == "q4_k_m"
        and row["quantization"].lower() == "q4_k_m"
    )

    if is_qwen3_q4:
        row_key = key(row["device"], row["backend"], row)

        if row_key in observations:
            samples = observations[row_key]

            if row_key in matched_keys:
                raise RuntimeError(
                    f"Duplicate registry key: {row_key}"
                )

            matched_keys.add(row_key)

            def values(field):
                return [
                    float(sample[field])
                    for sample in samples
                    if sample[field] != ""
                ]

            rss = values("rss")
            footprint = values("footprint")
            working_set = values("working_set")
            private_commit = values("private_commit")

            if rss:
                row["process_peak_rss_mib_median"] = (
                    statistics.median(rss)
                )
                row["process_peak_rss_mib_max"] = max(rss)

            if footprint:
                row["macos_peak_footprint_mib_max"] = (
                    max(footprint)
                )

            if working_set:
                row["windows_peak_working_set_mib_median"] = (
                    statistics.median(working_set)
                )
                row["windows_peak_working_set_mib_max"] = (
                    max(working_set)
                )

            if private_commit:
                row["windows_sampled_private_commit_mib_max"] = (
                    max(private_commit)
                )

            row["memory_observation_count"] = len(samples)
            row["memory_source_files"] = ";".join(
                sorted({sample["source"] for sample in samples})
            )

    existing_memory = any(
        row.get(column, "").strip()
        for column in (
            "memory_allocated_mb",
            "memory_peak_mb",
            "device_memory_used_mb",
        )
    )

    new_memory = bool(row["memory_observation_count"])

    row["has_any_memory_observation"] = (
        "True" if existing_memory or new_memory else "False"
    )


unmatched = set(observations) - matched_keys

if unmatched:
    raise RuntimeError(
        "Memory source configs missing from registry:\n"
        + "\n".join(map(str, sorted(unmatched)))
    )

if len(matched_keys) != 17:
    raise RuntimeError(
        f"Expected 17 matched configs, got {len(matched_keys)}"
    )

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, lineterminator="\n",
        fieldnames=original_columns + NEW_COLUMNS,
    )
    writer.writeheader()
    writer.writerows(rows)


def is_core(row):
    return row["scheduler_core"].lower() in (
        "true", "1", "yes"
    )


core = [row for row in rows if is_core(row)]

qwen3_core = [
    row for row in core
    if row["model"].lower() == "qwen3-1.7b"
]

def memory_present(row):
    return row["has_any_memory_observation"] == "True"


print(f"Registry rows            : {len(rows)}")
print(f"New memory matches       : {len(matched_keys)}")
print(
    f"Qwen3 core memory        : "
    f"{sum(map(memory_present, qwen3_core))}/{len(qwen3_core)}"
)
print(
    f"All core memory evidence : "
    f"{sum(map(memory_present, core))}/{len(core)}"
)

print("\nQWEN3 CORE MEMORY BY BACKEND")

for backend in ("cpu", "llama.cpp/metal", "llama.cpp/cuda"):
    relevant = [
        row for row in qwen3_core
        if row["backend"] == backend
    ]

    print(
        f"{backend:16s}: "
        f"{sum(map(memory_present, relevant))}/{len(relevant)}"
    )

print(f"\nSaved: {OUT}")
