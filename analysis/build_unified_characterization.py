from pathlib import Path
import csv
import statistics

RESULTS = Path("results")

RAW_OUT = RESULTS / "unified_characterization_raw_normalized.csv"
OUT = RESULTS / "unified_characterization.csv"

# ---------------------------------------------------------------------
# Canonical source selection
# ---------------------------------------------------------------------

CANONICAL_FILES = [
    # TEXT EASY
    "distilgpt2_easy_mps_sweep.csv",
    "distilgpt2_easy_cuda_sweep.csv",
    "distilgpt2_easy_mps_power_final.csv",
    "distilgpt2_easy_cuda_nvml_final.csv",
    "mac_text_latency.csv",
    "msi_cpu_repeated_power.csv",

    # TEXT MEDIUM
    "qwen25_medium_mps_sweep.csv",
    "qwen25_medium_cuda_sweep.csv",
    "qwen25_medium_mac_cpu_scaling.csv",
    "qwen25_medium_msi_cpu_scaling.csv",
    "qwen25_medium_mps_power_summary.csv",
    "qwen25_medium_cuda_nvml_power.csv",
    "qwen25_medium_mac_cpu_power_summary.csv",
    "qwen25_medium_msi_cpu_power.csv",

    # TEXT HARD
    "qwen3_hard_q4_cross_device.csv",
    "qwen3_hard_q4_mac_cpu_scaling.csv",
    "qwen3_hard_q4_msi_cpu_scaling.csv",
    "qwen3_hard_q4_mac_accel_power_synced.csv",
    "qwen3_hard_q4_cuda_nvml_power.csv",
    "qwen3_hard_q4_mac_cpu_power.csv",
    "qwen3_hard_q4_msi_cpu_power.csv",

    # VISION EASY
    "resnet50_vision_easy_mps_sweep.csv",
    "resnet50_vision_easy_msi_cuda_sweep.csv",
    "resnet50_vision_easy_mac_cpu_scaling_final.csv",
    "resnet50_vision_easy_msi_cpu_scaling.csv",
    "resnet50_vision_easy_mac_cpu_power.csv",
    "resnet50_vision_easy_msi_cpu_power_highres16_2s.csv",
    "resnet50_vision_easy_mps_power_500ms.csv",
    "resnet50_vision_easy_msi_cuda_power.csv",

    # VISION MEDIUM
    "convnext_base_vision_medium_mac_mps_sweep.csv",
    "convnext_base_vision_medium_msi_cuda_sweep.csv",
    "convnext_base_vision_medium_mac_cpu_batch_sweep.csv",
    "convnext_base_vision_medium_msi_cpu_batch_sweep.csv",
    "convnext_base_vision_medium_mac_cpu_scaling.csv",
    "convnext_base_vision_medium_msi_cpu_scaling.csv",
    "convnext_base_vision_medium_mac_cpu_power.csv",
    "convnext_base_vision_medium_msi_cpu_power.csv",
    "convnext_base_vision_medium_mac_mps_power.csv",
    "convnext_base_vision_medium_msi_cuda_power.csv",

    # VISION HARD
    "convnext_large_vision_hard_mac_mps_sweep.csv",
    "convnext_large_vision_hard_msi_cuda_sweep.csv",
    "convnext_large_vision_hard_mac_cpu_batch_sweep.csv",
    "convnext_large_vision_hard_msi_cpu_batch_sweep.csv",
    "convnext_large_vision_hard_mac_cpu_scaling.csv",
    "convnext_large_vision_hard_msi_cpu_scaling.csv",
    "convnext_large_vision_hard_mac_cpu_power.csv",
    "convnext_large_vision_hard_msi_cpu_power.csv",
    "convnext_large_vision_hard_mac_mps_power_canonical.csv",
    "convnext_large_vision_hard_msi_cuda_power.csv",
]


MODEL_META = {
    "distilgpt2": {
        "modality": "text",
        "difficulty": "easy",
        "model": "DistilGPT2",
        "model_params_m": 82.0,
    },
    "qwen25": {
        "modality": "text",
        "difficulty": "medium",
        "model": "Qwen2.5-1.5B-Instruct",
        "model_params_m": 1500.0,
    },
    "qwen3": {
        "modality": "text",
        "difficulty": "hard",
        "model": "Qwen3-1.7B",
        "model_params_m": 1700.0,
    },
    "resnet50": {
        "modality": "vision",
        "difficulty": "easy",
        "model": "ResNet50",
        "model_params_m": 25.56,
    },
    "convnext_base": {
        "modality": "vision",
        "difficulty": "medium",
        "model": "ConvNeXt-Base",
        "model_params_m": 88.6,
    },
    "convnext_large": {
        "modality": "vision",
        "difficulty": "hard",
        "model": "ConvNeXt-Large",
        "model_params_m": 197.8,
    },
}


# ---------------------------------------------------------------------
# Qwen3 Q4 short sweep did not have its own canonical CSV.
# Create one from the already validated matrix.
# ---------------------------------------------------------------------

def ensure_qwen3_short_csv():
    path = RESULTS / "qwen3_hard_q4_cross_device.csv"

    if path.exists():
        return

    mac = {
        (128,64,1):76.19,  (128,64,2):112.18, (128,64,4):98.21,
        (128,128,1):72.66, (128,128,2):119.03,(128,128,4):102.99,
        (512,64,1):75.83,  (512,64,2):113.27, (512,64,4):97.86,
        (512,128,1):75.81, (512,128,2):112.75,(512,128,4):97.55,
        (1024,64,1):72.72, (1024,64,2):105.34,(1024,64,4):91.62,
        (1024,128,1):72.38,(1024,128,2):104.92,(1024,128,4):91.46,
    }

    msi = {
        (128,64,1):117.88,  (128,64,2):223.40, (128,64,4):300.64,
        (128,128,1):119.70, (128,128,2):222.87,(128,128,4):299.03,
        (512,64,1):114.82,  (512,64,2):208.36, (512,64,4):274.67,
        (512,128,1):113.83, (512,128,2):207.74,(512,128,4):273.17,
        (1024,64,1):109.15, (1024,64,2):194.63,(1024,64,4):252.51,
        (1024,128,1):109.80,(1024,128,2):194.70,(1024,128,4):250.55,
    }

    rows = []

    for (pp, tg, batch), throughput in mac.items():
        rows.append({
            "device": "Apple M4",
            "backend": "llama.cpp/metal",
            "precision": "q4_k_m",
            "pp": pp,
            "tg": tg,
            "batch": batch,
            "throughput_tok_sec": throughput,
            "status": "PASS",
        })

    for (pp, tg, batch), throughput in msi.items():
        rows.append({
            "device": "RTX 3050 Laptop",
            "backend": "llama.cpp/cuda",
            "precision": "q4_k_m",
            "pp": pp,
            "tg": tg,
            "batch": batch,
            "throughput_tok_sec": throughput,
            "status": "PASS",
        })

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("Created missing canonical file:", path)


def first(row, *keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def number(value):
    if value in ("", None):
        return ""

    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return ""


def normalized_precision(value):
    if value in ("", None):
        return "unspecified"

    x = str(value).lower().strip()
    x = x.replace("torch.", "")

    if x in ("float32", "fp32"):
        return "fp32"

    if x in ("float16", "half", "fp16"):
        return "fp16"

    if "q4" in x:
        return "q4_k_m"

    return x


def model_meta(filename):
    if filename in ("mac_text_latency.csv", "msi_cpu_repeated_power.csv"):
        return MODEL_META["distilgpt2"]

    for prefix, meta in MODEL_META.items():
        if filename.startswith(prefix):
            return meta

    raise ValueError(f"Unknown model source: {filename}")


def infer_device_backend(filename, row):
    # Explicit values in dedicated Qwen3 canonical file
    if filename == "qwen3_hard_q4_cross_device.csv":
        return row["device"], row["backend"]

    if filename == "mac_text_latency.csv":
        accel = str(first(row, "accelerator")).lower()

        if "mps" in accel:
            return "Apple M4", "mps"

        return "Apple M4", "cpu"

    if filename == "msi_cpu_repeated_power.csv":
        return "Intel i7-11800H", "cpu"

    if "mac_cpu" in filename:
        return "Apple M4", "cpu"

    if "mac_mps" in filename or "mps_" in filename:
        return "Apple M4", "mps"

    if "mac_accel" in filename:
        return "Apple M4", "llama.cpp/metal"

    if "msi_cpu" in filename:
        return "Intel i7-11800H", "cpu"

    if (
        "msi_cuda" in filename
        or "cuda_" in filename
        or "cuda_nvml" in filename
    ):
        if filename.startswith("qwen3"):
            return "RTX 3050 Laptop", "llama.cpp/cuda"

        return "RTX 3050 Laptop", "cuda"

    backend = str(first(row, "backend")).lower()
    device = first(row, "device")

    if backend == "mps":
        return "Apple M4", "mps"

    if backend == "cuda":
        return "RTX 3050 Laptop", "cuda"

    return str(device or "unknown"), backend or "unknown"


def infer_precision(filename, row, backend):
    value = first(row, "precision", "dtype")

    if value:
        return normalized_precision(value)

    if filename.startswith("qwen3_hard_q4"):
        return "q4_k_m"

    # Vision CPU policy was controlled FP32.
    if (
        "vision_" in filename
        and backend == "cpu"
    ):
        return "fp32"

    # Do not silently invent precision for legacy text files.
    return "unspecified"


def infer_quantization(filename, precision):
    if filename.startswith("qwen3_hard_q4") or precision == "q4_k_m":
        return "Q4_K_M"

    return "none"


def infer_measurement_mode(filename):
    if filename == "qwen3_hard_q4_cross_device.csv":
        return "short_sweep"

    if filename == "mac_text_latency.csv":
        return "short_latency"

    if "cpu_batch_sweep" in filename:
        return "cpu_batch_short"

    if "cpu_scaling" in filename:
        return "cpu_scaling_short"

    if "_sweep" in filename:
        return "short_sweep"

    if (
        "power" in filename
        or "nvml" in filename
        or "repeated" in filename
    ):
        return "sustained_power"

    return "other"


def energy_boundary(device, backend, mode):
    if mode != "sustained_power":
        return ""

    if device == "Apple M4":
        return "combined_soc"

    if device == "Intel i7-11800H":
        return "cpu_package"

    if device == "RTX 3050 Laptop":
        return "gpu_board"

    return ""


def normalize(filename, row):
    meta = model_meta(filename)

    modality = meta["modality"]
    device, backend = infer_device_backend(filename, row)
    precision = infer_precision(filename, row, backend)

    context = number(first(
        row,
        "context_tokens",
        "context",
        "pp",
    ))

    output = number(first(
        row,
        "output_tokens",
        "output",
        "tg",
    ))

    resolution = number(first(
        row,
        "resolution",
    ))

    batch = number(first(row, "batch"))
    threads = number(first(row, "threads", "cpu_threads"))
    repeat = number(first(row, "repeat", "session"))

    workload = first(row, "workload", "scenario")
    profile = first(row, "profile")

    if not workload:
        if filename == "qwen3_hard_q4_cross_device.csv":
            workload = "q4_grid"
        else:
            workload = "grid"

    mode = infer_measurement_mode(filename)

    # Prefer no-monitor throughput for paired A/B power experiments.
    throughput_basis = "measured"

    if first(row, "no_monitor_img_s") not in ("", None):
        throughput = number(row["no_monitor_img_s"])
        throughput_basis = "no_monitor"
    else:
        throughput = number(first(
            row,
            "throughput_img_s",
            "throughput_images_sec",
            "throughput_tok_sec",
            "tokens_per_sec",
            "s_tg_tok_s",
            "median_generation_tok_sec",
        ))

        if mode == "sustained_power":
            throughput_basis = "power_monitored_or_summary"
        elif mode.startswith("short") or mode.endswith("_short"):
            throughput_basis = "short_measurement"

    throughput_unit = "tok/s" if modality == "text" else "img/s"

    latency_sec = number(first(
        row,
        "latency_sec",
        "t_total_sec",
    ))

    power_w = number(first(
        row,
        "mean_power_w",
        "avg_power_w",
        "power_w",
        "workload_mean_w",
        "avg_cpu_package_power_w",
        "avg_gpu_power_w",
        "median_power_w",
    ))

    dynamic_power_w = number(first(
        row,
        "dynamic_power_w",
    ))

    energy_run = number(first(
        row,
        "energy_run_j",
        "energy_per_run_j",
        "energy_batch_j",
    ))

    energy_item = number(first(
        row,
        "energy_img_j",
        "energy_image_j",
        "energy_token_j",
        "j_per_img",
    ))

    dynamic_energy_item = number(first(
        row,
        "dynamic_energy_token_j",
    ))

    dynamic_energy_run = number(first(
        row,
        "dynamic_energy_run_j",
        "dynamic_energy_per_run_j",
    ))

    # Derive per-item energy only when the source provides enough
    # information and no direct per-item metric exists.
    if energy_item == "" and energy_run != "":
        if modality == "vision" and batch not in ("", 0):
            energy_item = energy_run / batch

        elif (
            modality == "text"
            and output not in ("", 0)
            and batch not in ("", 0)
        ):
            energy_item = energy_run / (output * batch)

    if (
        dynamic_energy_item == ""
        and dynamic_energy_run != ""
        and modality == "text"
        and output not in ("", 0)
        and batch not in ("", 0)
    ):
        dynamic_energy_item = dynamic_energy_run / (output * batch)

    memory_allocated = number(first(
        row,
        "mps_allocated_mb",
        "cuda_allocated_mb",
        "device_memory_mb",
        "allocated_mb",
    ))

    memory_peak = number(first(
        row,
        "torch_peak_alloc_mb",
        "cuda_peak_mb",
        "vram_peak_mb",
        "peak_vram_mb",
        "torch_peak_vram_mb",
    ))

    device_memory_used = number(first(
        row,
        "nvml_peak_used_mb",
        "nvml_vram_mb",
        "vram_mb",
    ))

    gpu_util = number(first(
        row,
        "mean_gpu_util_pct",
        "avg_gpu_util_percent",
        "gpu_util_percent",
    ))

    temp_c = number(first(
        row,
        "peak_temp_c",
        "gpu_temp_c",
    ))

    monitor_delta = number(first(
        row,
        "monitor_delta_pct",
    ))

    duration_sec = number(first(
        row,
        "duration_sec",
        "duration",
        "wall_sec",
    ))

    status = str(first(row, "status") or "PASS").strip().upper()

    if status in ("OK", "SUCCESS"):
        status = "PASS"

    return {
        "modality": modality,
        "difficulty": meta["difficulty"],
        "model": meta["model"],
        "model_params_m": meta["model_params_m"],
        "device": device,
        "backend": backend,
        "precision": precision,
        "quantization": infer_quantization(filename, precision),

        "workload": workload,
        "profile": profile,

        "resolution": resolution,
        "context_tokens": context,
        "output_tokens": output,
        "batch": batch,
        "threads": threads,

        "repeat": repeat,
        "measurement_mode": mode,
        "is_sustained": 1 if mode == "sustained_power" else 0,
        "throughput_basis": throughput_basis,

        "latency_sec": latency_sec,
        "throughput": throughput,
        "throughput_unit": throughput_unit,

        "power_w": power_w,
        "dynamic_power_w": dynamic_power_w,
        "energy_per_run_j": energy_run,
        "energy_per_item_j": energy_item,
        "dynamic_energy_per_item_j": dynamic_energy_item,

        "memory_allocated_mb": memory_allocated,
        "memory_peak_mb": memory_peak,
        "device_memory_used_mb": device_memory_used,

        "gpu_util_pct": gpu_util,
        "temp_c": temp_c,
        "monitor_delta_pct": monitor_delta,
        "duration_sec": duration_sec,

        "energy_boundary": energy_boundary(device, backend, mode),
        "quality_measured": 0,
        "status": status,
        "source_file": filename,
    }


def median_value(values):
    nums = [
        float(v)
        for v in values
        if v not in ("", None)
    ]

    if not nums:
        return ""

    return statistics.median(nums)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

ensure_qwen3_short_csv()

normalized = []
missing = []

for filename in CANONICAL_FILES:
    path = RESULTS / filename

    if not path.exists():
        missing.append(filename)
        continue

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # mac_text_latency can contain multiple models.
            # Keep only DistilGPT2 rows for the Easy dataset.
            if filename == "mac_text_latency.csv":
                model = str(row.get("model", "")).lower()

                if model and "distil" not in model:
                    continue

            normalized.append(
                normalize(filename, row)
            )


if not normalized:
    raise SystemExit("No canonical rows found.")


raw_fields = list(normalized[0].keys())

with RAW_OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=raw_fields)
    w.writeheader()
    w.writerows(normalized)


# ---------------------------------------------------------------------
# Aggregate repeats to one canonical row per configuration/source.
# ---------------------------------------------------------------------

GROUP_KEYS = [
    "modality",
    "difficulty",
    "model",
    "model_params_m",
    "device",
    "backend",
    "precision",
    "quantization",
    "workload",
    "profile",
    "resolution",
    "context_tokens",
    "output_tokens",
    "batch",
    "threads",
    "measurement_mode",
    "is_sustained",
    "throughput_basis",
    "throughput_unit",
    "energy_boundary",
    "quality_measured",
    "status",
    "source_file",
]

NUMERIC_FIELDS = [
    "latency_sec",
    "throughput",
    "power_w",
    "dynamic_power_w",
    "energy_per_run_j",
    "energy_per_item_j",
    "dynamic_energy_per_item_j",
    "memory_allocated_mb",
    "memory_peak_mb",
    "device_memory_used_mb",
    "gpu_util_pct",
    "temp_c",
    "monitor_delta_pct",
    "duration_sec",
]

groups = {}

for row in normalized:
    key = tuple(row[k] for k in GROUP_KEYS)
    groups.setdefault(key, []).append(row)


canonical = []

for key, rows in groups.items():
    out = dict(zip(GROUP_KEYS, key))

    for field in NUMERIC_FIELDS:
        out[field] = median_value(
            [r[field] for r in rows]
        )

    out["repeat_count"] = len(rows)

    canonical.append(out)


OUT_FIELDS = [
    "modality",
    "difficulty",
    "model",
    "model_params_m",

    "device",
    "backend",
    "precision",
    "quantization",

    "workload",
    "profile",

    "resolution",
    "context_tokens",
    "output_tokens",
    "batch",
    "threads",

    "measurement_mode",
    "is_sustained",
    "throughput_basis",

    "latency_sec",
    "throughput",
    "throughput_unit",

    "power_w",
    "dynamic_power_w",
    "energy_per_run_j",
    "energy_per_item_j",
    "dynamic_energy_per_item_j",

    "memory_allocated_mb",
    "memory_peak_mb",
    "device_memory_used_mb",

    "gpu_util_pct",
    "temp_c",
    "monitor_delta_pct",
    "duration_sec",

    "energy_boundary",
    "quality_measured",
    "status",
    "repeat_count",
    "source_file",
]


canonical.sort(
    key=lambda r: (
        r["modality"],
        r["difficulty"],
        r["model"],
        r["device"],
        r["measurement_mode"],
        str(r["precision"]),
        str(r["resolution"]),
        str(r["context_tokens"]),
        str(r["batch"]),
        str(r["threads"]),
    )
)


with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
    w.writeheader()

    for row in canonical:
        w.writerow({
            k: row.get(k, "")
            for k in OUT_FIELDS
        })


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("UNIFIED CHARACTERIZATION DATASET")
print("=" * 72)

print("Raw normalized rows :", len(normalized))
print("Canonical rows      :", len(canonical))
print("Output              :", OUT)

print("\nROWS BY WORKLOAD")

for modality in ("text", "vision"):
    for difficulty in ("easy", "medium", "hard"):
        n = sum(
            1 for r in canonical
            if r["modality"] == modality
            and r["difficulty"] == difficulty
        )

        print(
            f"{modality:6} {difficulty:6} : {n}"
        )

print("\nROWS BY DEVICE")

for device in sorted(set(r["device"] for r in canonical)):
    n = sum(
        1 for r in canonical
        if r["device"] == device
    )

    print(f"{device:24} : {n}")

print("\nMEASUREMENT MODES")

for mode in sorted(set(r["measurement_mode"] for r in canonical)):
    n = sum(
        1 for r in canonical
        if r["measurement_mode"] == mode
    )

    print(f"{mode:24} : {n}")

if missing:
    print("\nWARNING — missing canonical files:")
    for filename in missing:
        print("  -", filename)

print()
print("Saved:", RAW_OUT)
print("Saved:", OUT)
