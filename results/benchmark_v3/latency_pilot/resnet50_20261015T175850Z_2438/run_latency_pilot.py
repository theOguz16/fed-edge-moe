import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

outdir = Path(sys.argv[1])

validation_dir = Path(
    "results/benchmark_v3/validation/"
    "t05_20261015T173344Z_2438"
)
inventory_dir = Path(
    "results/benchmark_v3/inventory/"
    "inv_20261015T175122Z_2438"
)
manifest_path = Path(
    "configs/partition_manifests_v2/"
    "resnet50__stage6.json"
)

assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
assert torch.backends.mps.is_available()

torch.set_num_threads(4)
torch.manual_seed(2026)

WARMUPS = 3
REPEATS = 12
SESSION_ID = uuid.uuid4().hex

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def quantile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = pos - lo
    return ordered[lo] * (1 - fraction) + ordered[hi] * fraction

def sync(device):
    if device == "mps":
        torch.mps.synchronize()

def check_close(reference, actual, label):
    assert reference.shape == actual.shape, label
    assert bool(torch.isfinite(actual).all()), label
    assert torch.allclose(
        reference, actual, atol=1e-3, rtol=1e-3
    ), f"Numerical mismatch: {label}"
    return float((reference - actual).abs().max().item())

validation = json.loads(
    (validation_dir / "metadata.json").read_text()
)
manifest = json.loads(manifest_path.read_text())

inventory_path = inventory_dir / "structural_inventory.csv"
with inventory_path.open(newline="") as f:
    inventory = [
        row for row in csv.DictReader(f)
        if row["model"] == "resnet50"
    ]

assert len(inventory) == 6
assert len(manifest["groups"]) == 6
assert all(
    row["manifest_sha256"] == sha(manifest_path)
    for row in inventory
)

expected_groups = [
    ["conv1", "bn1", "relu", "maxpool"],
    ["layer1"],
    ["layer2"],
    ["layer3"],
    ["layer4"],
    ["avgpool", "fc"],
]

assert [
    group["modules"] for group in manifest["groups"]
] == expected_groups

image_info = validation["images"][0]
image_path = (
    Path("data/imagenetv2/"
         "imagenetv2-matched-frequency-format-val")
    / image_info["path"]
)
assert image_path.is_file()
assert sha(image_path) == image_info["sha256"]

weights = ResNet50_Weights.IMAGENET1K_V2
checkpoint = (
    Path(torch.hub.get_dir()) / "checkpoints"
    / Path(urlparse(weights.url).path).name
)

expected_weight_sha = validation[
    "models"
]["resnet50"]["weight_sha256"]

assert checkpoint.is_file()
assert sha(checkpoint) == expected_weight_sha

print("SOURCE CHECKS: PASS", flush=True)

model = resnet50(weights=None).eval().cpu()
state = torch.load(
    checkpoint, map_location="cpu", weights_only=True
)
model.load_state_dict(state, strict=True)
del state

stages = [
    nn.Sequential(
        model.conv1, model.bn1,
        model.relu, model.maxpool
    ),
    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4,
    nn.Sequential(
        model.avgpool,
        nn.Flatten(1),
        model.fc
    ),
]

with Image.open(image_path) as im:
    x_cpu = weights.transforms()(
        im.convert("RGB")
    ).unsqueeze(0)

assert tuple(x_cpu.shape) == (1, 3, 224, 224)
assert x_cpu.dtype == torch.float32

stage_inputs_cpu = []
stage_outputs_cpu = []

with torch.inference_mode():
    current = x_cpu

    for stage in stages:
        stage_inputs_cpu.append(current.clone())
        current = stage(current)
        stage_outputs_cpu.append(current.clone())

    reference = model(x_cpu)

check_close(reference, stage_outputs_cpu[-1], "CPU grouping")
print("CPU REFERENCE: PASS", flush=True)

raw_rows = []
summary_rows = []

def measure(device, scope, stage_id, fn):
    config = {
        "model": "resnet50",
        "device": device,
        "scope": scope,
        "stage_id": stage_id,
        "cpu_threads": 4,
        "input_sha256": image_info["sha256"],
        "weight_sha256": expected_weight_sha,
        "manifest_sha256": sha(manifest_path),
        "torch": torch.__version__,
        "batch_size": 1,
        "dtype": "float32",
    }

    config_id = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()

    run_id = uuid.uuid4().hex
    measurements = []

    with torch.inference_mode():
        for _ in range(WARMUPS):
            result = fn()
            sync(device)
            del result

        for iteration in range(REPEATS):
            sync(device)
            start_ns = time.perf_counter_ns()

            result = fn()
            sync(device)

            elapsed_ns = time.perf_counter_ns() - start_ns
            del result

            elapsed_ms = elapsed_ns / 1e6
            measurements.append(elapsed_ms)

            raw_rows.append({
                "session_id": SESSION_ID,
                "run_id": run_id,
                "config_id": config_id,
                "model": "resnet50",
                "device": device,
                "scope": scope,
                "stage_id": stage_id,
                "iteration": iteration + 1,
                "elapsed_ns": elapsed_ns,
                "elapsed_ms": elapsed_ms,
            })

    median = statistics.median(measurements)

    summary_rows.append({
        "session_id": SESSION_ID,
        "run_id": run_id,
        "config_id": config_id,
        "model": "resnet50",
        "device": device,
        "scope": scope,
        "stage_id": stage_id,
        "warmups": WARMUPS,
        "repetitions": REPEATS,
        "median_ms": median,
        "mean_ms": statistics.mean(measurements),
        "p10_ms": quantile(measurements, 0.10),
        "p90_ms": quantile(measurements, 0.90),
        "min_ms": min(measurements),
        "max_ms": max(measurements),
    })

    print(
        f"{device.upper():3} {stage_id:4} "
        f"median={median:.3f} ms | "
        f"p10={quantile(measurements, 0.10):.3f} | "
        f"p90={quantile(measurements, 0.90):.3f}",
        flush=True
    )

print("\n=== CPU BENCHMARK ===", flush=True)

for i, stage in enumerate(stages):
    input_tensor = stage_inputs_cpu[i]
    measure(
        "cpu", "isolated_stage", f"S{i}",
        lambda stage=stage, inp=input_tensor: stage(inp)
    )

measure(
    "cpu", "full_model", "FULL",
    lambda: model(x_cpu)
)

print("\n=== MPS NUMERICAL CHECK ===", flush=True)

model.to("mps")

x_mps = x_cpu.to("mps")
stage_inputs_mps = [
    tensor.to("mps") for tensor in stage_inputs_cpu
]

numerical_errors = {}

with torch.inference_mode():
    for i, stage in enumerate(stages):
        actual = stage(stage_inputs_mps[i])
        sync("mps")
        actual_cpu = actual.cpu()

        numerical_errors[f"S{i}"] = check_close(
            stage_outputs_cpu[i],
            actual_cpu,
            f"MPS S{i}"
        )

    actual_full = model(x_mps)
    sync("mps")

    numerical_errors["FULL"] = check_close(
        reference,
        actual_full.cpu(),
        "MPS FULL"
    )

print("MPS NUMERICAL CHECK: PASS", flush=True)

print("\n=== MPS BENCHMARK ===", flush=True)

for i, stage in enumerate(stages):
    input_tensor = stage_inputs_mps[i]
    measure(
        "mps", "isolated_stage", f"S{i}",
        lambda stage=stage, inp=input_tensor: stage(inp)
    )

measure(
    "mps", "full_model", "FULL",
    lambda: model(x_mps)
)

def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

raw_path = outdir / "latency_raw.csv"
summary_path = outdir / "latency_summary.csv"

write_csv(raw_path, raw_rows)
write_csv(summary_path, summary_rows)

metadata = {
    "protocol": "V3_resnet50_stage_latency_pilot_v1",
    "session_id": SESSION_ID,
    "model": "resnet50",
    "weight_enum": weights.name,
    "weight_sha256": expected_weight_sha,
    "manifest_path": str(manifest_path),
    "manifest_sha256": sha(manifest_path),
    "source_validation": str(validation_dir),
    "source_validation_metadata_sha256": sha(
        validation_dir / "metadata.json"
    ),
    "source_inventory": str(inventory_path),
    "source_inventory_sha256": sha(inventory_path),
    "input_image": image_info,
    "input_shape": list(x_cpu.shape),
    "preprocessing": str(weights.transforms()),
    "batch_size": 1,
    "precision": "float32",
    "cpu_threads": 4,
    "warmups_per_configuration": WARMUPS,
    "timed_repetitions_per_configuration": REPEATS,
    "independent_process_repetitions": 1,
    "timing_clock": "time.perf_counter_ns",
    "mps_synchronization": "before and after each timed call",
    "input_transfer_timed": False,
    "output_transfer_timed": False,
    "preprocessing_timed": False,
    "stage_inputs": "precomputed pretrained CPU activations",
    "timing_scope": {
        "isolated_stage": "one stage call, synchronized",
        "full_model": "whole model call, synchronized",
    },
    "numerical_tolerance": {
        "atol": 1e-3,
        "rtol": 1e-3,
    },
    "mps_numerical_max_abs_errors": numerical_errors,
    "python": sys.version.split()[0],
    "macos": platform.mac_ver()[0],
    "architecture": platform.machine(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip(),
    "git_dirty": bool(subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True
    ).strip()),
    "limitations": [
        "single process; pilot only",
        "one ImageNetV2 input image",
        "fixed CPU thread count of four",
        "isolated stage timings include one synchronization",
        "stage timings must not be summed as exact end-to-end latency",
        "no transfer, preprocessing, memory, or energy measurement",
        "not a statistical performance comparison",
    ],
}

meta_path = outdir / "metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))

artifacts = [
    Path(__file__),
    raw_path,
    summary_path,
    meta_path,
]

checksums = {
    path.name: sha(path) for path in artifacts
}

(outdir / "checksums.json").write_text(
    json.dumps(checksums, indent=2)
)

assert len(summary_rows) == 14
assert len(raw_rows) == 168
assert all(
    row["elapsed_ns"] > 0 for row in raw_rows
)

print("\n==============================")
print("LATENCY PILOT: PASS")
print("CONFIGURATIONS: 14")
print("TIMED SAMPLES: 168")
print("RESULT DIRECTORY:", outdir)
print("==============================")
