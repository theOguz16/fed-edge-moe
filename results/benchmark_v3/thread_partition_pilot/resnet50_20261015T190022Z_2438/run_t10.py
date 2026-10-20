import csv
import hashlib
import json
import math
import os
import platform
import random
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

OUT = Path(sys.argv[1])

VALIDATION = Path(
    "results/benchmark_v3/validation/"
    "t05_20261015T173344Z_2438/metadata.json"
)
INVENTORY = Path(
    "results/benchmark_v3/inventory/"
    "inv_20261015T175122Z_2438/structural_inventory.csv"
)
MANIFEST = Path(
    "configs/partition_manifests_v2/resnet50__stage6.json"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/imagenetv2-matched-frequency-format-val"
)

THREADS = [1, 2, 4, 6, 8, 10]
WARMUPS = 3
TIMED = 12
SEED = 20261015
SESSION = uuid.uuid4().hex

assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
assert torch.backends.mps.is_available()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (
        values[upper] - values[lower]
    ) * (position - lower)

validation = json.loads(VALIDATION.read_text())
manifest = json.loads(MANIFEST.read_text())

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
assert manifest["group_count"] == 6
assert manifest["cut_boundary_count"] == 5

with INVENTORY.open(newline="") as f:
    inventory = sorted(
        [
            row for row in csv.DictReader(f)
            if row["model"] == "resnet50"
        ],
        key=lambda row: int(row["stage_id"][1:])
    )

assert len(inventory) == 6
assert all(
    row["manifest_sha256"] == sha(MANIFEST)
    for row in inventory
)

image_info = validation["images"][0]
image_path = IMAGE_ROOT / image_info["path"]

assert image_path.is_file()
assert sha(image_path) == image_info["sha256"]

weights = ResNet50_Weights.IMAGENET1K_V2
checkpoint = (
    Path(torch.hub.get_dir()) / "checkpoints"
    / Path(urlparse(weights.url).path).name
)

expected_weight_sha = validation["models"]["resnet50"][
    "weight_sha256"
]

assert checkpoint.is_file()
assert sha(checkpoint) == expected_weight_sha

torch.set_num_threads(4)
torch.manual_seed(2026)

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
        model.avgpool, nn.Flatten(1), model.fc
    ),
]

with Image.open(image_path) as im:
    x_cpu = weights.transforms()(
        im.convert("RGB")
    ).unsqueeze(0)

assert tuple(x_cpu.shape) == (1, 3, 224, 224)

with torch.inference_mode():
    reference = model(x_cpu).detach().clone()

assert tuple(reference.shape) == (1, 1000)
assert bool(torch.isfinite(reference).all())

print("SOURCE + CPU REFERENCE: PASS", flush=True)

LAYOUTS = [
    ("FULL_CPU", None, ["cpu"] * 6),
    ("FULL_MPS", None, ["mps"] * 6),
    ("CPU_TO_MPS", 1, ["cpu"] + ["mps"] * 5),
    ("CPU_TO_MPS", 3, ["cpu"] * 3 + ["mps"] * 3),
    ("MPS_TO_CPU", 3, ["mps"] * 3 + ["cpu"] * 3),
    ("MPS_TO_CPU", 5, ["mps"] * 5 + ["cpu"]),
]

def transfer_events(devices):
    events = []

    if devices[0] == "mps":
        events.append({
            "at": "input",
            "from": "cpu",
            "to": "mps",
            "logical_bytes": (
                x_cpu.numel() * x_cpu.element_size()
            ),
        })

    for i in range(1, 6):
        if devices[i] != devices[i - 1]:
            events.append({
                "at": f"after_S{i-1}",
                "from": devices[i-1],
                "to": devices[i],
                "logical_bytes": int(
                    inventory[i-1]["output_activation_bytes"]
                ),
            })

    if devices[-1] == "mps":
        events.append({
            "at": "output",
            "from": "mps",
            "to": "cpu",
            "logical_bytes": (
                reference.numel() * reference.element_size()
            ),
        })

    return events

def forward(devices):
    value = x_cpu

    for stage, device in zip(stages, devices):
        if value.device.type != device:
            value = value.to(device)
        value = stage(value)

    if value.device.type != "cpu":
        value = value.cpu()

    if "mps" in devices:
        torch.mps.synchronize()

    return value

raw_fields = [
    "session_id", "run_id", "config_id",
    "thread_count", "thread_execution_order",
    "placement_execution_order",
    "mapping", "cut_after",
    "iteration", "elapsed_ns", "elapsed_ms",
]

summary_fields = [
    "session_id", "run_id", "config_id",
    "thread_count", "observed_thread_count",
    "thread_execution_order",
    "placement_execution_order",
    "mapping", "cut_after", "device_sequence",
    "transfer_events", "logical_transfer_bytes",
    "warmups", "timed_repetitions",
    "numerical_pass", "top1_agreement",
    "max_abs_error", "median_ms", "mean_ms",
    "p10_ms", "p90_ms", "min_ms", "max_ms",
]

raw_file = OUT / "latency_raw.csv"
summary_file = OUT / "latency_summary.csv"

thread_order = list(THREADS)
random.Random(SEED).shuffle(thread_order)

order_log = []
summary_rows = []
total_raw = 0

with raw_file.open("w", newline="") as rf, \
     summary_file.open("w", newline="") as sf:

    rw = csv.DictWriter(rf, fieldnames=raw_fields)
    sw = csv.DictWriter(sf, fieldnames=summary_fields)

    rw.writeheader()
    sw.writeheader()

    for thread_index, threads in enumerate(
        thread_order, start=1
    ):
        torch.set_num_threads(threads)
        observed = torch.get_num_threads()
        assert observed == threads

        layouts = list(LAYOUTS)
        random.Random(SEED + threads).shuffle(layouts)

        print(
            f"\n=== THREADS={threads} "
            f"({thread_index}/6) ===",
            flush=True
        )

        order_log.append({
            "threads": threads,
            "thread_execution_order": thread_index,
            "placement_order": [
                {
                    "mapping": mapping,
                    "cut_after": cut,
                }
                for mapping, cut, _ in layouts
            ],
        })

        for placement_index, (
            mapping, cut, devices
        ) in enumerate(layouts, start=1):

            for stage, device in zip(stages, devices):
                stage.to(device)

            torch.mps.synchronize()

            with torch.inference_mode():
                actual = forward(devices)

            assert actual.shape == reference.shape
            assert bool(torch.isfinite(actual).all())

            numerical_pass = bool(torch.allclose(
                reference, actual,
                atol=1e-3, rtol=1e-3
            ))
            top1_agreement = (
                reference.argmax().item()
                == actual.argmax().item()
            )
            max_error = float(
                (reference - actual).abs().max().item()
            )

            assert numerical_pass, (
                f"Numerical failure: threads={threads}, "
                f"{mapping}, cut={cut}"
            )
            assert top1_agreement

            events = transfer_events(devices)

            config = {
                "model": "resnet50",
                "threads": threads,
                "mapping": mapping,
                "cut_after": cut,
                "devices": devices,
                "image_sha256": image_info["sha256"],
                "weight_sha256": expected_weight_sha,
                "manifest_sha256": sha(MANIFEST),
                "torch": torch.__version__,
                "precision": "float32",
                "batch_size": 1,
                "cpu_input_output": True,
            }

            config_id = "cfg_" + digest(config)
            run_id = "run_" + uuid.uuid4().hex

            measurements = []

            with torch.inference_mode():
                for _ in range(WARMUPS):
                    warmup = forward(devices)
                    del warmup

                for iteration in range(1, TIMED + 1):
                    if "mps" in devices:
                        torch.mps.synchronize()

                    start_ns = time.perf_counter_ns()
                    output = forward(devices)
                    elapsed_ns = (
                        time.perf_counter_ns() - start_ns
                    )
                    del output

                    assert elapsed_ns > 0
                    elapsed_ms = elapsed_ns / 1e6
                    assert math.isfinite(elapsed_ms)

                    measurements.append(elapsed_ms)

                    rw.writerow({
                        "session_id": SESSION,
                        "run_id": run_id,
                        "config_id": config_id,
                        "thread_count": threads,
                        "thread_execution_order": thread_index,
                        "placement_execution_order": placement_index,
                        "mapping": mapping,
                        "cut_after": cut,
                        "iteration": iteration,
                        "elapsed_ns": elapsed_ns,
                        "elapsed_ms": elapsed_ms,
                    })
                    total_raw += 1

            row = {
                "session_id": SESSION,
                "run_id": run_id,
                "config_id": config_id,
                "thread_count": threads,
                "observed_thread_count": observed,
                "thread_execution_order": thread_index,
                "placement_execution_order": placement_index,
                "mapping": mapping,
                "cut_after": cut,
                "device_sequence": json.dumps(devices),
                "transfer_events": json.dumps(events),
                "logical_transfer_bytes": sum(
                    e["logical_bytes"] for e in events
                ),
                "warmups": WARMUPS,
                "timed_repetitions": TIMED,
                "numerical_pass": int(numerical_pass),
                "top1_agreement": int(top1_agreement),
                "max_abs_error": max_error,
                "median_ms": statistics.median(measurements),
                "mean_ms": statistics.mean(measurements),
                "p10_ms": quantile(measurements, 0.10),
                "p90_ms": quantile(measurements, 0.90),
                "min_ms": min(measurements),
                "max_ms": max(measurements),
            }

            summary_rows.append(row)
            sw.writerow(row)
            rf.flush()
            sf.flush()

            label = (
                mapping if cut is None
                else f"{mapping} cut={cut}"
            )

            print(
                f"  {label:<20} "
                f"median={row['median_ms']:.3f} ms "
                f"max_error={max_error:.8f} | PASS",
                flush=True
            )

assert len(summary_rows) == 36
assert total_raw == 432
assert len({
    row["config_id"] for row in summary_rows
}) == 36
assert len({
    row["run_id"] for row in summary_rows
}) == 36

metadata = {
    "protocol": "T10_resnet50_threads_partition_pilot_v1",
    "session_id": SESSION,
    "model": "resnet50",
    "threads_tested": THREADS,
    "thread_execution_order": thread_order,
    "execution_orders": order_log,
    "weight_sha256": expected_weight_sha,
    "weight_enum": weights.name,
    "manifest": str(MANIFEST),
    "manifest_sha256": sha(MANIFEST),
    "source_validation": str(VALIDATION),
    "source_validation_sha256": sha(VALIDATION),
    "source_inventory": str(INVENTORY),
    "source_inventory_sha256": sha(INVENTORY),
    "image": image_info,
    "input_shape": [1, 3, 224, 224],
    "batch_size": 1,
    "precision": "float32",
    "warmups_per_configuration": WARMUPS,
    "timed_calls_per_configuration": TIMED,
    "independent_processes": 1,
    "preprocessing_timed": False,
    "input_and_output_device": "cpu",
    "device_transfers_timed": True,
    "weight_placement_timed": False,
    "thread_pool_reconfiguration_timed": False,
    "timing_clock": "time.perf_counter_ns",
    "mps_sync": "before and after inference",
    "execution": "sequential; no pipeline overlap",
    "atol": 1e-3,
    "rtol": 1e-3,
    "logical_transfer_bytes_note": (
        "Tensor payload size, not physical "
        "unified-memory traffic"
    ),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "python": sys.version.split()[0],
    "macos": platform.mac_ver()[0],
    "architecture": platform.machine(),
    "mps_fallback": os.environ[
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ],
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip(),
    "limitations": [
        "One input image",
        "One independent Python process",
        "Thread count changed within one process",
        "Only four selected hybrid cut configurations",
        "One randomized thread and placement order",
        "No independent counterbalanced repetitions",
        "No energy or peak memory measurement",
        "No pipeline overlap",
        "Pilot only; not an inferential comparison",
    ],
}

metadata_file = OUT / "metadata.json"
metadata_file.write_text(
    json.dumps(metadata, indent=2)
)

artifacts = [
    Path(__file__),
    raw_file,
    summary_file,
    metadata_file,
]

(OUT / "checksums.json").write_text(
    json.dumps(
        {p.name: sha(p) for p in artifacts},
        indent=2
    )
)

lookup = {
    (
        int(row["thread_count"]),
        row["mapping"],
        int(row["cut_after"])
        if row["cut_after"] is not None
        else None,
    ): float(row["median_ms"])
    for row in summary_rows
}

print("\n=== THREAD × PARTITION COMPARISON ===")
print(
    f"{'Threads':>7} {'Full CPU':>11} "
    f"{'Full MPS':>11} {'Best hybrid':>13} "
    f"{'Vs MPS':>10}"
)

for threads in THREADS:
    cpu = lookup[(threads, "FULL_CPU", None)]
    mps = lookup[(threads, "FULL_MPS", None)]

    hybrids = [
        row for row in summary_rows
        if int(row["thread_count"]) == threads
        and row["mapping"] not in (
            "FULL_CPU", "FULL_MPS"
        )
    ]
    best = min(
        hybrids, key=lambda row: float(row["median_ms"])
    )
    best_ms = float(best["median_ms"])

    print(
        f"{threads:>7} {cpu:>11.3f} "
        f"{mps:>11.3f} {best_ms:>13.3f} "
        f"{100*(best_ms/mps-1):>+9.2f}%"
    )

print("\n==============================")
print("T10 THREAD × PARTITION PILOT: PASS")
print("CONFIGURATIONS: 36")
print("TIMED SAMPLES: 432")
print("RESULT DIRECTORY:", OUT)
print("==============================")
