import csv
import hashlib
import json
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

outdir = Path(sys.argv[1])
validation_dir = Path(
    "results/benchmark_v3/validation/t05_20261015T173344Z_2438"
)
inventory_file = Path(
    "results/benchmark_v3/inventory/"
    "inv_20261015T175122Z_2438/structural_inventory.csv"
)
manifest_file = Path(
    "configs/partition_manifests_v2/resnet50__stage6.json"
)
image_root = Path(
    "data/imagenetv2/imagenetv2-matched-frequency-format-val"
)

assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
assert torch.backends.mps.is_available()

torch.set_num_threads(4)
torch.manual_seed(2026)

WARMUPS = 3
REPETITIONS = 12
SESSION_ID = uuid.uuid4().hex

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)

validation = json.loads(
    (validation_dir / "metadata.json").read_text()
)

manifest = json.loads(manifest_file.read_text())
expected_groups = [
    ["conv1", "bn1", "relu", "maxpool"],
    ["layer1"],
    ["layer2"],
    ["layer3"],
    ["layer4"],
    ["avgpool", "fc"],
]
assert [g["modules"] for g in manifest["groups"]] == expected_groups

with inventory_file.open(newline="") as f:
    inventory = [
        r for r in csv.DictReader(f)
        if r["model"] == "resnet50"
    ]
inventory.sort(key=lambda r: int(r["stage_id"][1:]))
assert [r["stage_id"] for r in inventory] == [
    f"S{i}" for i in range(6)
]
assert all(
    r["manifest_sha256"] == sha(manifest_file)
    for r in inventory
)

image_info = validation["images"][0]
image_file = image_root / image_info["path"]
assert sha(image_file) == image_info["sha256"]

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

model = resnet50(weights=None).eval().cpu()
state = torch.load(
    checkpoint, map_location="cpu", weights_only=True
)
model.load_state_dict(state, strict=True)
del state

groups = [
    nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool
    ),
    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4,
    nn.Sequential(model.avgpool, nn.Flatten(1), model.fc),
]

with Image.open(image_file) as image:
    x_cpu = weights.transforms()(
        image.convert("RGB")
    ).unsqueeze(0)

assert tuple(x_cpu.shape) == (1, 3, 224, 224)
assert x_cpu.dtype == torch.float32

with torch.inference_mode():
    reference = model(x_cpu).detach().clone()

assert tuple(reference.shape) == (1, 1000)
assert bool(torch.isfinite(reference).all())

print("SOURCE + CPU REFERENCE: PASS", flush=True)

layouts = [
    ("FULL_CPU", None, ["cpu"] * 6),
    ("FULL_MPS", None, ["mps"] * 6),
]

for cut in range(1, 6):
    layouts.append((
        "CPU_TO_MPS", cut,
        ["cpu"] * cut + ["mps"] * (6 - cut)
    ))
    layouts.append((
        "MPS_TO_CPU", cut,
        ["mps"] * cut + ["cpu"] * (6 - cut)
    ))

# Randomize one pilot pass; record the actual order.
random.Random(20261015).shuffle(layouts)

def transfer_description(devices):
    events = []
    input_bytes = x_cpu.numel() * x_cpu.element_size()

    if devices[0] == "mps":
        events.append({
            "at": "input",
            "from": "cpu",
            "to": "mps",
            "logical_bytes": input_bytes,
        })

    for i in range(1, 6):
        if devices[i] != devices[i - 1]:
            events.append({
                "at": f"after_S{i-1}",
                "from": devices[i - 1],
                "to": devices[i],
                "logical_bytes": int(
                    inventory[i - 1]["output_activation_bytes"]
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

def forward():
    value = x_cpu
    for group, device in zip(groups, active_devices):
        value = group(value.to(device))
    value = value.to("cpu")
    if uses_mps:
        torch.mps.synchronize()
    return value

raw_file = outdir / "hybrid_latency_raw.csv"
summary_file = outdir / "hybrid_latency_summary.csv"

raw_fields = [
    "session_id", "run_id", "config_id", "model",
    "mapping", "cut_after", "iteration",
    "elapsed_ns", "elapsed_ms",
]

summary_fields = [
    "session_id", "run_id", "config_id", "model",
    "execution_order", "mapping", "cut_after",
    "device_sequence", "warmups", "repetitions",
    "transfer_events", "logical_transfer_bytes",
    "numerical_pass", "top1_agreement",
    "max_abs_error", "median_ms", "mean_ms",
    "p10_ms", "p90_ms", "min_ms", "max_ms",
]

summaries = []
total_samples = 0

with raw_file.open("w", newline="") as rf, \
     summary_file.open("w", newline="") as sf:

    rw = csv.DictWriter(rf, fieldnames=raw_fields)
    sw = csv.DictWriter(sf, fieldnames=summary_fields)
    rw.writeheader()
    sw.writeheader()

    for order, (mapping, cut, devices) in enumerate(
        layouts, start=1
    ):
        active_devices = devices
        uses_mps = "mps" in devices

        for group, device in zip(groups, devices):
            group.to(device)

        if uses_mps:
            torch.mps.synchronize()

        with torch.inference_mode():
            actual = forward()

        assert actual.shape == reference.shape
        assert bool(torch.isfinite(actual).all())

        numeric_ok = bool(torch.allclose(
            reference, actual, atol=1e-3, rtol=1e-3
        ))
        top1_ok = (
            actual.argmax().item() == reference.argmax().item()
        )
        max_error = float(
            (reference - actual).abs().max().item()
        )

        if not numeric_ok:
            raise RuntimeError(
                f"Numerical failure: {mapping}, cut={cut}"
            )

        events = transfer_description(devices)
        transfer_bytes = sum(e["logical_bytes"] for e in events)

        config = {
            "model": "resnet50",
            "mapping": mapping,
            "cut_after": cut,
            "devices": devices,
            "weight_sha256": expected_weight_sha,
            "manifest_sha256": sha(manifest_file),
            "image_sha256": image_info["sha256"],
            "cpu_threads": 4,
            "precision": "float32",
            "batch_size": 1,
            "input_location": "cpu",
            "output_location": "cpu",
            "transfer_in_timing": True,
            "torch": torch.__version__,
        }
        config_id = "cfg_" + digest(config)
        run_id = "run_" + uuid.uuid4().hex
        values = []

        with torch.inference_mode():
            for _ in range(WARMUPS):
                warmup_result = forward()
                del warmup_result

            for iteration in range(1, REPETITIONS + 1):
                if uses_mps:
                    torch.mps.synchronize()

                start_ns = time.perf_counter_ns()
                result = forward()
                elapsed_ns = (
                    time.perf_counter_ns() - start_ns
                )
                del result

                assert elapsed_ns > 0
                values.append(elapsed_ns / 1e6)

                rw.writerow({
                    "session_id": SESSION_ID,
                    "run_id": run_id,
                    "config_id": config_id,
                    "model": "resnet50",
                    "mapping": mapping,
                    "cut_after": cut,
                    "iteration": iteration,
                    "elapsed_ns": elapsed_ns,
                    "elapsed_ms": elapsed_ns / 1e6,
                })
                total_samples += 1

        record = {
            "session_id": SESSION_ID,
            "run_id": run_id,
            "config_id": config_id,
            "model": "resnet50",
            "execution_order": order,
            "mapping": mapping,
            "cut_after": cut,
            "device_sequence": json.dumps(devices),
            "warmups": WARMUPS,
            "repetitions": REPETITIONS,
            "transfer_events": json.dumps(events),
            "logical_transfer_bytes": transfer_bytes,
            "numerical_pass": int(numeric_ok),
            "top1_agreement": int(top1_ok),
            "max_abs_error": max_error,
            "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values),
            "p10_ms": percentile(values, 0.10),
            "p90_ms": percentile(values, 0.90),
            "min_ms": min(values),
            "max_ms": max(values),
        }

        summaries.append(record)
        sw.writerow(record)
        rf.flush()
        sf.flush()

        print(
            f"{order:02}/12 {mapping:10} cut={str(cut):4} "
            f"median={record['median_ms']:.3f} ms | "
            f"transfers={len(events)} | "
            f"max_error={max_error:.8f} | PASS",
            flush=True
        )

assert len(summaries) == 12
assert total_samples == 144
assert len({r["config_id"] for r in summaries}) == 12

metadata = {
    "protocol": "V3_resnet50_hybrid_e2e_pilot_v1",
    "session_id": SESSION_ID,
    "model": "resnet50",
    "weight_enum": weights.name,
    "weight_sha256": expected_weight_sha,
    "manifest": str(manifest_file),
    "manifest_sha256": sha(manifest_file),
    "source_validation": str(validation_dir),
    "source_validation_metadata_sha256": sha(
        validation_dir / "metadata.json"
    ),
    "source_inventory": str(inventory_file),
    "source_inventory_sha256": sha(inventory_file),
    "input_image": image_info,
    "input_shape": list(x_cpu.shape),
    "preprocessing": str(weights.transforms()),
    "batch_size": 1,
    "precision": "float32",
    "cpu_threads": 4,
    "warmups": WARMUPS,
    "timed_repetitions": REPETITIONS,
    "independent_process_runs": 1,
    "randomization_seed": 20261015,
    "execution_order": [
        {"mapping": m, "cut_after": c}
        for m, c, _ in layouts
    ],
    "input_starts_on": "cpu",
    "result_delivered_on": "cpu",
    "preprocessing_timed": False,
    "input_device_transfer_timed": True,
    "partition_device_transfer_timed": True,
    "output_device_transfer_timed": True,
    "model_placement_timed": False,
    "timing_clock": "time.perf_counter_ns",
    "mps_sync": "before and at end of each timed inference",
    "execution": "sequential stages; no pipeline overlap",
    "logical_transfer_bytes_note": (
        "Derived tensor size, not measured bandwidth or "
        "physical interconnect traffic; Apple unified memory."
    ),
    "numerical_tolerance": {
        "atol": 1e-3,
        "rtol": 1e-3,
    },
    "python": sys.version.split()[0],
    "macos": platform.mac_ver()[0],
    "architecture": platform.machine(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "mps_fallback": os.environ[
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ],
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_dirty": bool(subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()),
    "limitations": [
        "One ImageNetV2 image",
        "One independent process",
        "Fixed CPU threads=4",
        "Only single-cut mappings",
        "Warmup=3 and measured repetitions=12",
        "Mapping order randomized once; no counterbalanced repeats",
        "No energy or RAM/VRAM measurement",
        "No simultaneous CPU/GPU pipeline overlap",
        "Exploratory latency pilot, not inferential benchmark",
    ],
}

metadata_file = outdir / "metadata.json"
metadata_file.write_text(
    json.dumps(metadata, indent=2)
)

artifacts = [
    Path(__file__),
    raw_file,
    summary_file,
    metadata_file,
]

(outdir / "checksums.json").write_text(
    json.dumps(
        {p.name: sha(p) for p in artifacts},
        indent=2
    )
)

lookup = {
    (r["mapping"], r["cut_after"]): r["median_ms"]
    for r in summaries
}

cpu_ms = lookup[("FULL_CPU", None)]
mps_ms = lookup[("FULL_MPS", None)]

hybrids = [
    r for r in summaries
    if r["mapping"] not in ("FULL_CPU", "FULL_MPS")
]
best = min(hybrids, key=lambda r: r["median_ms"])

print("\n=== PILOT COMPARISON ===")
print(f"FULL CPU: {cpu_ms:.3f} ms")
print(f"FULL MPS: {mps_ms:.3f} ms")
print(
    f"FASTEST OBSERVED HYBRID: "
    f"{best['mapping']} cut={best['cut_after']} "
    f"{best['median_ms']:.3f} ms"
)
print(
    "Note: exploratory observation; "
    "not a statistically established winner."
)

print("\n==============================")
print("HYBRID LATENCY PILOT: PASS")
print("CONFIGURATIONS: 12")
print("TIMED SAMPLES: 144")
print("RESULT DIRECTORY:", outdir)
print("==============================")
