import csv
import gc
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
from torchvision.models import (
    convnext_base, convnext_large,
    ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
)

outdir = Path(sys.argv[1])
validation_dir = Path(
    "results/benchmark_v3/validation/t05_20261015T173344Z_2438"
)
inventory_path = Path(
    "results/benchmark_v3/inventory/"
    "inv_20261015T175122Z_2438/structural_inventory.csv"
)
image_root = Path(
    "data/imagenetv2/imagenetv2-matched-frequency-format-val"
)

assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
assert torch.backends.mps.is_available()

torch.set_num_threads(4)
torch.manual_seed(2026)

WARMUPS = 3
REPEATS = 12
SESSION = uuid.uuid4().hex

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def qtile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo = int(pos)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)

def sync(device):
    if device == "mps":
        torch.mps.synchronize()

def compare(ref, actual, label):
    assert ref.shape == actual.shape, label
    assert bool(torch.isfinite(actual).all()), label
    assert torch.allclose(
        ref, actual, atol=1e-3, rtol=1e-3
    ), f"Numerical failure: {label}"
    return float((ref-actual).abs().max())

def stage_groups(model):
    return [
        model.features[0],
        model.features[1],
        nn.Sequential(model.features[2], model.features[3]),
        nn.Sequential(model.features[4], model.features[5]),
        nn.Sequential(model.features[6], model.features[7]),
        nn.Sequential(model.avgpool, model.classifier),
    ]

expected_modules = (
    [["features.0"]]
    + [[f"features.1.{i}"] for i in range(3)]
    + [["features.2"]]
    + [[f"features.3.{i}"] for i in range(3)]
    + [["features.4"]]
    + [[f"features.5.{i}"] for i in range(27)]
    + [["features.6"]]
    + [[f"features.7.{i}"] for i in range(3)]
    + [["avgpool", "classifier"]]
)

specs = [
    (
        "convnext_base", convnext_base,
        ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "convnext_base__fine41.json",
    ),
    (
        "convnext_large", convnext_large,
        ConvNeXt_Large_Weights.IMAGENET1K_V1,
        "convnext_large__fine41.json",
    ),
]

validation = json.loads(
    (validation_dir / "metadata.json").read_text()
)

with inventory_path.open(newline="") as f:
    inventory = list(csv.DictReader(f))

image_info = validation["images"][0]
image_path = image_root / image_info["path"]
assert sha(image_path) == image_info["sha256"]

raw_path = outdir / "latency_raw.csv"
summary_path = outdir / "latency_summary.csv"

raw_fields = [
    "session_id", "run_id", "config_id", "model",
    "device", "scope", "stage_id", "iteration",
    "elapsed_ns", "elapsed_ms",
]
summary_fields = [
    "session_id", "run_id", "config_id", "model",
    "device", "scope", "stage_id", "warmups",
    "repetitions", "median_ms", "mean_ms",
    "p10_ms", "p90_ms", "min_ms", "max_ms",
]

source_records = {}
all_numeric_errors = {}
total_configs = 0
total_samples = 0

with raw_path.open("w", newline="") as rf, \
     summary_path.open("w", newline="") as sf:

    raw_writer = csv.DictWriter(rf, fieldnames=raw_fields)
    summary_writer = csv.DictWriter(sf, fieldnames=summary_fields)
    raw_writer.writeheader()
    summary_writer.writeheader()

    def measure(model_name, device, scope, stage_id, fn, source):
        global total_configs, total_samples

        config = {
            "model": model_name,
            "device": device,
            "scope": scope,
            "stage_id": stage_id,
            "weight_sha256": source["weight_sha256"],
            "manifest_sha256": source["manifest_sha256"],
            "image_sha256": image_info["sha256"],
            "cpu_threads": 4,
            "torch": torch.__version__,
            "batch_size": 1,
            "precision": "float32",
        }
        config_id = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        run_id = uuid.uuid4().hex
        values = []

        with torch.inference_mode():
            for _ in range(WARMUPS):
                result = fn()
                sync(device)
                del result

            for iteration in range(1, REPEATS+1):
                sync(device)
                start = time.perf_counter_ns()
                result = fn()
                sync(device)
                elapsed = time.perf_counter_ns() - start
                del result

                values.append(elapsed/1e6)
                raw_writer.writerow({
                    "session_id": SESSION,
                    "run_id": run_id,
                    "config_id": config_id,
                    "model": model_name,
                    "device": device,
                    "scope": scope,
                    "stage_id": stage_id,
                    "iteration": iteration,
                    "elapsed_ns": elapsed,
                    "elapsed_ms": elapsed/1e6,
                })
                total_samples += 1

        median = statistics.median(values)
        summary_writer.writerow({
            "session_id": SESSION,
            "run_id": run_id,
            "config_id": config_id,
            "model": model_name,
            "device": device,
            "scope": scope,
            "stage_id": stage_id,
            "warmups": WARMUPS,
            "repetitions": REPEATS,
            "median_ms": median,
            "mean_ms": statistics.mean(values),
            "p10_ms": qtile(values, 0.10),
            "p90_ms": qtile(values, 0.90),
            "min_ms": min(values),
            "max_ms": max(values),
        })
        rf.flush()
        sf.flush()
        total_configs += 1

        print(
            f"{device.upper():3} {stage_id:4} "
            f"median={median:.3f} ms "
            f"p10={qtile(values,0.10):.3f} "
            f"p90={qtile(values,0.90):.3f}",
            flush=True
        )

    for name, factory, weights, manifest_name in specs:
        print(f"\n=== {name.upper()} ===", flush=True)

        manifest_path = (
            Path("configs/partition_manifests_v2")
            / manifest_name
        )
        manifest_hash = sha(manifest_path)
        manifest = json.loads(manifest_path.read_text())

        assert len(manifest["groups"]) == 41
        assert [
            g["modules"] for g in manifest["groups"]
        ] == expected_modules

        inventory_rows = [
            r for r in inventory if r["model"] == name
        ]
        assert len(inventory_rows) == 6
        assert all(
            r["manifest_sha256"] == manifest_hash
            for r in inventory_rows
        )

        checkpoint = (
            Path(torch.hub.get_dir()) / "checkpoints"
            / Path(urlparse(weights.url).path).name
        )
        expected_sha = validation["models"][name][
            "weight_sha256"
        ]
        assert checkpoint.is_file()
        assert sha(checkpoint) == expected_sha

        source = {
            "weight_enum": weights.name,
            "weight_sha256": expected_sha,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "inventory_sha256": sha(inventory_path),
        }
        source_records[name] = source

        model = factory(weights=None).eval().cpu()
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        del state

        groups = stage_groups(model)

        with Image.open(image_path) as image:
            x_cpu = weights.transforms()(
                image.convert("RGB")
            ).unsqueeze(0)

        assert list(x_cpu.shape) == [1, 3, 224, 224]

        inputs_cpu = []
        outputs_cpu = []

        with torch.inference_mode():
            current = x_cpu
            for group in groups:
                inputs_cpu.append(current.clone())
                current = group(current)
                outputs_cpu.append(current.clone())

            cpu_reference = model(x_cpu)

        compare(
            cpu_reference, outputs_cpu[-1],
            f"{name} CPU grouping"
        )

        print("SOURCE + CPU REFERENCE: PASS", flush=True)
        print("CPU benchmark:", flush=True)

        for i, group in enumerate(groups):
            inp = inputs_cpu[i]
            measure(
                name, "cpu", "isolated_stage", f"S{i}",
                lambda g=group, x=inp: g(x),
                source,
            )

        measure(
            name, "cpu", "full_model", "FULL",
            lambda: model(x_cpu),
            source,
        )

        model.to("mps")
        x_mps = x_cpu.to("mps")
        inputs_mps = [x.to("mps") for x in inputs_cpu]

        errors = {}

        with torch.inference_mode():
            for i, group in enumerate(groups):
                output = group(inputs_mps[i])
                sync("mps")
                errors[f"S{i}"] = compare(
                    outputs_cpu[i],
                    output.cpu(),
                    f"{name} MPS S{i}"
                )
                del output

            output = model(x_mps)
            sync("mps")
            errors["FULL"] = compare(
                cpu_reference,
                output.cpu(),
                f"{name} MPS FULL"
            )
            del output

        all_numeric_errors[name] = errors

        print("MPS NUMERICAL CHECK: PASS", flush=True)
        print("MPS benchmark:", flush=True)

        for i, group in enumerate(groups):
            inp = inputs_mps[i]
            measure(
                name, "mps", "isolated_stage", f"S{i}",
                lambda g=group, x=inp: g(x),
                source,
            )

        measure(
            name, "mps", "full_model", "FULL",
            lambda: model(x_mps),
            source,
        )

        del (
            x_mps, inputs_mps, inputs_cpu, outputs_cpu,
            x_cpu, cpu_reference, groups, model
        )
        gc.collect()
        torch.mps.empty_cache()

assert total_configs == 28
assert total_samples == 336

metadata = {
    "protocol": "V3_convnext_stage_latency_pilot_v1",
    "session_id": SESSION,
    "models": list(source_records),
    "sources": source_records,
    "source_validation": str(validation_dir),
    "source_validation_metadata_sha256": sha(
        validation_dir / "metadata.json"
    ),
    "source_inventory": str(inventory_path),
    "source_inventory_sha256": sha(inventory_path),
    "image": image_info,
    "input_shape": [1, 3, 224, 224],
    "precision": "float32",
    "batch_size": 1,
    "cpu_threads": 4,
    "warmups": WARMUPS,
    "repetitions": REPEATS,
    "independent_process_runs": 1,
    "time_source": "time.perf_counter_ns",
    "mps_sync": "before and after timed call",
    "stage_input": "precomputed CPU reference activation",
    "transfer_timed": False,
    "preprocessing_timed": False,
    "energy_measured": False,
    "numerical_tolerance": {
        "atol": 1e-3, "rtol": 1e-3
    },
    "numerical_max_abs_errors": all_numeric_errors,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "python": sys.version.split()[0],
    "macos": platform.mac_ver()[0],
    "architecture": platform.machine(),
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
        "single image",
        "single process",
        "fixed four CPU threads",
        "isolated stage timings include synchronization",
        "stage medians are not additive",
        "transfer and preprocessing excluded",
        "not a statistical performance comparison",
    ],
}

metadata_path = outdir / "metadata.json"
metadata_path.write_text(
    json.dumps(metadata, indent=2)
)

artifacts = [
    Path(__file__),
    raw_path,
    summary_path,
    metadata_path,
]
checksums = {
    path.name: sha(path) for path in artifacts
}
(outdir / "checksums.json").write_text(
    json.dumps(checksums, indent=2)
)

print("\n==============================")
print("CONVNEXT LATENCY PILOT: PASS")
print("CONFIGURATIONS:", total_configs)
print("TIMED SAMPLES:", total_samples)
print("RESULT DIRECTORY:", outdir)
print("==============================")
