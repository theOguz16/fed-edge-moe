import csv
import gc
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
from torchvision.models import (
    convnext_base, convnext_large,
    ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
)

outdir = Path(sys.argv[1])
validation_dir = Path(
    "results/benchmark_v3/validation/t05_20261015T173344Z_2438"
)
inventory_file = Path(
    "results/benchmark_v3/inventory/"
    "inv_20261015T175122Z_2438/structural_inventory.csv"
)
image_root = Path(
    "data/imagenetv2/imagenetv2-matched-frequency-format-val"
)
manifest_dir = Path("configs/partition_manifests_v2")

assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
assert torch.backends.mps.is_available()

torch.set_num_threads(4)
torch.manual_seed(2026)

WARMUPS = 3
REPETITIONS = 12
SESSION = uuid.uuid4().hex

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (
        values[hi] - values[lo]
    ) * (pos - lo)

def synchronize():
    torch.mps.synchronize()

def stages_of(model):
    return [
        model.features[0],
        model.features[1],
        nn.Sequential(model.features[2], model.features[3]),
        nn.Sequential(model.features[4], model.features[5]),
        nn.Sequential(model.features[6], model.features[7]),
        nn.Sequential(model.avgpool, model.classifier),
    ]

expected_manifest_modules = (
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

expected_ranges = [
    "G1-G1", "G2-G4", "G5-G8",
    "G9-G36", "G37-G40", "G41-G41",
]

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

with inventory_file.open(newline="") as f:
    inventory_all = list(csv.DictReader(f))

image_info = validation["images"][0]
image_file = image_root / image_info["path"]
assert image_file.is_file()
assert sha(image_file) == image_info["sha256"]

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

raw_file = outdir / "hybrid_latency_raw.csv"
summary_file = outdir / "hybrid_latency_summary.csv"

all_summaries = []
source_records = {}
execution_orders = {}
numeric_checks = {}
total_samples = 0

with raw_file.open("w", newline="") as rf, \
     summary_file.open("w", newline="") as sf:

    rw = csv.DictWriter(rf, fieldnames=raw_fields)
    sw = csv.DictWriter(sf, fieldnames=summary_fields)
    rw.writeheader()
    sw.writeheader()

    for model_index, (
        name, factory, weights, manifest_name
    ) in enumerate(specs):

        print(f"\n=== {name.upper()} ===", flush=True)

        manifest_file = manifest_dir / manifest_name
        manifest_sha = sha(manifest_file)
        manifest = json.loads(manifest_file.read_text())

        assert manifest["group_count"] == 41
        assert manifest["cut_boundary_count"] == 40
        assert [
            g["modules"] for g in manifest["groups"]
        ] == expected_manifest_modules

        inv = sorted(
            [
                r for r in inventory_all
                if r["model"] == name
            ],
            key=lambda r: int(r["stage_id"][1:])
        )
        assert len(inv) == 6
        assert [
            r["stage_id"] for r in inv
        ] == [f"S{i}" for i in range(6)]
        assert [
            r["source_groups"] for r in inv
        ] == expected_ranges
        assert all(
            r["manifest_sha256"] == manifest_sha
            for r in inv
        )

        checkpoint = (
            Path(torch.hub.get_dir()) / "checkpoints"
            / Path(urlparse(weights.url).path).name
        )
        expected_weight_sha = validation["models"][name][
            "weight_sha256"
        ]
        assert checkpoint.is_file()
        assert sha(checkpoint) == expected_weight_sha

        source_records[name] = {
            "weight_enum": weights.name,
            "weight_sha256": expected_weight_sha,
            "manifest": str(manifest_file),
            "manifest_sha256": manifest_sha,
            "inventory_sha256": sha(inventory_file),
        }

        model = factory(weights=None).eval().cpu()
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        del state

        groups = stages_of(model)

        with Image.open(image_file) as image:
            x_cpu = weights.transforms()(
                image.convert("RGB")
            ).unsqueeze(0)

        assert list(x_cpu.shape) == [1, 3, 224, 224]
        assert x_cpu.dtype == torch.float32

        with torch.inference_mode():
            value = x_cpu
            for i, group in enumerate(groups):
                value = group(value)
                assert list(value.shape) == json.loads(
                    inv[i]["output_shape"]
                )

            reference = model(x_cpu)

        assert tuple(reference.shape) == (1, 1000)
        assert bool(torch.isfinite(reference).all())
        assert torch.allclose(
            value, reference, atol=1e-3, rtol=1e-3
        )

        print(
            "SOURCE + MANIFEST + CPU REFERENCE: PASS",
            flush=True
        )

        layouts = [
            ("FULL_CPU", None, ["cpu"] * 6),
            ("FULL_MPS", None, ["mps"] * 6),
        ]
        for cut in range(1, 6):
            layouts.extend([
                (
                    "CPU_TO_MPS", cut,
                    ["cpu"] * cut + ["mps"] * (6-cut)
                ),
                (
                    "MPS_TO_CPU", cut,
                    ["mps"] * cut + ["cpu"] * (6-cut)
                ),
            ])

        random.Random(
            20261015 + model_index
        ).shuffle(layouts)

        execution_orders[name] = [
            {
                "mapping": mapping,
                "cut_after": cut,
                "devices": devices,
            }
            for mapping, cut, devices in layouts
        ]

        def transfers(devices):
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
                if devices[i] != devices[i-1]:
                    events.append({
                        "at": f"after_S{i-1}",
                        "from": devices[i-1],
                        "to": devices[i],
                        "logical_bytes": int(
                            inv[i-1]["output_activation_bytes"]
                        ),
                    })

            if devices[-1] == "mps":
                events.append({
                    "at": "output",
                    "from": "mps",
                    "to": "cpu",
                    "logical_bytes": (
                        reference.numel()
                        * reference.element_size()
                    ),
                })

            return events

        def forward(devices):
            y = x_cpu
            for group, device in zip(groups, devices):
                if y.device.type != device:
                    y = y.to(device)
                y = group(y)
            if y.device.type != "cpu":
                y = y.cpu()
            if "mps" in devices:
                synchronize()
            return y

        model_summaries = []

        for order, (mapping, cut, devices) in enumerate(
            layouts, start=1
        ):
            has_mps = "mps" in devices

            # Move weights before timing: placement is not timed.
            for group, device in zip(groups, devices):
                group.to(device)

            if has_mps:
                synchronize()

            # Check output for this exact configuration.
            with torch.inference_mode():
                actual = forward(devices)

            assert actual.shape == reference.shape
            assert bool(torch.isfinite(actual).all())

            numerical_pass = bool(torch.allclose(
                reference, actual,
                atol=1e-3, rtol=1e-3
            ))
            top1_pass = (
                actual.argmax().item()
                == reference.argmax().item()
            )
            max_error = float(
                (reference-actual).abs().max().item()
            )

            assert numerical_pass, (
                f"{name}: {mapping}, cut={cut} "
                "failed numerical validation"
            )
            assert top1_pass, (
                f"{name}: {mapping}, cut={cut} "
                "failed top1 agreement"
            )

            events = transfers(devices)
            assert len(events) == (
                0 if mapping == "FULL_CPU" else 2
            )

            config = {
                "model": name,
                "weight_sha256": expected_weight_sha,
                "manifest_sha256": manifest_sha,
                "image_sha256": image_info["sha256"],
                "mapping": mapping,
                "cut_after": cut,
                "devices": devices,
                "cpu_threads": 4,
                "dtype": "float32",
                "batch_size": 1,
                "cpu_input_and_output": True,
                "transfers_timed": True,
                "torch": torch.__version__,
            }

            config_id = "cfg_" + digest(config)
            run_id = "run_" + uuid.uuid4().hex
            samples = []

            with torch.inference_mode():
                for _ in range(WARMUPS):
                    result = forward(devices)
                    del result

                for iteration in range(
                    1, REPETITIONS + 1
                ):
                    if has_mps:
                        synchronize()

                    start_ns = time.perf_counter_ns()
                    result = forward(devices)
                    elapsed_ns = (
                        time.perf_counter_ns() - start_ns
                    )
                    del result

                    assert elapsed_ns > 0
                    samples.append(elapsed_ns / 1e6)

                    rw.writerow({
                        "session_id": SESSION,
                        "run_id": run_id,
                        "config_id": config_id,
                        "model": name,
                        "mapping": mapping,
                        "cut_after": cut,
                        "iteration": iteration,
                        "elapsed_ns": elapsed_ns,
                        "elapsed_ms": elapsed_ns / 1e6,
                    })
                    total_samples += 1

            record = {
                "session_id": SESSION,
                "run_id": run_id,
                "config_id": config_id,
                "model": name,
                "execution_order": order,
                "mapping": mapping,
                "cut_after": cut,
                "device_sequence": json.dumps(devices),
                "warmups": WARMUPS,
                "repetitions": REPETITIONS,
                "transfer_events": json.dumps(events),
                "logical_transfer_bytes": sum(
                    e["logical_bytes"] for e in events
                ),
                "numerical_pass": int(numerical_pass),
                "top1_agreement": int(top1_pass),
                "max_abs_error": max_error,
                "median_ms": statistics.median(samples),
                "mean_ms": statistics.mean(samples),
                "p10_ms": percentile(samples, 0.10),
                "p90_ms": percentile(samples, 0.90),
                "min_ms": min(samples),
                "max_ms": max(samples),
            }

            sw.writerow(record)
            rf.flush()
            sf.flush()

            all_summaries.append(record)
            model_summaries.append(record)

            print(
                f"{order:02}/12 {mapping:10} "
                f"cut={str(cut):4} "
                f"median={record['median_ms']:.3f} ms | "
                f"transfers={len(events)} | "
                f"max_error={max_error:.8f} | PASS",
                flush=True
            )

        assert len(model_summaries) == 12

        numeric_checks[name] = {
            "configurations_checked": 12,
            "all_numerical_pass": True,
            "all_top1_agreement": True,
        }

        baselines = {
            r["mapping"]: r["median_ms"]
            for r in model_summaries
            if r["mapping"] in ("FULL_CPU", "FULL_MPS")
        }
        hybrids = [
            r for r in model_summaries
            if r["mapping"] not in (
                "FULL_CPU", "FULL_MPS"
            )
        ]
        best = min(
            hybrids, key=lambda r: r["median_ms"]
        )

        print(f"\n{name} PILOT COMPARISON:")
        print(
            f"  FULL CPU: "
            f"{baselines['FULL_CPU']:.3f} ms"
        )
        print(
            f"  FULL MPS: "
            f"{baselines['FULL_MPS']:.3f} ms"
        )
        print(
            f"  FASTEST HYBRID: "
            f"{best['mapping']} cut={best['cut_after']} "
            f"{best['median_ms']:.3f} ms"
        )
        print(
            f"  HYBRID VS MPS: "
            f"{100 * (best['median_ms'] / baselines['FULL_MPS'] - 1):+.2f}%"
        )
        print(
            "  Exploratory only; not a statistical winner."
        )

        del (
            model, groups, x_cpu, reference,
            value, actual, group
        )
        gc.collect()
        torch.mps.empty_cache()

assert len(all_summaries) == 24
assert total_samples == 288
assert len({
    r["config_id"] for r in all_summaries
}) == 24
assert len({
    r["run_id"] for r in all_summaries
}) == 24

metadata = {
    "protocol": "V3_convnext_hybrid_e2e_pilot_v1",
    "session_id": SESSION,
    "models": list(source_records),
    "sources": source_records,
    "source_validation": str(validation_dir),
    "source_validation_metadata_sha256": sha(
        validation_dir / "metadata.json"
    ),
    "source_inventory": str(inventory_file),
    "source_inventory_sha256": sha(inventory_file),
    "input_image": image_info,
    "input_shape": [1, 3, 224, 224],
    "precision": "float32",
    "batch_size": 1,
    "cpu_threads": 4,
    "warmups": WARMUPS,
    "timed_repetitions": REPETITIONS,
    "independent_process_runs": 1,
    "randomization_seeds": {
        "convnext_base": 20261015,
        "convnext_large": 20261016,
    },
    "execution_orders": execution_orders,
    "numerical_checks": numeric_checks,
    "input_starts_on": "cpu",
    "output_delivered_on": "cpu",
    "preprocessing_timed": False,
    "input_device_transfer_timed": True,
    "boundary_transfer_timed": True,
    "output_device_transfer_timed": True,
    "model_weight_placement_timed": False,
    "timing_clock": "time.perf_counter_ns",
    "mps_synchronization": (
        "before and at end of timed inference"
    ),
    "execution": (
        "sequential six-stage execution; "
        "no pipeline overlap"
    ),
    "logical_transfer_bytes_note": (
        "Tensor payload sizes only; not measured "
        "physical memory traffic on unified memory."
    ),
    "numerical_tolerance": {
        "atol": 1e-3,
        "rtol": 1e-3,
    },
    "energy_measured": False,
    "memory_peak_measured": False,
    "python": sys.version.split()[0],
    "macos": platform.mac_ver()[0],
    "architecture": platform.machine(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "mps_fallback": os.environ[
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ],
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip(),
    "git_dirty": bool(subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True
    ).strip()),
    "limitations": [
        "One ImageNetV2 image per model",
        "One independent Python process",
        "Fixed CPU thread count=4",
        "Only single-cut mappings",
        "Three warmups and twelve timed calls",
        "One randomized ordering per model",
        "No counterbalanced independent repeats",
        "No energy or peak RAM measurement",
        "No CPU/GPU pipeline overlap",
        "Exploratory pilot, not inferential benchmark",
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

print("\n==============================")
print("CONVNEXT HYBRID PILOT: PASS")
print("CONFIGURATIONS: 24")
print("TIMED SAMPLES: 288")
print("RESULT DIRECTORY:", outdir)
print("==============================")
