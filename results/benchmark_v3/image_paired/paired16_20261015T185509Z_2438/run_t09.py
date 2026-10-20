import csv
import gc
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
from collections import defaultdict
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

ROOT = Path(sys.argv[1]).resolve()
VALIDATION = Path(
    "results/benchmark_v3/validation/"
    "t05_20261015T173344Z_2438/metadata.json"
)
INVENTORY = Path(
    "results/benchmark_v3/inventory/"
    "inv_20261015T175122Z_2438/structural_inventory.csv"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/imagenetv2-matched-frequency-format-val"
)
MANIFEST_ROOT = Path("configs/partition_manifests_v2")

WARMUPS = 3
TIMED = 10
PROCESSES = 2
CPU_THREADS = 4

RAW_FIELDS = [
    "process_repeat", "process_id", "run_id", "config_id",
    "model", "image_index", "image_path", "image_sha256",
    "mapping", "cut_after", "image_order", "execution_order",
    "iteration", "elapsed_ns", "elapsed_ms",
]

SUMMARY_FIELDS = [
    "process_repeat", "process_id", "run_id", "config_id",
    "model", "image_index", "image_path", "image_sha256",
    "mapping", "cut_after", "device_sequence",
    "image_order", "execution_order",
    "warmups", "repetitions",
    "transfer_events", "logical_transfer_bytes",
    "numerical_pass", "top1_agreement", "max_abs_error",
    "median_ms", "mean_ms", "min_ms", "max_ms",
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()

def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def write_checksums(directory, paths):
    (directory / "checksums.json").write_text(
        json.dumps(
            {p.name: sha(p) for p in paths},
            indent=2
        )
    )

def verify_checksums(directory):
    checksums = json.loads(
        (directory / "checksums.json").read_text()
    )
    for name, expected in checksums.items():
        assert sha(directory / name) == expected, (
            f"Checksum failed: {directory / name}"
        )

def stages_of(model):
    return [
        model.features[0],
        model.features[1],
        nn.Sequential(model.features[2], model.features[3]),
        nn.Sequential(model.features[4], model.features[5]),
        nn.Sequential(model.features[6], model.features[7]),
        nn.Sequential(model.avgpool, model.classifier),
    ]

EXPECTED_PATHS = (
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

EXPECTED_RANGES = [
    "G1-G1", "G2-G4", "G5-G8",
    "G9-G36", "G37-G40", "G41-G41",
]

SPECS = [
    (
        "convnext_base",
        convnext_base,
        ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "convnext_base__fine41.json",
        "CPU_TO_MPS",
        1,
    ),
    (
        "convnext_large",
        convnext_large,
        ConvNeXt_Large_Weights.IMAGENET1K_V1,
        "convnext_large__fine41.json",
        "MPS_TO_CPU",
        5,
    ),
]

def make_events(devices, inv, x, reference):
    events = []

    if devices[0] == "mps":
        events.append({
            "at": "input",
            "from": "cpu",
            "to": "mps",
            "logical_bytes": x.numel() * x.element_size(),
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
                reference.numel() * reference.element_size()
            ),
        })

    assert len(events) == 2
    return events

def forward(stages, x_cpu, devices):
    value = x_cpu

    for stage, device in zip(stages, devices):
        if value.device.type != device:
            value = value.to(device)
        value = stage(value)

    if value.device.type != "cpu":
        value = value.cpu()

    torch.mps.synchronize()
    return value

def run_worker(process_repeat):
    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()

    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(2026)

    process_dir = ROOT / f"process_{process_repeat:02}"
    process_dir.mkdir(parents=True, exist_ok=False)
    process_id = uuid.uuid4().hex

    validation = json.loads(VALIDATION.read_text())
    images = validation["images"]
    assert len(images) == 16

    for entry in images:
        path = IMAGE_ROOT / entry["path"]
        assert path.is_file()
        assert sha(path) == entry["sha256"]

    inventory_all = read_csv(INVENTORY)

    specs = list(SPECS)
    if process_repeat == 2:
        specs.reverse()

    raw_rows = []
    summary_rows = []
    source_records = {}
    image_orders = {}

    for name, factory, weights, manifest_name, hybrid_name, cut in specs:
        print(
            f"\nPROCESS {process_repeat}/{PROCESSES}: "
            f"{name}",
            flush=True
        )

        manifest_path = MANIFEST_ROOT / manifest_name
        manifest_sha = sha(manifest_path)
        manifest = json.loads(manifest_path.read_text())

        assert manifest["group_count"] == 41
        assert [
            group["modules"] for group in manifest["groups"]
        ] == EXPECTED_PATHS

        inv = sorted(
            [
                row for row in inventory_all
                if row["model"] == name
            ],
            key=lambda row: int(row["stage_id"][1:])
        )

        assert len(inv) == 6
        assert [
            row["source_groups"] for row in inv
        ] == EXPECTED_RANGES
        assert all(
            row["manifest_sha256"] == manifest_sha
            for row in inv
        )

        checkpoint = (
            Path(torch.hub.get_dir()) / "checkpoints"
            / Path(urlparse(weights.url).path).name
        )

        expected_weight_sha = validation[
            "models"
        ][name]["weight_sha256"]

        assert checkpoint.is_file()
        assert sha(checkpoint) == expected_weight_sha

        source_records[name] = {
            "weight_enum": weights.name,
            "weight_sha256": expected_weight_sha,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
        }

        model = factory(weights=None).eval().cpu()
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        del state

        stages = stages_of(model)
        preprocessing = weights.transforms()

        inputs = []
        for image_info in images:
            path = IMAGE_ROOT / image_info["path"]
            with Image.open(path) as image:
                tensor = preprocessing(
                    image.convert("RGB")
                ).unsqueeze(0)
            assert list(tensor.shape) == [1, 3, 224, 224]
            inputs.append(tensor)

        # CPU reference logits: never included in timed latency.
        with torch.inference_mode():
            references = [
                model(x).detach().clone()
                for x in inputs
            ]

        assert all(
            tuple(y.shape) == (1, 1000)
            and bool(torch.isfinite(y).all())
            for y in references
        )

        indices = list(range(16))
        seed = (
            20261015
            + 100 * process_repeat
            + (0 if name == "convnext_base" else 1)
        )
        random.Random(seed).shuffle(indices)
        image_orders[name] = indices

        full_devices = ["mps"] * 6
        hybrid_devices = (
            ["cpu"] + ["mps"] * 5
            if name == "convnext_base"
            else ["mps"] * 5 + ["cpu"]
        )

        full_layout = (
            "FULL_MPS", None, full_devices
        )
        hybrid_layout = (
            hybrid_name, cut, hybrid_devices
        )

        for image_order, image_index in enumerate(
            indices, start=1
        ):
            x_cpu = inputs[image_index]
            reference = references[image_index]
            image_info = images[image_index]

            # Each image receives opposite placement order
            # in the second independent process.
            full_first = (
                (image_index + process_repeat) % 2 == 0
            )

            layouts = (
                [full_layout, hybrid_layout]
                if full_first
                else [hybrid_layout, full_layout]
            )

            for execution_order, (
                mapping, cut_after, devices
            ) in enumerate(layouts, start=1):

                # Weight placement excluded from timed calls.
                for stage, device in zip(stages, devices):
                    stage.to(device)

                torch.mps.synchronize()

                with torch.inference_mode():
                    actual = forward(
                        stages, x_cpu, devices
                    )

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
                    f"{name} image={image_index} "
                    f"mapping={mapping}"
                )
                assert top1_agreement

                events = make_events(
                    devices, inv, x_cpu, reference
                )

                config = {
                    "model": name,
                    "mapping": mapping,
                    "cut_after": cut_after,
                    "devices": devices,
                    "image_sha256": image_info["sha256"],
                    "weight_sha256": expected_weight_sha,
                    "manifest_sha256": manifest_sha,
                    "cpu_threads": CPU_THREADS,
                    "dtype": "float32",
                    "batch_size": 1,
                    "input_device": "cpu",
                    "output_device": "cpu",
                    "transfers_timed": True,
                    "torch": torch.__version__,
                }

                config_id = "cfg_" + digest(config)
                run_id = "run_" + uuid.uuid4().hex
                elapsed_values = []

                with torch.inference_mode():
                    for _ in range(WARMUPS):
                        warmup_output = forward(
                            stages, x_cpu, devices
                        )
                        del warmup_output

                    for iteration in range(1, TIMED + 1):
                        torch.mps.synchronize()

                        start_ns = time.perf_counter_ns()
                        output = forward(
                            stages, x_cpu, devices
                        )
                        elapsed_ns = (
                            time.perf_counter_ns()
                            - start_ns
                        )
                        del output

                        assert elapsed_ns > 0
                        elapsed_ms = elapsed_ns / 1e6
                        elapsed_values.append(elapsed_ms)

                        raw_rows.append({
                            "process_repeat": process_repeat,
                            "process_id": process_id,
                            "run_id": run_id,
                            "config_id": config_id,
                            "model": name,
                            "image_index": image_index,
                            "image_path": image_info["path"],
                            "image_sha256": image_info["sha256"],
                            "mapping": mapping,
                            "cut_after": cut_after,
                            "image_order": image_order,
                            "execution_order": execution_order,
                            "iteration": iteration,
                            "elapsed_ns": elapsed_ns,
                            "elapsed_ms": elapsed_ms,
                        })

                summary_rows.append({
                    "process_repeat": process_repeat,
                    "process_id": process_id,
                    "run_id": run_id,
                    "config_id": config_id,
                    "model": name,
                    "image_index": image_index,
                    "image_path": image_info["path"],
                    "image_sha256": image_info["sha256"],
                    "mapping": mapping,
                    "cut_after": cut_after,
                    "device_sequence": json.dumps(devices),
                    "image_order": image_order,
                    "execution_order": execution_order,
                    "warmups": WARMUPS,
                    "repetitions": TIMED,
                    "transfer_events": json.dumps(events),
                    "logical_transfer_bytes": sum(
                        event["logical_bytes"]
                        for event in events
                    ),
                    "numerical_pass": int(numerical_pass),
                    "top1_agreement": int(top1_agreement),
                    "max_abs_error": max_error,
                    "median_ms": statistics.median(
                        elapsed_values
                    ),
                    "mean_ms": statistics.mean(
                        elapsed_values
                    ),
                    "min_ms": min(elapsed_values),
                    "max_ms": max(elapsed_values),
                })

            print(
                f"  Image {image_order:02}/16 "
                f"(index={image_index:02}) "
                f"pair PASS",
                flush=True
            )

        del model, stages, inputs, references
        gc.collect()
        torch.mps.empty_cache()

    assert len(summary_rows) == 64
    assert len(raw_rows) == 640

    raw_file = process_dir / "raw.csv"
    summary_file = process_dir / "summary.csv"
    metadata_file = process_dir / "metadata.json"

    write_csv(raw_file, RAW_FIELDS, raw_rows)
    write_csv(
        summary_file, SUMMARY_FIELDS, summary_rows
    )

    metadata = {
        "protocol": "T09_image_paired_latency_v1",
        "process_repeat": process_repeat,
        "process_id": process_id,
        "script_sha256": sha(__file__),
        "validation_sha256": sha(VALIDATION),
        "inventory_sha256": sha(INVENTORY),
        "sources": source_records,
        "image_sha256": [
            info["sha256"] for info in images
        ],
        "image_orders": image_orders,
        "warmups": WARMUPS,
        "timed_iterations": TIMED,
        "cpu_threads": CPU_THREADS,
        "batch_size": 1,
        "precision": "float32",
        "atol": 1e-3,
        "rtol": 1e-3,
        "input_device": "cpu",
        "output_device": "cpu",
        "preprocessing_timed": False,
        "device_transfers_timed": True,
        "model_placement_timed": False,
        "mps_synchronization": (
            "before and at end of each timed call"
        ),
        "logical_transfer_bytes": (
            "Tensor payload sizes; not physical "
            "Apple unified-memory traffic"
        ),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": sys.version.split()[0],
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip(),
        "mps_fallback": os.environ[
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ],
        "limitations": [
            "Two independent processes only",
            "Sixteen fixed ImageNetV2 images",
            "Ten timed calls per image/configuration",
            "Repeated calls are not independent images",
            "Thermal and background load not controlled",
            "Only selected single-cut hybrid layouts",
            "Fixed four CPU threads",
            "No energy or peak-memory measurement",
            "No overlapping pipeline execution",
        ],
    }

    metadata_file.write_text(
        json.dumps(metadata, indent=2)
    )

    write_checksums(
        process_dir,
        [raw_file, summary_file, metadata_file]
    )

    print(
        f"\nPROCESS {process_repeat}: PASS "
        "(64 configurations, 640 timings)",
        flush=True
    )

def aggregate():
    all_raw = []
    all_summary = []
    process_ids = set()
    worker_hashes = {}
    validation = json.loads(VALIDATION.read_text())

    for process_repeat in range(1, PROCESSES + 1):
        directory = ROOT / f"process_{process_repeat:02}"
        verify_checksums(directory)

        meta = json.loads(
            (directory / "metadata.json").read_text()
        )

        assert meta["process_repeat"] == process_repeat
        assert meta["script_sha256"] == sha(__file__)
        assert meta["validation_sha256"] == sha(VALIDATION)
        assert meta["inventory_sha256"] == sha(INVENTORY)
        assert meta["process_id"] not in process_ids

        process_ids.add(meta["process_id"])
        worker_hashes[
            directory.name
        ] = sha(directory / "checksums.json")

        for name, source in meta["sources"].items():
            assert sha(source["manifest"]) == (
                source["manifest_sha256"]
            )
            assert source["weight_sha256"] == (
                validation["models"][name]["weight_sha256"]
            )

        raw = read_csv(directory / "raw.csv")
        summary = read_csv(directory / "summary.csv")

        assert len(raw) == 640
        assert len(summary) == 64

        all_raw.extend(raw)
        all_summary.extend(summary)

    assert len(process_ids) == 2
    assert len(all_raw) == 1280
    assert len(all_summary) == 128
    assert len({
        r["run_id"] for r in all_summary
    }) == 128

    # Same image/config repeated in two distinct processes.
    assert len({
        r["config_id"] for r in all_summary
    }) == 64

    by_run = defaultdict(list)
    for row in all_raw:
        by_run[row["run_id"]].append(row)

    assert len(by_run) == 128

    for row in all_summary:
        measurements = by_run[row["run_id"]]
        assert len(measurements) == TIMED

        iterations = sorted(
            int(v["iteration"]) for v in measurements
        )
        assert iterations == list(range(1, TIMED + 1))

        assert int(row["numerical_pass"]) == 1
        assert int(row["top1_agreement"]) == 1
        assert int(row["warmups"]) == WARMUPS

        values = []
        for sample in measurements:
            assert sample["config_id"] == row["config_id"]
            assert sample["image_sha256"] == row["image_sha256"]
            assert sample["model"] == row["model"]
            assert sample["mapping"] == row["mapping"]
            assert sample["cut_after"] == row["cut_after"]

            elapsed_ns = int(sample["elapsed_ns"])
            assert elapsed_ns > 0

            elapsed_ms = float(sample["elapsed_ms"])
            assert math.isclose(
                elapsed_ms, elapsed_ns / 1e6,
                abs_tol=1e-9
            )
            values.append(elapsed_ms)

        assert math.isclose(
            statistics.median(values),
            float(row["median_ms"]),
            abs_tol=1e-9
        )

    grouped = defaultdict(list)

    for row in all_summary:
        key = (
            int(row["process_repeat"]),
            row["model"],
            int(row["image_index"]),
        )
        grouped[key].append(row)

    assert len(grouped) == 64
    assert all(len(rows) == 2 for rows in grouped.values())

    paired = []

    for key, rows in sorted(grouped.items()):
        process_repeat, model, image_index = key

        full = next(
            r for r in rows
            if r["mapping"] == "FULL_MPS"
        )
        hybrid = next(
            r for r in rows
            if r["mapping"] != "FULL_MPS"
        )

        assert {
            int(full["execution_order"]),
            int(hybrid["execution_order"]),
        } == {1, 2}

        assert full["image_sha256"] == hybrid["image_sha256"]

        full_ms = float(full["median_ms"])
        hybrid_ms = float(hybrid["median_ms"])

        paired.append({
            "process_repeat": process_repeat,
            "model": model,
            "image_index": image_index,
            "image_path": full["image_path"],
            "image_sha256": full["image_sha256"],
            "full_mps_ms": full_ms,
            "hybrid_ms": hybrid_ms,
            "delta_ms": hybrid_ms - full_ms,
            "delta_percent": (
                100 * (hybrid_ms / full_ms - 1)
            ),
            "hybrid_faster": int(hybrid_ms < full_ms),
            "full_mps_order": int(
                full["execution_order"]
            ),
            "hybrid_order": int(
                hybrid["execution_order"]
            ),
        })

    assert len(paired) == 64

    print("\n=== T09 IMAGE-PAIRED COMPARISON ===")

    for model in ("convnext_base", "convnext_large"):
        rows = [
            r for r in paired if r["model"] == model
        ]
        assert len(rows) == 32

        deltas = [r["delta_percent"] for r in rows]
        wins = sum(r["hybrid_faster"] for r in rows)

        both_faster = 0
        both_slower = 0
        mixed = 0

        for image_index in range(16):
            image_rows = [
                r for r in rows
                if r["image_index"] == image_index
            ]
            assert len(image_rows) == 2

            orders = {
                r["full_mps_order"] for r in image_rows
            }
            assert orders == {1, 2}

            image_deltas = [
                r["delta_percent"] for r in image_rows
            ]

            if all(v < 0 for v in image_deltas):
                both_faster += 1
            elif all(v > 0 for v in image_deltas):
                both_slower += 1
            else:
                mixed += 1

        print(f"\n{model}:")
        print(
            f"  Hybrid faster: {wins}/32 "
            "image-process pairs"
        )
        print(
            "  Median paired delta: "
            f"{statistics.median(deltas):+.3f}%"
        )
        print(
            "  Paired delta range: "
            f"{min(deltas):+.3f}% to "
            f"{max(deltas):+.3f}%"
        )
        print(
            "  Image agreement across 2 processes: "
            f"{both_faster} both faster, "
            f"{both_slower} both slower, "
            f"{mixed} mixed"
        )

        for process_repeat in (1, 2):
            process_deltas = [
                r["delta_percent"] for r in rows
                if r["process_repeat"] == process_repeat
            ]
            assert len(process_deltas) == 16

            print(
                f"  Process {process_repeat} "
                f"median paired delta: "
                f"{statistics.median(process_deltas):+.3f}%"
            )

    raw_path = ROOT / "all_raw.csv"
    summary_path = ROOT / "all_summary.csv"
    paired_path = ROOT / "paired_images.csv"
    metadata_path = ROOT / "campaign_metadata.json"

    write_csv(raw_path, RAW_FIELDS, all_raw)
    write_csv(
        summary_path, SUMMARY_FIELDS, all_summary
    )
    write_csv(
        paired_path, list(paired[0]), paired
    )

    campaign_meta = {
        "protocol": "T09_image_paired_latency_v1",
        "independent_processes": PROCESSES,
        "models": [
            "convnext_base", "convnext_large"
        ],
        "fixed_image_count": 16,
        "warmups_per_configuration": WARMUPS,
        "timed_calls_per_configuration": TIMED,
        "configuration_runs": len(all_summary),
        "timed_samples": len(all_raw),
        "paired_image_process_comparisons": len(paired),
        "image_order": "Deterministic shuffled order per process",
        "placement_order": (
            "Opposite for each image across two processes"
        ),
        "comparison_unit": (
            "Per-image paired configuration medians "
            "within each process"
        ),
        "source_validation": str(VALIDATION),
        "source_validation_sha256": sha(VALIDATION),
        "source_inventory": str(INVENTORY),
        "source_inventory_sha256": sha(INVENTORY),
        "script_sha256": sha(__file__),
        "worker_checksums_sha256": worker_hashes,
        "limitations": [
            "Exploratory image-variability study",
            "Not a population-level statistical claim",
            "Only two independent processes",
            "Images fixed from earlier validation",
            "Fixed four CPU threads",
            "No energy or pipeline-overlap measurements",
        ],
    }

    metadata_path.write_text(
        json.dumps(campaign_meta, indent=2)
    )

    write_checksums(
        ROOT,
        [
            Path(__file__),
            raw_path,
            summary_path,
            paired_path,
            metadata_path,
        ]
    )

    print("\n==============================")
    print("T09 IMAGE-PAIRED CAMPAIGN: PASS")
    print("INDEPENDENT PROCESSES: 2")
    print("IMAGES PER MODEL: 16")
    print("CONFIGURATION RUNS: 128")
    print("TIMED SAMPLES: 1280")
    print("PAIRED COMPARISONS: 64")
    print("RESULT DIRECTORY:", ROOT)
    print("==============================")

def main():
    assert ROOT.is_dir()

    if len(sys.argv) == 4 and sys.argv[2] == "--worker":
        run_worker(int(sys.argv[3]))
        return

    print(
        "Starting T09 image-paired campaign...",
        flush=True
    )

    for process_repeat in range(1, PROCESSES + 1):
        print(
            f"\n=== INDEPENDENT PROCESS "
            f"{process_repeat}/{PROCESSES} ===",
            flush=True
        )

        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                str(ROOT),
                "--worker",
                str(process_repeat),
            ],
            check=True,
        )

    aggregate()

if __name__ == "__main__":
    main()
