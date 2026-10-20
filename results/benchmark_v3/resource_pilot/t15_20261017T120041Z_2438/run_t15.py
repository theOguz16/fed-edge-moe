import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(sys.argv[1]).resolve()

BASE = Path("results/benchmark_v3")
VALIDATION = BASE / (
    "validation/t05_20261015T173344Z_2438/metadata.json"
)
INVENTORY = BASE / (
    "inventory/inv_20261015T175122Z_2438/structural_inventory.csv"
)
MANIFEST = Path(
    "configs/partition_manifests_v2/convnext_large__fine41.json"
)
HOLDOUT_DIR = BASE / (
    "holdout_thread_confirmation/t12_20261015T192923Z_2438"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/imagenetv2-matched-frequency-format-val"
)

WARMUPS = 3
TIMED = 12

# Interleaved execution; memory measurements are
# exploratory and are not paired-process latency tests.
SCHEDULE = [
    (1, "FULL_MPS"),
    (4, "MPS_TO_CPU"),
    (4, "FULL_MPS"),
    (1, "MPS_TO_CPU"),
]

RAW_FIELDS = [
    "process_number", "process_id", "run_id",
    "threads", "mapping", "iteration",
    "elapsed_ns", "elapsed_ms"
]
SUMMARY_FIELDS = [
    "process_number", "process_id", "run_id",
    "threads", "mapping", "image_sha256",
    "warmups", "timed_calls", "median_ms",
    "mean_ms", "min_ms", "max_ms",
    "numerical_pass", "top1_agreement",
    "max_abs_error"
]
MEMORY_FIELDS = [
    "process_number", "process_id",
    "threads", "mapping", "checkpoint",
    "rss_bytes", "mps_allocated_bytes",
    "mps_driver_bytes", "mps_recommended_bytes"
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def readcsv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def writecsv(path, columns, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def checksums(directory, files):
    (directory / "checksums.json").write_text(
        json.dumps({
            p.name: sha(p) for p in files
        }, indent=2)
    )

def verify(directory):
    records = json.loads(
        (directory / "checksums.json").read_text()
    )
    for filename, expected in records.items():
        path = directory / filename
        assert path.is_file(), path
        assert sha(path) == expected, path

def sources():
    validation = json.loads(VALIDATION.read_text())
    holdout = json.loads(
        (HOLDOUT_DIR / "holdout_images.json").read_text()
    )
    assert len(holdout) == 16

    previous = {
        item["sha256"] for item in validation["images"]
    }
    assert holdout[0]["sha256"] not in previous

    image = IMAGE_ROOT / holdout[0]["path"]
    assert image.is_file()
    assert sha(image) == holdout[0]["sha256"]

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["group_count"] == 41

    inventory = sorted(
        [
            row for row in readcsv(INVENTORY)
            if row["model"] == "convnext_large"
        ],
        key=lambda row: int(row["stage_id"][1:])
    )
    assert len(inventory) == 6
    assert all(
        row["manifest_sha256"] == sha(MANIFEST)
        for row in inventory
    )

    return validation, holdout[0], image

def rss_bytes():
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True
    )
    # macOS ps RSS is expressed in KiB.
    return int(result.stdout.strip()) * 1024

def mps_counter(torch, name):
    method = getattr(torch.mps, name, None)

    if not callable(method):
        return None

    try:
        value = int(method())
        assert value >= 0
        return value
    except (RuntimeError, NotImplementedError):
        return None

def worker(number, threads, mapping):
    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torchvision.models import (
        convnext_large, ConvNeXt_Large_Weights
    )

    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()

    torch.set_num_threads(threads)
    assert torch.get_num_threads() == threads
    torch.manual_seed(2026)

    directory = ROOT / f"process_{number:02d}"
    directory.mkdir(parents=True, exist_ok=False)

    process_id = uuid.uuid4().hex
    run_id = uuid.uuid4().hex
    memory = []

    def snapshot(checkpoint):
        torch.mps.synchronize()
        row = {
            "process_number": number,
            "process_id": process_id,
            "threads": threads,
            "mapping": mapping,
            "checkpoint": checkpoint,
            "rss_bytes": rss_bytes(),
            "mps_allocated_bytes": mps_counter(
                torch, "current_allocated_memory"
            ),
            "mps_driver_bytes": mps_counter(
                torch, "driver_allocated_memory"
            ),
            "mps_recommended_bytes": mps_counter(
                torch, "recommended_max_memory"
            ),
        }
        memory.append(row)

    validation, image_info, image_path = sources()

    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1

    checkpoint_file = (
        Path(torch.hub.get_dir()) / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    expected_weight_sha = validation[
        "models"
    ]["convnext_large"]["weight_sha256"]

    assert checkpoint_file.is_file()
    assert sha(checkpoint_file) == expected_weight_sha

    torch.mps.empty_cache()
    snapshot("before_model_load")

    model = convnext_large(weights=None).eval().cpu()

    state = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state, strict=True)
    del state

    stages = [
        model.features[0],
        model.features[1],
        nn.Sequential(
            model.features[2], model.features[3]
        ),
        nn.Sequential(
            model.features[4], model.features[5]
        ),
        nn.Sequential(
            model.features[6], model.features[7]
        ),
        nn.Sequential(model.avgpool, model.classifier),
    ]

    with Image.open(image_path) as image:
        x_cpu = weights.transforms()(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tuple(x_cpu.shape) == (1, 3, 224, 224)

    with torch.inference_mode():
        reference = model(x_cpu).detach().clone()

    assert tuple(reference.shape) == (1, 1000)
    assert bool(torch.isfinite(reference).all())

    snapshot("cpu_model_and_reference_ready")

    devices = (
        ["mps"] * 6
        if mapping == "FULL_MPS"
        else ["mps"] * 5 + ["cpu"]
    )

    for stage, device in zip(stages, devices):
        stage.to(device)

    torch.mps.synchronize()
    snapshot("after_weight_placement")

    def forward():
        y = x_cpu
        for stage, device in zip(stages, devices):
            if y.device.type != device:
                y = y.to(device)
            y = stage(y)

        if y.device.type != "cpu":
            y = y.cpu()

        torch.mps.synchronize()
        return y

    with torch.inference_mode():
        actual = forward()

    numerical_ok = bool(torch.allclose(
        actual, reference, atol=1e-3, rtol=1e-3
    ))
    top1_ok = (
        actual.argmax().item()
        == reference.argmax().item()
    )
    max_error = float(
        (actual - reference).abs().max().item()
    )

    assert numerical_ok and top1_ok
    del actual

    with torch.inference_mode():
        for _ in range(WARMUPS):
            output = forward()
            del output

    snapshot("after_warmups")

    samples = []
    raw = []

    with torch.inference_mode():
        for iteration in range(1, TIMED + 1):
            torch.mps.synchronize()

            start = time.perf_counter_ns()
            output = forward()
            elapsed_ns = time.perf_counter_ns() - start

            del output
            assert elapsed_ns > 0
            elapsed_ms = elapsed_ns / 1e6
            assert math.isfinite(elapsed_ms)

            samples.append(elapsed_ms)
            raw.append({
                "process_number": number,
                "process_id": process_id,
                "run_id": run_id,
                "threads": threads,
                "mapping": mapping,
                "iteration": iteration,
                "elapsed_ns": elapsed_ns,
                "elapsed_ms": elapsed_ms
            })

    snapshot("after_timed_calls")

    summary = [{
        "process_number": number,
        "process_id": process_id,
        "run_id": run_id,
        "threads": threads,
        "mapping": mapping,
        "image_sha256": image_info["sha256"],
        "warmups": WARMUPS,
        "timed_calls": TIMED,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "numerical_pass": int(numerical_ok),
        "top1_agreement": int(top1_ok),
        "max_abs_error": max_error
    }]

    assert len(raw) == 12
    assert len(memory) == 5

    raw_path = directory / "raw.csv"
    summary_path = directory / "summary.csv"
    memory_path = directory / "memory_snapshots.csv"
    metadata_path = directory / "metadata.json"

    writecsv(raw_path, RAW_FIELDS, raw)
    writecsv(summary_path, SUMMARY_FIELDS, summary)
    writecsv(memory_path, MEMORY_FIELDS, memory)

    metadata_path.write_text(json.dumps({
        "protocol": "T15_sampled_resource_pilot_v1",
        "process_number": number,
        "process_id": process_id,
        "threads": threads,
        "mapping": mapping,
        "script_sha256": sha(__file__),
        "validation_sha256": sha(VALIDATION),
        "inventory_sha256": sha(INVENTORY),
        "manifest_sha256": sha(MANIFEST),
        "holdout_sha256": sha(
            HOLDOUT_DIR / "holdout_images.json"
        ),
        "image": image_info,
        "weight_sha256": expected_weight_sha,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": sys.version.split()[0],
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "mps_fallback": os.environ[
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ],
        "batch_size": 1,
        "precision": "float32",
        "cpu_input_output": True,
        "transfers_timed": True,
        "preprocessing_timed": False,
        "weight_placement_timed": False,
        "memory_polling_inside_timed_calls": False,
        "rss_definition": "macOS ps current RSS, not peak",
        "mps_memory_definition": (
            "PyTorch allocator/driver snapshots; "
            "not physical-system peak usage"
        ),
        "energy_measured": False,
        "network_transfer_measured": False,
        "resource_constraints_enforced": False
    }, indent=2))

    checksums(
        directory,
        [raw_path, summary_path, memory_path, metadata_path]
    )

    print(
        f"PROCESS {number}/4: PASS | "
        f"threads={threads} mapping={mapping} | "
        f"median={statistics.median(samples):.3f} ms",
        flush=True
    )

def mib(value):
    if value in ("", None):
        return "N/A"
    return f"{int(value)/1048576:.1f}"

def aggregate():
    all_raw = []
    all_summary = []
    all_memory = []
    ids = set()

    for number, (threads, mapping) in enumerate(
        SCHEDULE, start=1
    ):
        directory = ROOT / f"process_{number:02d}"
        verify(directory)

        meta = json.loads(
            (directory / "metadata.json").read_text()
        )
        assert meta["process_number"] == number
        assert meta["threads"] == threads
        assert meta["mapping"] == mapping
        assert meta["script_sha256"] == sha(__file__)
        assert meta["validation_sha256"] == sha(VALIDATION)
        assert meta["inventory_sha256"] == sha(INVENTORY)
        assert meta["manifest_sha256"] == sha(MANIFEST)
        assert meta["process_id"] not in ids
        ids.add(meta["process_id"])

        raw = readcsv(directory / "raw.csv")
        summary = readcsv(directory / "summary.csv")
        memory = readcsv(
            directory / "memory_snapshots.csv"
        )

        assert len(raw) == 12
        assert len(summary) == 1
        assert len(memory) == 5

        row = summary[0]
        assert int(row["numerical_pass"]) == 1
        assert int(row["top1_agreement"]) == 1
        assert len({
            r["process_id"] for r in raw
        }) == 1
        assert len({
            r["process_id"] for r in memory
        }) == 1

        values = [
            float(r["elapsed_ms"]) for r in raw
        ]

        assert sorted(
            int(r["iteration"]) for r in raw
        ) == list(range(1, 13))

        assert all(
            math.isclose(
                float(r["elapsed_ms"]),
                int(r["elapsed_ns"]) / 1e6,
                abs_tol=1e-9
            ) for r in raw
        )

        assert math.isclose(
            statistics.median(values),
            float(row["median_ms"]),
            abs_tol=1e-9
        )

        assert [r["checkpoint"] for r in memory] == [
            "before_model_load",
            "cpu_model_and_reference_ready",
            "after_weight_placement",
            "after_warmups",
            "after_timed_calls"
        ]

        assert all(
            int(r["rss_bytes"]) > 0
            for r in memory
        )

        all_raw.extend(raw)
        all_summary.extend(summary)
        all_memory.extend(memory)

    assert len(ids) == 4
    assert len(all_raw) == 48
    assert len(all_summary) == 4
    assert len(all_memory) == 20

    raw_path = ROOT / "all_raw.csv"
    summary_path = ROOT / "all_summary.csv"
    memory_path = ROOT / "all_memory_snapshots.csv"
    metadata_path = ROOT / "campaign_metadata.json"

    writecsv(raw_path, RAW_FIELDS, all_raw)
    writecsv(summary_path, SUMMARY_FIELDS, all_summary)
    writecsv(memory_path, MEMORY_FIELDS, all_memory)

    metadata_path.write_text(json.dumps({
        "protocol": "T15_sampled_resource_pilot_v1",
        "model": "convnext_large",
        "independent_processes": 4,
        "thread_settings": [1, 4],
        "mappings": ["FULL_MPS", "MPS_TO_CPU"],
        "configuration_runs": 4,
        "timed_samples": 48,
        "memory_snapshots": 20,
        "rss_is_peak": False,
        "mps_driver_bytes_is_peak": False,
        "mps_allocator_bytes_is_peak": False,
        "cpu_rss_and_mps_bytes_additive": False,
        "energy_measured": False,
        "memory_constraint_enforced": False,
        "actual_scheduler_executed": False,
        "interpretation": (
            "Capability and sampled-memory pilot only; "
            "no latency superiority or peak memory claim."
        )
    }, indent=2))

    checksums(
        ROOT,
        [
            Path(__file__),
            raw_path,
            summary_path,
            memory_path,
            metadata_path
        ]
    )
    verify(ROOT)

    print("\n=== T15 SAMPLED RESOURCE RESULTS ===")
    print(
        f"{'Threads':>7} {'Mapping':<11} "
        f"{'RSS MiB':>10} "
        f"{'MPS Alloc MiB':>14} "
        f"{'MPS Driver MiB':>16}"
    )

    for row in all_summary:
        selected = next(
            r for r in all_memory
            if r["process_id"] == row["process_id"]
            and r["checkpoint"] == "after_timed_calls"
        )

        print(
            f"{int(row['threads']):>7} "
            f"{row['mapping']:<11} "
            f"{mib(selected['rss_bytes']):>10} "
            f"{mib(selected['mps_allocated_bytes']):>14} "
            f"{mib(selected['mps_driver_bytes']):>16}"
        )

    print("\nT15 SAMPLED RESOURCE PILOT: PASS")
    print("INDEPENDENT PROCESSES: 4")
    print("TIMED SAMPLES: 48")
    print("MEMORY SNAPSHOTS: 20")
    print("ENERGY MEASURED: NO")
    print("TRUE PEAK MEMORY MEASURED: NO")
    print("RESULT DIRECTORY:", ROOT)

def main():
    assert ROOT.is_dir()

    if len(sys.argv) == 6 and sys.argv[2] == "--worker":
        worker(
            int(sys.argv[3]),
            int(sys.argv[4]),
            sys.argv[5]
        )
        return

    verify(HOLDOUT_DIR)
    print("T12 SOURCE CHECKSUMS: PASS", flush=True)

    for number, (threads, mapping) in enumerate(
        SCHEDULE, start=1
    ):
        print(
            f"\n=== PROCESS {number}/4 "
            f"THREADS={threads} {mapping} ===",
            flush=True
        )

        subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                str(ROOT),
                "--worker",
                str(number),
                str(threads),
                mapping
            ],
            check=True
        )

    aggregate()

if __name__ == "__main__":
    main()
