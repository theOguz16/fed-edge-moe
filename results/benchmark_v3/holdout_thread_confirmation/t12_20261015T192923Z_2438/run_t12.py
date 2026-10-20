import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()

VALIDATION = Path(
    "results/benchmark_v3/validation/"
    "t05_20261015T173344Z_2438/metadata.json"
)
INVENTORY = Path(
    "results/benchmark_v3/inventory/"
    "inv_20261015T175122Z_2438/structural_inventory.csv"
)
MANIFEST = Path(
    "configs/partition_manifests_v2/"
    "convnext_large__fine41.json"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/"
    "imagenetv2-matched-frequency-format-val"
)

THREADS = [1, 4]
ROUNDS = 4
WARMUPS = 3
TIMED = 12
IMAGE_COUNT = 16
SEED = 20261016

RAW_FIELDS = [
    "process_number", "round", "threads",
    "run_id", "config_id", "image_index",
    "image_path", "image_sha256", "mapping",
    "placement_order", "iteration",
    "elapsed_ns", "elapsed_ms",
]

SUMMARY_FIELDS = [
    "process_number", "round", "threads",
    "run_id", "config_id", "image_index",
    "image_path", "image_sha256", "mapping",
    "placement_order", "device_sequence",
    "transfer_events", "logical_transfer_bytes",
    "numerical_pass", "top1_agreement",
    "max_abs_error", "warmups", "repetitions",
    "median_ms", "mean_ms", "min_ms", "max_ms",
]

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

def readcsv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def writecsv(path, fields, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def write_checksums(directory, files):
    (directory / "checksums.json").write_text(
        json.dumps(
            {p.name: sha(p) for p in files},
            indent=2
        )
    )

def verify(directory):
    checks = json.loads(
        (directory / "checksums.json").read_text()
    )
    for filename, expected in checks.items():
        assert sha(directory / filename) == expected, (
            f"SHA256 mismatch: {directory / filename}"
        )

def select_holdout():
    validation = json.loads(VALIDATION.read_text())

    prior = {
        item["path"] for item in validation["images"]
    }
    assert len(prior) == 16

    eligible = sorted(
        (
            p for p in IMAGE_ROOT.rglob("*")
            if p.suffix.lower() in {
                ".jpg", ".jpeg", ".png"
            }
            and str(p.relative_to(IMAGE_ROOT)) not in prior
        ),
        key=lambda p: str(p.relative_to(IMAGE_ROOT))
    )

    assert len(eligible) >= IMAGE_COUNT

    selected = random.Random(SEED).sample(
        eligible, IMAGE_COUNT
    )

    images = [
        {
            "index": index,
            "path": str(p.relative_to(IMAGE_ROOT)),
            "sha256": sha(p),
        }
        for index, p in enumerate(selected)
    ]

    assert len({
        item["sha256"] for item in images
    }) == IMAGE_COUNT

    assert not (
        {item["path"] for item in images} & prior
    )

    return images

def worker(number, round_no, threads):
    import gc
    import platform
    import time
    from urllib.parse import urlparse

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
    validation = json.loads(VALIDATION.read_text())
    images = select_holdout()

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["group_count"] == 41

    expected_paths = (
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

    assert [
        g["modules"] for g in manifest["groups"]
    ] == expected_paths

    inv = sorted(
        [
            row for row in readcsv(INVENTORY)
            if row["model"] == "convnext_large"
        ],
        key=lambda row: int(row["stage_id"][1:])
    )

    assert len(inv) == 6
    assert all(
        row["manifest_sha256"] == sha(MANIFEST)
        for row in inv
    )

    weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
    checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    expected_weight_sha = validation[
        "models"
    ]["convnext_large"]["weight_sha256"]

    assert checkpoint.is_file()
    assert sha(checkpoint) == expected_weight_sha

    model = convnext_large(weights=None).eval().cpu()
    state = torch.load(
        checkpoint,
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
        nn.Sequential(
            model.avgpool, model.classifier
        ),
    ]

    preprocess = weights.transforms()
    inputs = []

    for item in images:
        path = IMAGE_ROOT / item["path"]
        assert sha(path) == item["sha256"]

        with Image.open(path) as image:
            x = preprocess(
                image.convert("RGB")
            ).unsqueeze(0)

        assert tuple(x.shape) == (1, 3, 224, 224)
        assert x.dtype == torch.float32
        inputs.append(x)

    with torch.inference_mode():
        references = [
            model(x).detach().clone()
            for x in inputs
        ]

    assert all(
        tuple(ref.shape) == (1, 1000)
        and bool(torch.isfinite(ref).all())
        for ref in references
    )

    full = ("FULL_MPS", ["mps"] * 6)
    hybrid = (
        "MPS_TO_CPU",
        ["mps"] * 5 + ["cpu"]
    )

    def forward(x_cpu, devices):
        y = x_cpu

        for stage, device in zip(stages, devices):
            if y.device.type != device:
                y = y.to(device)
            y = stage(y)

        if y.device.type != "cpu":
            y = y.cpu()

        torch.mps.synchronize()
        return y

    image_order = list(range(IMAGE_COUNT))
    random.Random(
        SEED + round_no * 100 + threads
    ).shuffle(image_order)

    raw_rows = []
    summary_rows = []
    process_deltas = []

    for image_pos, image_index in enumerate(
        image_order, 1
    ):
        x_cpu = inputs[image_index]
        reference = references[image_index]
        image_info = images[image_index]

        # Each image receives each mapping first twice
        # over four rounds.
        full_first = (
            (image_index + round_no) % 2 == 0
        )

        layouts = (
            [full, hybrid]
            if full_first
            else [hybrid, full]
        )

        pair_medians = {}

        for placement_order, (
            mapping, devices
        ) in enumerate(layouts, 1):

            # Weight placement is not timed.
            for stage, device in zip(stages, devices):
                stage.to(device)

            torch.mps.synchronize()

            with torch.inference_mode():
                actual = forward(x_cpu, devices)

            assert actual.shape == reference.shape
            assert bool(torch.isfinite(actual).all())

            numerical_ok = bool(torch.allclose(
                reference, actual,
                atol=1e-3, rtol=1e-3
            ))

            top1_ok = (
                reference.argmax().item()
                == actual.argmax().item()
            )

            assert numerical_ok and top1_ok

            max_error = float(
                (reference - actual).abs().max().item()
            )

            if mapping == "FULL_MPS":
                transfer_events = [
                    {
                        "at": "input",
                        "from": "cpu",
                        "to": "mps",
                        "logical_bytes":
                            x_cpu.numel() * x_cpu.element_size(),
                    },
                    {
                        "at": "output",
                        "from": "mps",
                        "to": "cpu",
                        "logical_bytes":
                            reference.numel() * reference.element_size(),
                    },
                ]
            else:
                transfer_events = [
                    {
                        "at": "input",
                        "from": "cpu",
                        "to": "mps",
                        "logical_bytes":
                            x_cpu.numel() * x_cpu.element_size(),
                    },
                    {
                        "at": "after_S4",
                        "from": "mps",
                        "to": "cpu",
                        "logical_bytes": int(
                            inv[4]["output_activation_bytes"]
                        ),
                    },
                ]

            config = {
                "model": "convnext_large",
                "threads": threads,
                "mapping": mapping,
                "cut_after": (
                    5 if mapping == "MPS_TO_CPU"
                    else None
                ),
                "devices": devices,
                "image_sha256": image_info["sha256"],
                "weight_sha256": expected_weight_sha,
                "manifest_sha256": sha(MANIFEST),
                "precision": "float32",
                "batch_size": 1,
                "cpu_input_and_output": True,
                "torch": torch.__version__,
            }

            config_id = "cfg_" + digest(config)
            run_id = "run_" + uuid.uuid4().hex
            values = []

            with torch.inference_mode():
                for _ in range(WARMUPS):
                    output = forward(x_cpu, devices)
                    del output

                for iteration in range(1, TIMED + 1):
                    torch.mps.synchronize()

                    start = time.perf_counter_ns()
                    output = forward(x_cpu, devices)
                    elapsed_ns = (
                        time.perf_counter_ns() - start
                    )
                    del output

                    assert elapsed_ns > 0
                    elapsed_ms = elapsed_ns / 1e6
                    values.append(elapsed_ms)

                    raw_rows.append({
                        "process_number": number,
                        "round": round_no,
                        "threads": threads,
                        "run_id": run_id,
                        "config_id": config_id,
                        "image_index": image_index,
                        "image_path": image_info["path"],
                        "image_sha256": image_info["sha256"],
                        "mapping": mapping,
                        "placement_order": placement_order,
                        "iteration": iteration,
                        "elapsed_ns": elapsed_ns,
                        "elapsed_ms": elapsed_ms,
                    })

            median = statistics.median(values)
            pair_medians[mapping] = median

            summary_rows.append({
                "process_number": number,
                "round": round_no,
                "threads": threads,
                "run_id": run_id,
                "config_id": config_id,
                "image_index": image_index,
                "image_path": image_info["path"],
                "image_sha256": image_info["sha256"],
                "mapping": mapping,
                "placement_order": placement_order,
                "device_sequence": json.dumps(devices),
                "transfer_events": json.dumps(transfer_events),
                "logical_transfer_bytes": sum(
                    e["logical_bytes"]
                    for e in transfer_events
                ),
                "numerical_pass": int(numerical_ok),
                "top1_agreement": int(top1_ok),
                "max_abs_error": max_error,
                "warmups": WARMUPS,
                "repetitions": TIMED,
                "median_ms": median,
                "mean_ms": statistics.mean(values),
                "min_ms": min(values),
                "max_ms": max(values),
            })

        delta = 100 * (
            pair_medians["MPS_TO_CPU"]
            / pair_medians["FULL_MPS"] - 1
        )
        process_deltas.append(delta)

        print(
            f"  Image {image_pos:02}/16 "
            f"paired delta={delta:+.3f}% | PASS",
            flush=True
        )

    assert len(summary_rows) == 32
    assert len(raw_rows) == 32 * TIMED

    raw_path = directory / "raw.csv"
    summary_path = directory / "summary.csv"
    metadata_path = directory / "metadata.json"

    writecsv(raw_path, RAW_FIELDS, raw_rows)
    writecsv(
        summary_path, SUMMARY_FIELDS, summary_rows
    )

    metadata_path.write_text(
        json.dumps({
            "protocol": "T12_holdout_confirmation_v1",
            "process_number": number,
            "process_id": process_id,
            "round": round_no,
            "threads": threads,
            "observed_threads": torch.get_num_threads(),
            "model": "convnext_large",
            "weight_sha256": expected_weight_sha,
            "manifest_sha256": sha(MANIFEST),
            "validation_sha256": sha(VALIDATION),
            "inventory_sha256": sha(INVENTORY),
            "script_sha256": sha(__file__),
            "images": images,
            "image_order": image_order,
            "warmups": WARMUPS,
            "timed_iterations": TIMED,
            "batch_size": 1,
            "precision": "float32",
            "input_device": "cpu",
            "output_device": "cpu",
            "device_transfers_timed": True,
            "model_placement_timed": False,
            "preprocessing_timed": False,
            "mps_synchronization": (
                "before and after timed inference"
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
        }, indent=2)
    )

    write_checksums(
        directory, [
            raw_path, summary_path, metadata_path
        ]
    )

    print(
        f"PROCESS {number}/8: PASS | "
        f"median paired delta="
        f"{statistics.median(process_deltas):+.3f}%",
        flush=True
    )

    del model, stages, inputs, references
    gc.collect()
    torch.mps.empty_cache()

def aggregate():
    all_raw = []
    all_summary = []
    process_ids = set()
    images = select_holdout()

    for number in range(1, 9):
        directory = ROOT / f"process_{number:02d}"
        verify(directory)

        meta = json.loads(
            (directory / "metadata.json").read_text()
        )
        assert meta["process_number"] == number
        assert meta["script_sha256"] == sha(__file__)
        assert meta["validation_sha256"] == sha(VALIDATION)
        assert meta["inventory_sha256"] == sha(INVENTORY)
        assert meta["manifest_sha256"] == sha(MANIFEST)
        assert meta["images"] == images
        assert meta["process_id"] not in process_ids
        process_ids.add(meta["process_id"])

        local_raw = readcsv(directory / "raw.csv")
        local_summary = readcsv(
            directory / "summary.csv"
        )

        assert len(local_raw) == 384
        assert len(local_summary) == 32

        all_raw.extend(local_raw)
        all_summary.extend(local_summary)

    assert len(process_ids) == 8
    assert len(all_raw) == 3072
    assert len(all_summary) == 256
    assert len({
        row["run_id"] for row in all_summary
    }) == 256

    config_counts = Counter(
        r["config_id"] for r in all_summary
    )
    assert len(config_counts) == 64
    assert set(config_counts.values()) == {4}

    by_run = defaultdict(list)
    for row in all_raw:
        by_run[row["run_id"]].append(row)

    assert len(by_run) == 256

    for row in all_summary:
        samples = by_run[row["run_id"]]
        assert len(samples) == TIMED
        assert sorted(
            int(x["iteration"]) for x in samples
        ) == list(range(1, TIMED + 1))
        assert int(row["numerical_pass"]) == 1
        assert int(row["top1_agreement"]) == 1

        values = []

        for sample in samples:
            assert sample["config_id"] == row["config_id"]
            assert sample["image_sha256"] == row["image_sha256"]
            assert sample["mapping"] == row["mapping"]
            assert sample["threads"] == row["threads"]
            assert int(sample["elapsed_ns"]) > 0

            value = float(sample["elapsed_ms"])
            assert math.isclose(
                value,
                int(sample["elapsed_ns"]) / 1e6,
                abs_tol=1e-9
            )
            values.append(value)

        assert math.isclose(
            statistics.median(values),
            float(row["median_ms"]),
            abs_tol=1e-9
        )

    grouped = defaultdict(list)

    for row in all_summary:
        key = (
            int(row["threads"]),
            int(row["round"]),
            int(row["image_index"]),
        )
        grouped[key].append(row)

    assert len(grouped) == 128
    assert all(
        len(rows) == 2 for rows in grouped.values()
    )

    pairs = []

    for key, rows in sorted(grouped.items()):
        threads, round_no, image_index = key

        full = next(
            r for r in rows
            if r["mapping"] == "FULL_MPS"
        )
        hybrid = next(
            r for r in rows
            if r["mapping"] == "MPS_TO_CPU"
        )

        assert {
            int(full["placement_order"]),
            int(hybrid["placement_order"])
        } == {1, 2}

        full_ms = float(full["median_ms"])
        hybrid_ms = float(hybrid["median_ms"])

        pairs.append({
            "threads": threads,
            "round": round_no,
            "image_index": image_index,
            "image_path": full["image_path"],
            "image_sha256": full["image_sha256"],
            "full_mps_ms": full_ms,
            "hybrid_ms": hybrid_ms,
            "delta_percent": 100 * (
                hybrid_ms / full_ms - 1
            ),
            "hybrid_faster": int(
                hybrid_ms < full_ms
            ),
            "full_mps_order": int(
                full["placement_order"]
            ),
        })

    assert len(pairs) == 128

    print("\n=== T12 HOLDOUT COMPARISON ===")

    for threads in THREADS:
        selected = [
            r for r in pairs
            if r["threads"] == threads
        ]
        assert len(selected) == 64

        print(f"\nCPU THREADS={threads}")
        print(
            f"  Hybrid faster in "
            f"{sum(r['hybrid_faster'] for r in selected)}/64 "
            "image-process pairs"
        )

        process_deltas = []

        for round_no in range(1, 5):
            one_round = [
                r for r in selected
                if r["round"] == round_no
            ]
            assert len(one_round) == 16
            delta = statistics.median(
                r["delta_percent"]
                for r in one_round
            )
            process_deltas.append(delta)

            print(
                f"  Round {round_no} "
                f"median paired delta: {delta:+.3f}%"
            )

        print(
            "  Faster process rounds: "
            f"{sum(d < 0 for d in process_deltas)}/4"
        )
        print(
            "  Median of process deltas: "
            f"{statistics.median(process_deltas):+.3f}%"
        )

        image_medians = []

        for image_index in range(IMAGE_COUNT):
            image_rows = [
                r for r in selected
                if r["image_index"] == image_index
            ]
            assert len(image_rows) == 4

            orders = [
                r["full_mps_order"]
                for r in image_rows
            ]
            assert sorted(orders) == [1, 1, 2, 2]

            image_medians.append(
                statistics.median(
                    r["delta_percent"]
                    for r in image_rows
                )
            )

        print(
            "  Images with negative four-round "
            f"median delta: "
            f"{sum(d < 0 for d in image_medians)}/16"
        )

    raw_path = ROOT / "all_raw.csv"
    summary_path = ROOT / "all_summary.csv"
    paired_path = ROOT / "paired_images.csv"
    image_path = ROOT / "holdout_images.json"
    metadata_path = ROOT / "campaign_metadata.json"

    writecsv(raw_path, RAW_FIELDS, all_raw)
    writecsv(
        summary_path, SUMMARY_FIELDS, all_summary
    )
    writecsv(
        paired_path, list(pairs[0]), pairs
    )

    image_path.write_text(
        json.dumps(images, indent=2)
    )

    metadata_path.write_text(
        json.dumps({
            "protocol": "T12_holdout_confirmation_v1",
            "model": "convnext_large",
            "thread_counts": THREADS,
            "rounds_per_thread": ROUNDS,
            "independent_processes": 8,
            "holdout_images": IMAGE_COUNT,
            "holdout_selection_seed": SEED,
            "excluded_previous_validation_images": 16,
            "configuration_runs": 256,
            "timed_samples": 3072,
            "paired_comparisons": 128,
            "warmups": WARMUPS,
            "timed_calls": TIMED,
            "placement_order": (
                "Each image has Full MPS first twice "
                "and hybrid first twice per thread"
            ),
            "script_sha256": sha(__file__),
            "validation_sha256": sha(VALIDATION),
            "inventory_sha256": sha(INVENTORY),
            "manifest_sha256": sha(MANIFEST),
            "interpretation": (
                "Confirmation of a T11-selected candidate. "
                "Process rounds and images are repeated "
                "comparison levels; raw inference calls "
                "are not independent replicates."
            ),
            "limitations": [
                "One model and two thread settings",
                "Sixteen images from one dataset",
                "Same device across all rounds",
                "No controlled thermal conditions",
                "No energy or memory measurements",
                "No pipeline overlap",
            ],
        }, indent=2)
    )

    write_checksums(
        ROOT, [
            Path(__file__),
            raw_path,
            summary_path,
            paired_path,
            image_path,
            metadata_path,
        ]
    )

    print("\n==============================")
    print("T12 HOLDOUT CAMPAIGN: PASS")
    print("INDEPENDENT PROCESSES: 8")
    print("HOLDOUT IMAGES: 16")
    print("CONFIGURATION RUNS: 256")
    print("TIMED SAMPLES: 3072")
    print("PAIRED COMPARISONS: 128")
    print("RESULT DIRECTORY:", ROOT)
    print("==============================")

def main():
    assert ROOT.is_dir()

    if len(sys.argv) == 6 and sys.argv[2] == "--worker":
        worker(
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
        )
        return

    images = select_holdout()

    print(
        f"HOLDOUT SELECTION: {len(images)} "
        "new images; prior 16 excluded — PASS",
        flush=True
    )

    process_number = 0

    for round_no in range(1, ROUNDS + 1):
        thread_order = (
            [1, 4] if round_no % 2
            else [4, 1]
        )

        for threads in thread_order:
            process_number += 1

            print(
                f"\n=== PROCESS {process_number}/8 "
                f"ROUND={round_no} THREADS={threads} ===",
                flush=True
            )

            subprocess.run([
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                str(ROOT),
                "--worker",
                str(process_number),
                str(round_no),
                str(threads),
            ], check=True)

    aggregate()

if __name__ == "__main__":
    main()
