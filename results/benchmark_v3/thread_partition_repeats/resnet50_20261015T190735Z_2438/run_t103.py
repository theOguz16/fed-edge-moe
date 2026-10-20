import csv
import hashlib
import json
import math
import os
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
    "configs/partition_manifests_v2/resnet50__stage6.json"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/imagenetv2-matched-frequency-format-val"
)

THREADS = [2, 4, 6, 10]
ROUNDS = 4
WARMUPS = 3
TIMED = 12

RAW_FIELDS = [
    "process_number", "round_number", "thread_count",
    "process_id", "run_id", "config_id", "mapping",
    "cut_after", "placement_order", "iteration",
    "elapsed_ns", "elapsed_ms",
]

SUMMARY_FIELDS = [
    "process_number", "round_number", "thread_count",
    "process_id", "run_id", "config_id", "mapping",
    "cut_after", "placement_order", "device_sequence",
    "transfer_events", "logical_transfer_bytes",
    "warmups", "repetitions", "numerical_pass",
    "top1_agreement", "max_abs_error",
    "median_ms", "mean_ms", "min_ms", "max_ms",
]

LAYOUTS = [
    ("FULL_CPU", None, ["cpu"] * 6),
    ("FULL_MPS", None, ["mps"] * 6),
    ("CPU_TO_MPS", 1, ["cpu"] + ["mps"] * 5),
    ("MPS_TO_CPU", 5, ["mps"] * 5 + ["cpu"]),
]

EXPECTED_GROUPS = [
    ["conv1", "bn1", "relu", "maxpool"],
    ["layer1"], ["layer2"], ["layer3"], ["layer4"],
    ["avgpool", "fc"],
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def digest(obj):
    encoded = json.dumps(
        obj, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

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
            {p.name: sha(p) for p in paths}, indent=2
        )
    )

def verify_checksums(directory):
    checks = json.loads(
        (directory / "checksums.json").read_text()
    )
    for name, expected in checks.items():
        assert sha(directory / name) == expected, (
            f"Checksum failure: {directory / name}"
        )

def worker(process_number, round_number, threads):
    import platform
    import time
    from urllib.parse import urlparse

    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torchvision.models import (
        resnet50, ResNet50_Weights,
    )

    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()

    # Each process receives one fixed thread setting.
    torch.set_num_threads(threads)
    assert torch.get_num_threads() == threads
    torch.manual_seed(2026)

    directory = ROOT / f"process_{process_number:02d}"
    directory.mkdir(parents=True, exist_ok=False)

    process_id = uuid.uuid4().hex

    validation = json.loads(VALIDATION.read_text())
    manifest = json.loads(MANIFEST.read_text())

    assert manifest["group_count"] == 6
    assert [
        g["modules"] for g in manifest["groups"]
    ] == EXPECTED_GROUPS

    inventory = sorted(
        [
            r for r in read_csv(INVENTORY)
            if r["model"] == "resnet50"
        ],
        key=lambda r: int(r["stage_id"][1:])
    )

    assert len(inventory) == 6
    assert all(
        r["manifest_sha256"] == sha(MANIFEST)
        for r in inventory
    )

    image_info = validation["images"][0]
    image_path = IMAGE_ROOT / image_info["path"]
    assert sha(image_path) == image_info["sha256"]

    weights = ResNet50_Weights.IMAGENET1K_V2
    checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    weight_sha = validation[
        "models"
    ]["resnet50"]["weight_sha256"]

    assert checkpoint.is_file()
    assert sha(checkpoint) == weight_sha

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

    with Image.open(image_path) as image:
        x_cpu = weights.transforms()(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tuple(x_cpu.shape) == (1, 3, 224, 224)

    with torch.inference_mode():
        reference = model(x_cpu).detach().clone()

    assert tuple(reference.shape) == (1, 1000)
    assert bool(torch.isfinite(reference).all())

    def transfer_events(devices):
        events = []

        if devices[0] == "mps":
            events.append({
                "at": "input",
                "from": "cpu",
                "to": "mps",
                "logical_bytes":
                    x_cpu.numel() * x_cpu.element_size(),
            })

        for i in range(1, 6):
            if devices[i] != devices[i-1]:
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
                "logical_bytes":
                    reference.numel() * reference.element_size(),
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

    # Each of the four placements appears in each order
    # exactly once across the four process rounds.
    shift = round_number - 1
    layouts = LAYOUTS[shift:] + LAYOUTS[:shift]

    raw_rows = []
    summary_rows = []

    for placement_order, (
        mapping, cut, devices
    ) in enumerate(layouts, 1):

        # Model placement excluded from timed inference.
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
            f"mapping={mapping}"
        )
        assert top1_agreement

        config = {
            "model": "resnet50",
            "threads": threads,
            "mapping": mapping,
            "cut_after": cut,
            "devices": devices,
            "weight_sha256": weight_sha,
            "manifest_sha256": sha(MANIFEST),
            "image_sha256": image_info["sha256"],
            "torch": torch.__version__,
            "precision": "float32",
            "batch_size": 1,
            "cpu_input_output": True,
            "transfers_timed": True,
        }

        config_id = "cfg_" + digest(config)
        run_id = "run_" + uuid.uuid4().hex
        samples = []

        with torch.inference_mode():
            for _ in range(WARMUPS):
                result = forward(devices)
                del result

            for iteration in range(1, TIMED + 1):
                if "mps" in devices:
                    torch.mps.synchronize()

                start_ns = time.perf_counter_ns()
                result = forward(devices)
                elapsed_ns = (
                    time.perf_counter_ns() - start_ns
                )
                del result

                assert elapsed_ns > 0
                elapsed_ms = elapsed_ns / 1e6
                assert math.isfinite(elapsed_ms)

                samples.append(elapsed_ms)

                raw_rows.append({
                    "process_number": process_number,
                    "round_number": round_number,
                    "thread_count": threads,
                    "process_id": process_id,
                    "run_id": run_id,
                    "config_id": config_id,
                    "mapping": mapping,
                    "cut_after": cut,
                    "placement_order": placement_order,
                    "iteration": iteration,
                    "elapsed_ns": elapsed_ns,
                    "elapsed_ms": elapsed_ms,
                })

        events = transfer_events(devices)
        assert len(events) == (
            0 if mapping == "FULL_CPU" else 2
        )

        median = statistics.median(samples)

        summary_rows.append({
            "process_number": process_number,
            "round_number": round_number,
            "thread_count": threads,
            "process_id": process_id,
            "run_id": run_id,
            "config_id": config_id,
            "mapping": mapping,
            "cut_after": cut,
            "placement_order": placement_order,
            "device_sequence": json.dumps(devices),
            "transfer_events": json.dumps(events),
            "logical_transfer_bytes": sum(
                e["logical_bytes"] for e in events
            ),
            "warmups": WARMUPS,
            "repetitions": TIMED,
            "numerical_pass": int(numerical_pass),
            "top1_agreement": int(top1_agreement),
            "max_abs_error": max_error,
            "median_ms": median,
            "mean_ms": statistics.mean(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        })

        print(
            f"  {mapping:10} cut={str(cut):4} "
            f"median={median:.3f} ms | PASS",
            flush=True
        )

    assert len(summary_rows) == 4
    assert len(raw_rows) == 48

    raw_path = directory / "raw.csv"
    summary_path = directory / "summary.csv"
    meta_path = directory / "metadata.json"

    write_csv(raw_path, RAW_FIELDS, raw_rows)
    write_csv(
        summary_path, SUMMARY_FIELDS, summary_rows
    )

    metadata = {
        "protocol": "T10_3_independent_thread_partition_v1",
        "process_number": process_number,
        "round_number": round_number,
        "process_id": process_id,
        "cpu_threads": threads,
        "observed_cpu_threads": torch.get_num_threads(),
        "placement_order": [
            {"mapping": m, "cut_after": c}
            for m, c, _ in layouts
        ],
        "validation_source": str(VALIDATION),
        "validation_sha256": sha(VALIDATION),
        "inventory_source": str(INVENTORY),
        "inventory_sha256": sha(INVENTORY),
        "manifest_source": str(MANIFEST),
        "manifest_sha256": sha(MANIFEST),
        "weight_sha256": weight_sha,
        "input_image": image_info,
        "script_sha256": sha(__file__),
        "warmups": WARMUPS,
        "timed_calls": TIMED,
        "batch_size": 1,
        "precision": "float32",
        "input_device": "cpu",
        "output_device": "cpu",
        "transfer_timed": True,
        "preprocessing_timed": False,
        "model_placement_timed": False,
        "mps_sync": "before and at end of inference",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": sys.version.split()[0],
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "mps_fallback": os.environ[
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ],
        "limitations": [
            "one fixed ImageNetV2 image",
            "four independent processes per thread count",
            "same machine; thermal drift not controlled",
            "only four selected thread counts",
            "only two selected hybrid mappings",
            "sequential execution, no pipeline overlap",
            "no energy or peak-memory measurement",
            "exploratory repeated pilot",
        ],
    }

    meta_path.write_text(
        json.dumps(metadata, indent=2)
    )

    write_checksums(
        directory, [raw_path, summary_path, meta_path]
    )

    print(
        f"PROCESS {process_number}: PASS",
        flush=True
    )

def aggregate():
    raw_all = []
    summary_all = []
    process_ids = set()
    process_order = []
    validation = json.loads(VALIDATION.read_text())

    for number in range(1, 17):
        directory = ROOT / f"process_{number:02}"
        verify_checksums(directory)

        metadata = json.loads(
            (directory / "metadata.json").read_text()
        )

        assert metadata["process_number"] == number
        assert metadata["script_sha256"] == sha(__file__)
        assert metadata["validation_sha256"] == sha(VALIDATION)
        assert metadata["inventory_sha256"] == sha(INVENTORY)
        assert metadata["manifest_sha256"] == sha(MANIFEST)
        assert metadata["weight_sha256"] == validation[
            "models"
        ]["resnet50"]["weight_sha256"]
        assert metadata["process_id"] not in process_ids

        process_ids.add(metadata["process_id"])
        process_order.append({
            "process_number": number,
            "round_number": metadata["round_number"],
            "threads": metadata["cpu_threads"],
        })

        raw = read_csv(directory / "raw.csv")
        summary = read_csv(directory / "summary.csv")

        assert len(raw) == 48
        assert len(summary) == 4

        raw_all.extend(raw)
        summary_all.extend(summary)

    assert len(process_ids) == 16
    assert len(summary_all) == 64
    assert len(raw_all) == 768

    assert len({
        r["run_id"] for r in summary_all
    }) == 64

    assert len({
        r["config_id"] for r in summary_all
    }) == 16

    counts = Counter(
        r["config_id"] for r in summary_all
    )
    assert set(counts.values()) == {4}

    by_run = defaultdict(list)
    for row in raw_all:
        by_run[row["run_id"]].append(row)

    assert len(by_run) == 64

    for row in summary_all:
        samples = by_run[row["run_id"]]
        assert len(samples) == TIMED

        assert sorted(
            int(s["iteration"]) for s in samples
        ) == list(range(1, TIMED + 1))

        assert int(row["numerical_pass"]) == 1
        assert int(row["top1_agreement"]) == 1

        values = []

        for sample in samples:
            assert sample["config_id"] == row["config_id"]
            assert sample["mapping"] == row["mapping"]
            assert sample["cut_after"] == row["cut_after"]
            assert sample["process_id"] == row["process_id"]
            assert sample["thread_count"] == row["thread_count"]

            elapsed_ns = int(sample["elapsed_ns"])
            elapsed_ms = float(sample["elapsed_ms"])

            assert elapsed_ns > 0
            assert math.isclose(
                elapsed_ms,
                elapsed_ns / 1e6,
                abs_tol=1e-9
            )
            values.append(elapsed_ms)

        assert math.isclose(
            statistics.median(values),
            float(row["median_ms"]),
            abs_tol=1e-9
        )

    paired = []

    for threads in THREADS:
        thread_rows = [
            r for r in summary_all
            if int(r["thread_count"]) == threads
        ]

        assert len(thread_rows) == 16

        for round_number in range(1, ROUNDS + 1):
            rows = [
                r for r in thread_rows
                if int(r["round_number"]) == round_number
            ]
            assert len(rows) == 4
            assert sorted(
                int(r["placement_order"]) for r in rows
            ) == [1, 2, 3, 4]

            full_mps = next(
                r for r in rows
                if r["mapping"] == "FULL_MPS"
            )
            baseline = float(full_mps["median_ms"])

            for mapping in ("CPU_TO_MPS", "MPS_TO_CPU"):
                candidate = next(
                    r for r in rows
                    if r["mapping"] == mapping
                )
                candidate_ms = float(candidate["median_ms"])

                paired.append({
                    "thread_count": threads,
                    "round_number": round_number,
                    "hybrid_mapping": mapping,
                    "cut_after": candidate["cut_after"],
                    "full_mps_ms": baseline,
                    "hybrid_ms": candidate_ms,
                    "delta_ms": candidate_ms - baseline,
                    "delta_percent":
                        100 * (candidate_ms / baseline - 1),
                    "hybrid_faster":
                        int(candidate_ms < baseline),
                })

        for mapping, _, _ in LAYOUTS:
            positions = sorted(
                int(r["placement_order"])
                for r in thread_rows
                if r["mapping"] == mapping
            )
            assert positions == [1, 2, 3, 4]

    assert len(paired) == 32

    raw_path = ROOT / "all_raw.csv"
    summary_path = ROOT / "all_summary.csv"
    paired_path = ROOT / "paired_comparison.csv"
    meta_path = ROOT / "campaign_metadata.json"

    write_csv(raw_path, RAW_FIELDS, raw_all)
    write_csv(
        summary_path, SUMMARY_FIELDS, summary_all
    )
    write_csv(
        paired_path, list(paired[0]), paired
    )

    print("\n=== T10.3 REPEATED THREAD COMPARISON ===")
    print(
        f"{'Threads':>7} {'CPU ms':>10} "
        f"{'MPS ms':>10} {'Cut5 ms':>10} "
        f"{'Cut5 wins':>11} {'Cut5 delta':>12}"
    )

    for threads in THREADS:
        rows = [
            r for r in summary_all
            if int(r["thread_count"]) == threads
        ]

        def med(mapping):
            return statistics.median(
                float(r["median_ms"])
                for r in rows
                if r["mapping"] == mapping
            )

        differences = [
            r for r in paired
            if r["thread_count"] == threads
            and r["hybrid_mapping"] == "MPS_TO_CPU"
        ]

        delta = statistics.median(
            r["delta_percent"] for r in differences
        )
        wins = sum(r["hybrid_faster"] for r in differences)

        print(
            f"{threads:>7} "
            f"{med('FULL_CPU'):>10.3f} "
            f"{med('FULL_MPS'):>10.3f} "
            f"{med('MPS_TO_CPU'):>10.3f} "
            f"{wins:>9}/4 "
            f"{delta:>+11.3f}%"
        )

    print("\n=== FULL CPU ACROSS INDEPENDENT PROCESSES ===")
    for threads in THREADS:
        values = sorted(
            (
                int(r["round_number"]),
                float(r["median_ms"])
            )
            for r in summary_all
            if int(r["thread_count"]) == threads
            and r["mapping"] == "FULL_CPU"
        )

        formatted = ", ".join(
            f"R{round_no}={ms:.3f}"
            for round_no, ms in values
        )

        print(f"Threads={threads:2}: {formatted} ms")

    campaign_metadata = {
        "protocol": "T10_3_independent_thread_partition_v1",
        "independent_processes": 16,
        "processes_per_thread": 4,
        "thread_counts": THREADS,
        "process_schedule": process_order,
        "model": "resnet50",
        "image_sha256": validation[
            "images"
        ][0]["sha256"],
        "configuration_runs": len(summary_all),
        "timed_samples": len(raw_all),
        "paired_hybrid_comparisons": len(paired),
        "warmups_per_configuration": WARMUPS,
        "timed_calls_per_configuration": TIMED,
        "order_balance": (
            "Each placement occupies each of four "
            "positions once per thread count"
        ),
        "script_sha256": sha(__file__),
        "validation_sha256": sha(VALIDATION),
        "inventory_sha256": sha(INVENTORY),
        "manifest_sha256": sha(MANIFEST),
        "interpretation": (
            "Per-process medians are comparison units; "
            "timed calls are not independent replicates"
        ),
    }

    meta_path.write_text(
        json.dumps(campaign_metadata, indent=2)
    )

    write_checksums(
        ROOT,
        [
            Path(__file__), raw_path,
            summary_path, paired_path, meta_path,
        ],
    )

    print("\n==============================")
    print("T10.3 REPEATED CAMPAIGN: PASS")
    print("INDEPENDENT PROCESSES: 16")
    print("CONFIGURATION RUNS: 64")
    print("TIMED SAMPLES: 768")
    print("PAIRED HYBRID COMPARISONS: 32")
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

    process_number = 0

    for round_number in range(1, ROUNDS + 1):
        shift = round_number - 1
        thread_order = THREADS[shift:] + THREADS[:shift]

        print(
            f"\n=== ROUND {round_number}/{ROUNDS} ===",
            flush=True
        )

        for threads in thread_order:
            process_number += 1

            print(
                f"\nPROCESS {process_number}/16 "
                f"THREADS={threads}",
                flush=True
            )

            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    str(ROOT),
                    "--worker",
                    str(process_number),
                    str(round_number),
                    str(threads),
                ],
                check=True,
            )

    aggregate()

if __name__ == "__main__":
    main()
