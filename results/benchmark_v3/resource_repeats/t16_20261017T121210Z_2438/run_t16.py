import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASE = Path("results/benchmark_v3")

VAL = BASE / (
    "validation/t05_20261015T173344Z_2438/metadata.json"
)
INV = BASE / (
    "inventory/inv_20261015T175122Z_2438/"
    "structural_inventory.csv"
)
HOLDOUT = BASE / (
    "holdout_thread_confirmation/"
    "t12_20261015T192923Z_2438/holdout_images.json"
)
T15 = BASE / (
    "resource_pilot/t15_20261017T120041Z_2438"
)
MANIFEST = Path(
    "configs/partition_manifests_v2/"
    "convnext_large__fine41.json"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/"
    "imagenetv2-matched-frequency-format-val"
)

WARMUPS = 3
TIMED = 12

CONFIGS = [
    (1, "FULL_MPS"),
    (1, "MPS_TO_CPU"),
    (4, "FULL_MPS"),
    (4, "MPS_TO_CPU"),
]

CHECKPOINTS = [
    "before_model_load",
    "cpu_model_and_reference_ready",
    "after_weight_placement",
    "after_warmups",
    "after_timed_calls",
]

RAW_FIELDS = [
    "process_number", "round", "order",
    "process_id", "run_id", "threads",
    "mapping", "iteration",
    "elapsed_ns", "elapsed_ms",
]

SUMMARY_FIELDS = [
    "process_number", "round", "order",
    "process_id", "run_id", "threads",
    "mapping", "median_ms", "mean_ms",
    "min_ms", "max_ms",
    "numerical_pass", "top1_agreement",
    "max_abs_error", "warmups",
    "repetitions", "image_sha256",
]

MEMORY_FIELDS = [
    "process_number", "round", "order",
    "process_id", "threads", "mapping",
    "checkpoint", "rss_bytes",
    "mps_allocated_bytes",
    "mps_driver_bytes",
    "mps_recommended_bytes",
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def readcsv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def writecsv(path, fields, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

def savejson(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2)
        + "\n"
    )

def checksum(directory, files):
    savejson(
        directory / "checksums.json",
        {
            file.name: sha(file)
            for file in files
        }
    )

def verify(directory):
    checks = json.loads(
        (directory / "checksums.json").read_text()
    )
    assert checks
    for name, expected in checks.items():
        file = directory / name
        assert file.is_file(), str(file)
        assert sha(file) == expected, str(file)

def schedule(round_no):
    shift = round_no - 1
    return CONFIGS[shift:] + CONFIGS[:shift]

def rss_bytes():
    text = subprocess.check_output(
        [
            "ps", "-o", "rss=",
            "-p", str(os.getpid())
        ],
        text=True
    ).strip()
    return int(text) * 1024

def source_check():
    validation = json.loads(VAL.read_text())
    holdout = json.loads(HOLDOUT.read_text())
    manifest = json.loads(MANIFEST.read_text())

    assert len(holdout) == 16
    assert manifest["group_count"] == 41

    previous = {
        row["sha256"]
        for row in validation["images"]
    }

    image_info = holdout[0]
    assert image_info["sha256"] not in previous

    image_path = IMAGE_ROOT / image_info["path"]
    assert image_path.is_file()
    assert sha(image_path) == image_info["sha256"]

    inv = [
        row for row in readcsv(INV)
        if row["model"] == "convnext_large"
    ]
    inv.sort(
        key=lambda row: int(
            row["stage_id"][1:]
        )
    )

    assert len(inv) == 6
    assert all(
        row["manifest_sha256"] == sha(MANIFEST)
        for row in inv
    )

    return validation, image_info, image_path

def worker(number, round_no, order, threads, mapping):
    import platform
    import time
    from urllib.parse import urlparse

    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torchvision.models import (
        convnext_large,
        ConvNeXt_Large_Weights,
    )

    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()

    torch.set_num_threads(threads)
    assert torch.get_num_threads() == threads
    torch.manual_seed(2026)

    directory = ROOT / f"process_{number:02d}"
    directory.mkdir(
        parents=True, exist_ok=False
    )

    process_id = uuid.uuid4().hex
    run_id = uuid.uuid4().hex

    validation, image_info, image_path = (
        source_check()
    )

    weights = (
        ConvNeXt_Large_Weights.IMAGENET1K_V1
    )

    from_checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    weight_sha = validation[
        "models"
    ]["convnext_large"]["weight_sha256"]

    assert from_checkpoint.is_file()
    assert sha(from_checkpoint) == weight_sha

    memory = []

    def mps_value(name):
        function = getattr(
            torch.mps, name, None
        )

        if not callable(function):
            return None

        try:
            return int(function())
        except (
            RuntimeError,
            NotImplementedError
        ):
            return None

    def snapshot(checkpoint):
        torch.mps.synchronize()

        memory.append({
            "process_number": number,
            "round": round_no,
            "order": order,
            "process_id": process_id,
            "threads": threads,
            "mapping": mapping,
            "checkpoint": checkpoint,
            "rss_bytes": rss_bytes(),
            "mps_allocated_bytes":
                mps_value(
                    "current_allocated_memory"
                ),
            "mps_driver_bytes":
                mps_value(
                    "driver_allocated_memory"
                ),
            "mps_recommended_bytes":
                mps_value(
                    "recommended_max_memory"
                ),
        })

    torch.mps.empty_cache()
    snapshot("before_model_load")

    model = convnext_large(
        weights=None
    ).eval().cpu()

    state = torch.load(
        from_checkpoint,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(
        state, strict=True
    )
    del state

    stages = [
        model.features[0],
        model.features[1],
        nn.Sequential(
            model.features[2],
            model.features[3]
        ),
        nn.Sequential(
            model.features[4],
            model.features[5]
        ),
        nn.Sequential(
            model.features[6],
            model.features[7]
        ),
        nn.Sequential(
            model.avgpool,
            model.classifier
        ),
    ]

    with Image.open(image_path) as image:
        x_cpu = weights.transforms()(
            image.convert("RGB")
        ).unsqueeze(0)

    assert tuple(x_cpu.shape) == (
        1, 3, 224, 224
    )

    with torch.inference_mode():
        reference = model(
            x_cpu
        ).detach().clone()

    assert tuple(reference.shape) == (
        1, 1000
    )

    snapshot("cpu_model_and_reference_ready")

    devices = (
        ["mps"] * 6
        if mapping == "FULL_MPS"
        else ["mps"] * 5 + ["cpu"]
    )

    for stage, device in zip(
        stages, devices
    ):
        stage.to(device)

    snapshot("after_weight_placement")

    def forward():
        y = x_cpu

        for stage, device in zip(
            stages, devices
        ):
            if y.device.type != device:
                y = y.to(device)
            y = stage(y)

        if y.device.type != "cpu":
            y = y.cpu()

        torch.mps.synchronize()
        return y

    with torch.inference_mode():
        actual = forward()

    assert actual.shape == reference.shape
    assert bool(torch.isfinite(actual).all())

    numerical_ok = bool(
        torch.allclose(
            actual, reference,
            atol=1e-3,
            rtol=1e-3
        )
    )
    top1_ok = (
        actual.argmax().item()
        == reference.argmax().item()
    )

    assert numerical_ok
    assert top1_ok

    max_error = float(
        (actual - reference)
        .abs()
        .max()
        .item()
    )
    del actual

    with torch.inference_mode():
        for _ in range(WARMUPS):
            output = forward()
            del output

    snapshot("after_warmups")

    raw = []
    timings = []

    with torch.inference_mode():
        for iteration in range(
            1, TIMED + 1
        ):
            torch.mps.synchronize()

            start = time.perf_counter_ns()
            output = forward()
            elapsed_ns = (
                time.perf_counter_ns()
                - start
            )
            del output

            assert elapsed_ns > 0

            elapsed_ms = elapsed_ns / 1e6
            assert math.isfinite(elapsed_ms)

            timings.append(elapsed_ms)

            raw.append({
                "process_number": number,
                "round": round_no,
                "order": order,
                "process_id": process_id,
                "run_id": run_id,
                "threads": threads,
                "mapping": mapping,
                "iteration": iteration,
                "elapsed_ns": elapsed_ns,
                "elapsed_ms": elapsed_ms,
            })

    snapshot("after_timed_calls")

    summary = [{
        "process_number": number,
        "round": round_no,
        "order": order,
        "process_id": process_id,
        "run_id": run_id,
        "threads": threads,
        "mapping": mapping,
        "median_ms": statistics.median(
            timings
        ),
        "mean_ms": statistics.mean(
            timings
        ),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "numerical_pass": int(
            numerical_ok
        ),
        "top1_agreement": int(
            top1_ok
        ),
        "max_abs_error": max_error,
        "warmups": WARMUPS,
        "repetitions": TIMED,
        "image_sha256":
            image_info["sha256"],
    }]

    assert len(raw) == TIMED
    assert len(memory) == 5

    raw_path = directory / "raw.csv"
    sum_path = directory / "summary.csv"
    mem_path = directory / "memory.csv"
    meta_path = directory / "metadata.json"

    writecsv(
        raw_path, RAW_FIELDS, raw
    )
    writecsv(
        sum_path, SUMMARY_FIELDS, summary
    )
    writecsv(
        mem_path, MEMORY_FIELDS, memory
    )

    savejson(meta_path, {
        "protocol":
            "T16_repeated_sampled_memory_v1",
        "process_number": number,
        "round": round_no,
        "execution_order": order,
        "process_id": process_id,
        "threads": threads,
        "mapping": mapping,
        "script_sha256": sha(__file__),
        "validation_sha256": sha(VAL),
        "inventory_sha256": sha(INV),
        "manifest_sha256": sha(MANIFEST),
        "holdout_sha256": sha(HOLDOUT),
        "weight_sha256": weight_sha,
        "image": image_info,
        "checkpoints": CHECKPOINTS,
        "warmups": WARMUPS,
        "timed_iterations": TIMED,
        "torch": torch.__version__,
        "torchvision":
            torchvision.__version__,
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "mps_fallback": os.environ[
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ],
        "memory_type": (
            "Five snapshots per fresh process; "
            "not true peak usage"
        ),
        "rss_unit": "bytes",
        "mps_counter_unit": "bytes",
        "rss_mps_additive": False,
        "energy_measured": False,
        "input_device": "cpu",
        "output_device": "cpu",
        "preprocessing_timed": False,
        "weight_placement_timed": False,
        "transfer_timed": True,
    })

    checksum(
        directory,
        [
            raw_path,
            sum_path,
            mem_path,
            meta_path,
        ]
    )

    last = memory[-1]

    def to_mib(value):
        if value is None:
            return "N/A"
        return f"{value/1048576:.1f}"

    print(
        f"PROCESS {number}/16: PASS | "
        f"{threads} threads | {mapping} | "
        f"latency={summary[0]['median_ms']:.3f} ms | "
        f"RSS={to_mib(last['rss_bytes'])} MiB | "
        "MPS allocated="
        f"{to_mib(last['mps_allocated_bytes'])} MiB",
        flush=True
    )

def aggregate():
    combined_raw = []
    combined_summary = []
    combined_memory = []
    process_ids = set()

    for number in range(1, 17):
        directory = ROOT / f"process_{number:02d}"
        verify(directory)

        meta = json.loads(
            (directory / "metadata.json").read_text()
        )

        round_no = (
            (number - 1) // 4 + 1
        )
        order = (
            (number - 1) % 4 + 1
        )
        threads, mapping = (
            schedule(round_no)[order-1]
        )

        assert meta["process_number"] == number
        assert meta["round"] == round_no
        assert meta["execution_order"] == order
        assert meta["threads"] == threads
        assert meta["mapping"] == mapping

        assert meta["process_id"] not in (
            process_ids
        )
        process_ids.add(meta["process_id"])

        assert meta["script_sha256"] == (
            sha(__file__)
        )
        assert meta["validation_sha256"] == (
            sha(VAL)
        )
        assert meta["inventory_sha256"] == (
            sha(INV)
        )
        assert meta["manifest_sha256"] == (
            sha(MANIFEST)
        )
        assert meta["holdout_sha256"] == (
            sha(HOLDOUT)
        )

        raw = readcsv(
            directory / "raw.csv"
        )
        summary = readcsv(
            directory / "summary.csv"
        )
        memory = readcsv(
            directory / "memory.csv"
        )

        assert len(raw) == 12
        assert len(summary) == 1
        assert len(memory) == 5

        assert [
            r["checkpoint"]
            for r in memory
        ] == CHECKPOINTS

        row = summary[0]
        assert int(
            row["numerical_pass"]
        ) == 1
        assert int(
            row["top1_agreement"]
        ) == 1

        assert all(
            r["process_id"]
            == meta["process_id"]
            for r in raw + memory
        )

        values = []

        for iteration, r in enumerate(
            raw, start=1
        ):
            assert int(
                r["iteration"]
            ) == iteration

            elapsed_ns = int(
                r["elapsed_ns"]
            )
            elapsed_ms = float(
                r["elapsed_ms"]
            )
            assert elapsed_ns > 0
            assert math.isclose(
                elapsed_ms,
                elapsed_ns / 1e6,
                abs_tol=1e-9
            )
            values.append(
                elapsed_ms
            )

        assert math.isclose(
            statistics.median(values),
            float(row["median_ms"]),
            abs_tol=1e-9
        )

        for r in memory:
            assert int(
                r["rss_bytes"]
            ) > 0

            for field in (
                "mps_allocated_bytes",
                "mps_driver_bytes",
                "mps_recommended_bytes",
            ):
                if r[field] != "":
                    assert int(
                        r[field]
                    ) >= 0

        combined_raw.extend(raw)
        combined_summary.extend(summary)
        combined_memory.extend(memory)

    assert len(process_ids) == 16
    assert len(combined_raw) == 192
    assert len(combined_summary) == 16
    assert len(combined_memory) == 80

    config_counts = Counter(
        (
            int(r["threads"]),
            r["mapping"]
        )
        for r in combined_summary
    )
    assert len(config_counts) == 4
    assert set(
        config_counts.values()
    ) == {4}

    for config in CONFIGS:
        selected = [
            r for r in combined_summary
            if (
                int(r["threads"]),
                r["mapping"]
            ) == config
        ]

        assert sorted(
            int(r["order"])
            for r in selected
        ) == [1, 2, 3, 4]

    raw_path = ROOT / "all_raw.csv"
    sum_path = ROOT / "all_summary.csv"
    mem_path = ROOT / "all_memory.csv"
    pair_path = ROOT / "paired_memory.csv"
    meta_path = ROOT / "campaign_metadata.json"

    writecsv(
        raw_path,
        RAW_FIELDS,
        combined_raw
    )
    writecsv(
        sum_path,
        SUMMARY_FIELDS,
        combined_summary
    )
    writecsv(
        mem_path,
        MEMORY_FIELDS,
        combined_memory
    )

    snapshots = {}

    for row in combined_memory:
        key = (
            int(row["round"]),
            int(row["threads"]),
            row["mapping"],
            row["checkpoint"],
        )
        assert key not in snapshots
        snapshots[key] = row

    pair_rows = []

    for round_no in range(1, 5):
        for threads in (1, 4):
            def get(mapping, checkpoint):
                return snapshots[(
                    round_no,
                    threads,
                    mapping,
                    checkpoint,
                )]

            full_before = get(
                "FULL_MPS",
                "before_model_load"
            )
            hybrid_before = get(
                "MPS_TO_CPU",
                "before_model_load"
            )
            full_after = get(
                "FULL_MPS",
                "after_timed_calls"
            )
            hybrid_after = get(
                "MPS_TO_CPU",
                "after_timed_calls"
            )

            def memory_delta(field):
                a = full_after[field]
                b = hybrid_after[field]
                if a == "" or b == "":
                    return None
                return (
                    int(b) - int(a)
                ) / 1048576

            full_rss_growth = (
                int(full_after["rss_bytes"])
                - int(full_before["rss_bytes"])
            )
            hybrid_rss_growth = (
                int(hybrid_after["rss_bytes"])
                - int(hybrid_before["rss_bytes"])
            )

            full_lat = next(
                float(r["median_ms"])
                for r in combined_summary
                if int(r["round"]) == round_no
                and int(r["threads"]) == threads
                and r["mapping"] == "FULL_MPS"
            )
            hybrid_lat = next(
                float(r["median_ms"])
                for r in combined_summary
                if int(r["round"]) == round_no
                and int(r["threads"]) == threads
                and r["mapping"] == "MPS_TO_CPU"
            )

            pair_rows.append({
                "round": round_no,
                "threads": threads,
                "full_mps_ms": full_lat,
                "hybrid_ms": hybrid_lat,
                "latency_delta_percent":
                    100 * (
                        hybrid_lat / full_lat - 1
                    ),
                "rss_after_delta_mib":
                    memory_delta(
                        "rss_bytes"
                    ),
                "rss_growth_delta_mib":
                    (
                        hybrid_rss_growth
                        - full_rss_growth
                    ) / 1048576,
                "mps_allocated_delta_mib":
                    memory_delta(
                        "mps_allocated_bytes"
                    ),
                "mps_driver_delta_mib":
                    memory_delta(
                        "mps_driver_bytes"
                    ),
                "full_rss_growth_mib":
                    full_rss_growth / 1048576,
                "hybrid_rss_growth_mib":
                    hybrid_rss_growth / 1048576,
            })

    assert len(pair_rows) == 8
    writecsv(
        pair_path,
        list(pair_rows[0]),
        pair_rows
    )

    print(
        "\n=== T16 REPEATED MEMORY COMPARISON ==="
    )
    print(
        "Deltas: Hybrid minus Full MPS; "
        "negative means less reported memory."
    )

    for threads in (1, 4):
        rows = [
            r for r in pair_rows
            if r["threads"] == threads
        ]

        def med(field):
            values = [
                float(r[field])
                for r in rows
                if r[field] not in ("", None)
            ]
            return (
                statistics.median(values)
                if len(values) == 4
                else None
            )

        print(f"\nCPU THREADS={threads}")

        for label, field in (
            ("RSS after-timed delta",
             "rss_after_delta_mib"),
            ("RSS growth delta",
             "rss_growth_delta_mib"),
            ("MPS allocated delta",
             "mps_allocated_delta_mib"),
            ("MPS driver delta",
             "mps_driver_delta_mib"),
        ):
            result = med(field)

            if result is None:
                print(f"  {label}: N/A")
            else:
                values = [
                    float(r[field])
                    for r in rows
                ]
                print(
                    f"  {label}: "
                    f"median {result:+.2f} MiB | "
                    f"range {min(values):+.2f} "
                    f"to {max(values):+.2f} MiB"
                )

        deltas = [
            r["latency_delta_percent"]
            for r in rows
        ]

        print(
            "  Latency median paired delta: "
            f"{statistics.median(deltas):+.3f}%"
        )
        print(
            "  Hybrid faster: "
            f"{sum(x < 0 for x in deltas)}/4"
        )

    savejson(
        meta_path,
        {
            "protocol":
                "T16_repeated_sampled_memory_v1",
            "model": "convnext_large",
            "independent_processes": 16,
            "rounds": 4,
            "configuration_runs": 16,
            "timed_samples": 192,
            "memory_snapshots": 80,
            "paired_memory_comparisons": 8,
            "schedule": [
                {
                    "round": round_no,
                    "ordered_configs":
                        schedule(round_no),
                }
                for round_no in range(1, 5)
            ],
            "script_sha256": sha(__file__),
            "validation_sha256": sha(VAL),
            "inventory_sha256": sha(INV),
            "manifest_sha256": sha(MANIFEST),
            "holdout_sha256": sha(HOLDOUT),
            "rss_is_true_peak": False,
            "mps_is_true_peak": False,
            "rss_mps_additive": False,
            "physical_memory_measured": False,
            "energy_measured": False,
            "memory_budget_enforced": False,
            "notes": [
                "Each configuration has four fresh processes.",
                "Each configuration occupies every order position.",
                "Round pairs compare different processes.",
                "RSS growth subtracts the before-model-load RSS.",
                "The growth adjustment does not remove all confounding.",
                "Memory counters are checkpoints, not peak usage.",
                "No actual scheduler is executed.",
            ],
        }
    )

    checksum(
        ROOT,
        [
            Path(__file__),
            raw_path,
            sum_path,
            mem_path,
            pair_path,
            meta_path,
        ]
    )
    verify(ROOT)

    print("\n==============================")
    print("T16 REPEATED RESOURCE PILOT: PASS")
    print("INDEPENDENT PROCESSES: 16")
    print("CONFIGURATION RUNS: 16")
    print("TIMED SAMPLES: 192")
    print("MEMORY SNAPSHOTS: 80")
    print("PAIRED MEMORY COMPARISONS: 8")
    print("TRUE PEAK MEMORY MEASURED: NO")
    print("ENERGY MEASURED: NO")
    print("RESULT DIRECTORY:", ROOT)
    print("==============================")

def main():
    assert ROOT.is_dir()

    if len(sys.argv) == 8 and sys.argv[2] == "--worker":
        worker(
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
            int(sys.argv[6]),
            sys.argv[7],
        )
        return

    verify(T15)
    print(
        "T15 SOURCE CHECKSUMS: PASS",
        flush=True
    )
    source_check()

    number = 0

    for round_no in range(1, 5):
        print(
            f"\n=== ROUND {round_no}/4 ===",
            flush=True
        )

        for order, (
            threads, mapping
        ) in enumerate(
            schedule(round_no), 1
        ):
            number += 1
            print(
                f"\nPROCESS {number}/16 | "
                f"threads={threads} | "
                f"{mapping}",
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
                    str(round_no),
                    str(order),
                    str(threads),
                    mapping,
                ],
                check=True,
            )

    aggregate()

if __name__ == "__main__":
    main()
