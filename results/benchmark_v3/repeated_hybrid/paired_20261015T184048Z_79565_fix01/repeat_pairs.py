import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path

REPEATS = 8
WARMUPS = 3
TIMED = 15

VALIDATION = Path(
    "results/benchmark_v3/validation/"
    "t05_20261015T173344Z_2438"
)
INVENTORY = Path(
    "results/benchmark_v3/inventory/"
    "inv_20261015T175122Z_2438/structural_inventory.csv"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/imagenetv2-matched-frequency-format-val"
)
MANIFEST_DIR = Path("configs/partition_manifests_v2")

RAW_FIELDS = [
    "process_repeat", "run_id", "config_id", "model",
    "mapping", "cut_after", "execution_order",
    "iteration", "elapsed_ns", "elapsed_ms",
]
SUMMARY_FIELDS = [
    "process_repeat", "run_id", "config_id", "model",
    "mapping", "cut_after", "execution_order",
    "device_sequence", "logical_transfer_bytes",
    "warmups", "repetitions", "numerical_pass",
    "top1_agreement", "max_abs_error",
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
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()

def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def verify_checksums(directory):
    checksums = json.loads(
        (directory / "checksums.json").read_text()
    )
    for name, expected in checksums.items():
        assert sha(directory / name) == expected, (
            f"SHA256 failed: {directory / name}"
        )

def worker(outdir, process_repeat):
    import gc
    import time
    from urllib.parse import urlparse

    import torch
    import torchvision
    from PIL import Image
    from torch import nn
    from torchvision.models import (
        convnext_base, convnext_large,
        ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
    )

    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    run_dir = outdir / f"process_{process_repeat:02}"
    run_dir.mkdir(parents=True, exist_ok=False)

    validation_meta_path = VALIDATION / "metadata.json"
    validation = json.loads(
        validation_meta_path.read_text()
    )

    image_info = validation["images"][0]
    image_path = IMAGE_ROOT / image_info["path"]
    assert sha(image_path) == image_info["sha256"]

    inventory = read_csv(INVENTORY)

    expected_manifest_paths = (
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

    # Alternate which model is tested first.
    if process_repeat % 2 == 0:
        specs.reverse()

    raw_rows = []
    summary_rows = []
    source_records = {}
    process_id = uuid.uuid4().hex

    for name, factory, weights, manifest_name, hybrid_mapping, cut in specs:
        print(
            f"\nPROCESS {process_repeat}/{REPEATS}: {name}",
            flush=True
        )

        manifest_path = MANIFEST_DIR / manifest_name
        manifest_sha = sha(manifest_path)
        manifest = json.loads(manifest_path.read_text())

        assert manifest["group_count"] == 41
        assert [
            group["modules"]
            for group in manifest["groups"]
        ] == expected_manifest_paths

        stages_inventory = sorted(
            [
                row for row in inventory
                if row["model"] == name
            ],
            key=lambda row: int(row["stage_id"][1:])
        )

        assert len(stages_inventory) == 6
        assert [
            row["source_groups"]
            for row in stages_inventory
        ] == expected_ranges
        assert all(
            row["manifest_sha256"] == manifest_sha
            for row in stages_inventory
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
            "weights": weights.name,
            "weights_sha256": expected_weight_sha,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
        }

        model = factory(weights=None).eval().cpu()
        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
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

        with Image.open(image_path) as image:
            x_cpu = weights.transforms()(
                image.convert("RGB")
            ).unsqueeze(0)

        assert list(x_cpu.shape) == [1, 3, 224, 224]
        assert x_cpu.dtype == torch.float32

        with torch.inference_mode():
            reference = model(x_cpu).detach().clone()

        assert tuple(reference.shape) == (1, 1000)
        assert bool(torch.isfinite(reference).all())

        full = (
            "FULL_MPS", None, ["mps"] * 6
        )
        hybrid = (
            hybrid_mapping,
            cut,
            (
                ["cpu"] + ["mps"] * 5
                if name == "convnext_base"
                else ["mps"] * 5 + ["cpu"]
            ),
        )

        # Four processes execute each placement first.
        # The two models receive opposite order patterns.
        is_base = name == "convnext_base"
        full_first = (
            process_repeat % 2 == 0
            if is_base
            else process_repeat % 2 != 0
        )
        layouts = (
            [full, hybrid]
            if full_first
            else [hybrid, full]
        )

        def forward(devices):
            y = x_cpu

            for stage, device in zip(stages, devices):
                if y.device.type != device:
                    y = y.to(device)
                y = stage(y)

            if y.device.type != "cpu":
                y = y.cpu()

            torch.mps.synchronize()
            return y

        for execution_order, (
            mapping, cut_after, devices
        ) in enumerate(layouts, 1):

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
                actual.argmax().item()
                == reference.argmax().item()
            )
            max_error = float(
                (reference - actual).abs().max().item()
            )

            assert numerical_pass
            assert top1_agreement

            input_bytes = x_cpu.numel() * x_cpu.element_size()
            output_bytes = reference.numel() * reference.element_size()

            if mapping == "FULL_MPS":
                transfer_bytes = input_bytes + output_bytes
            elif mapping == "CPU_TO_MPS":
                transfer_bytes = (
                    int(stages_inventory[0]["output_activation_bytes"])
                    + output_bytes
                )
            else:
                transfer_bytes = (
                    input_bytes
                    + int(stages_inventory[4]["output_activation_bytes"])
                )

            config = {
                "model": name,
                "mapping": mapping,
                "cut_after": cut_after,
                "devices": devices,
                "weight_sha256": expected_weight_sha,
                "manifest_sha256": manifest_sha,
                "image_sha256": image_info["sha256"],
                "cpu_threads": 4,
                "dtype": "float32",
                "batch": 1,
                "torch": torch.__version__,
                "cpu_input": True,
                "cpu_output": True,
            }

            config_id = "cfg_" + digest(config)
            run_id = "run_" + uuid.uuid4().hex
            samples = []

            with torch.inference_mode():
                for _ in range(WARMUPS):
                    result = forward(devices)
                    del result

                for iteration in range(1, TIMED + 1):
                    torch.mps.synchronize()

                    start_ns = time.perf_counter_ns()
                    result = forward(devices)
                    elapsed_ns = (
                        time.perf_counter_ns() - start_ns
                    )
                    del result

                    assert elapsed_ns > 0
                    samples.append(elapsed_ns / 1e6)

                    raw_rows.append({
                        "process_repeat": process_repeat,
                        "run_id": run_id,
                        "config_id": config_id,
                        "model": name,
                        "mapping": mapping,
                        "cut_after": cut_after,
                        "execution_order": execution_order,
                        "iteration": iteration,
                        "elapsed_ns": elapsed_ns,
                        "elapsed_ms": elapsed_ns / 1e6,
                    })

            median = statistics.median(samples)

            summary_rows.append({
                "process_repeat": process_repeat,
                "run_id": run_id,
                "config_id": config_id,
                "model": name,
                "mapping": mapping,
                "cut_after": cut_after,
                "execution_order": execution_order,
                "device_sequence": json.dumps(devices),
                "logical_transfer_bytes": transfer_bytes,
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
                f"  {mapping:10} cut={cut_after} "
                f"order={execution_order} "
                f"median={median:.3f} ms | PASS",
                flush=True
            )

        del (
            model, stages, x_cpu, reference,
            actual
        )
        gc.collect()
        torch.mps.empty_cache()

    assert len(summary_rows) == 4
    assert len(raw_rows) == 4 * TIMED

    raw_path = run_dir / "raw.csv"
    summary_path = run_dir / "summary.csv"
    meta_path = run_dir / "metadata.json"

    write_csv(raw_path, RAW_FIELDS, raw_rows)
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)

    metadata = {
        "protocol": "V3_paired_process_repeats_v1",
        "process_repeat": process_repeat,
        "process_id": process_id,
        "script_sha256": sha(__file__),
        "sources": source_records,
        "source_validation": str(
            VALIDATION / "metadata.json"
        ),
        "source_validation_sha256": sha(
            VALIDATION / "metadata.json"
        ),
        "source_inventory": str(INVENTORY),
        "source_inventory_sha256": sha(INVENTORY),
        "input_image": image_info,
        "warmups": WARMUPS,
        "timed_iterations": TIMED,
        "batch_size": 1,
        "precision": "float32",
        "cpu_threads": 4,
        "mps_fallback": os.environ[
            "PYTORCH_ENABLE_MPS_FALLBACK"
        ],
        "input_device": "cpu",
        "output_device": "cpu",
        "transfer_included_in_latency": True,
        "weight_placement_included": False,
        "preprocessing_included": False,
        "mps_sync": "before and at end of inference",
        "execution": "sequential, no pipeline overlap",
        "independent_process": True,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": sys.version.split()[0],
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "limitations": [
            "one shared input image across repetitions",
            "not independent data samples",
            "single-cut comparison only",
            "no energy or memory measurement",
            "thermal and background load not controlled",
            "exploratory repeated pilot",
        ],
    }

    meta_path.write_text(
        json.dumps(metadata, indent=2)
    )

    files = [raw_path, summary_path, meta_path]
    (run_dir / "checksums.json").write_text(
        json.dumps(
            {p.name: sha(p) for p in files},
            indent=2
        )
    )

    print(
        f"PROCESS {process_repeat}: PASS",
        flush=True
    )

def aggregate(outdir):
    all_raw = []
    all_summary = []
    process_ids = set()
    script_hash = sha(__file__)

    for number in range(1, REPEATS + 1):
        directory = outdir / f"process_{number:02}"
        verify_checksums(directory)

        metadata = json.loads(
            (directory / "metadata.json").read_text()
        )
        assert metadata["script_sha256"] == script_hash
        assert metadata["process_repeat"] == number
        assert metadata["process_id"] not in process_ids
        process_ids.add(metadata["process_id"])

        assert sha(metadata["source_validation"]) == (
            metadata["source_validation_sha256"]
        )
        assert sha(metadata["source_inventory"]) == (
            metadata["source_inventory_sha256"]
        )
        for source in metadata["sources"].values():
            assert sha(source["manifest"]) == (
                source["manifest_sha256"]
            )

        raw = read_csv(directory / "raw.csv")
        summary = read_csv(directory / "summary.csv")

        assert len(raw) == 4 * TIMED
        assert len(summary) == 4

        by_run = defaultdict(list)
        for row in raw:
            by_run[row["run_id"]].append(row)

        for row in summary:
            assert int(row["process_repeat"]) == number
            assert int(row["numerical_pass"]) == 1
            assert int(row["top1_agreement"]) == 1
            assert len(by_run[row["run_id"]]) == TIMED

            values = []
            for sample in by_run[row["run_id"]]:
                assert sample["config_id"] == row["config_id"]
                assert sample["model"] == row["model"]
                assert sample["mapping"] == row["mapping"]
                assert sample["cut_after"] == row["cut_after"]

                elapsed_ns = int(sample["elapsed_ns"])
                assert elapsed_ns > 0

                elapsed_ms = float(sample["elapsed_ms"])
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

        all_raw.extend(raw)
        all_summary.extend(summary)

    assert len(process_ids) == REPEATS
    assert len(all_summary) == REPEATS * 4
    assert len(all_raw) == REPEATS * 4 * TIMED
    assert len({
        r["run_id"] for r in all_summary
    }) == REPEATS * 4

    for model, hybrid in [
        ("convnext_base", "CPU_TO_MPS"),
        ("convnext_large", "MPS_TO_CPU"),
    ]:
        model_rows = [
            r for r in all_summary
            if r["model"] == model
        ]
        assert len(model_rows) == 2 * REPEATS

        full_first = sum(
            1 for r in model_rows
            if r["mapping"] == "FULL_MPS"
            and int(r["execution_order"]) == 1
        )
        assert full_first == REPEATS // 2

        hybrid_first = sum(
            1 for r in model_rows
            if r["mapping"] == hybrid
            and int(r["execution_order"]) == 1
        )
        assert hybrid_first == REPEATS // 2

    pairs = []
    for number in range(1, REPEATS + 1):
        for model, hybrid in [
            ("convnext_base", "CPU_TO_MPS"),
            ("convnext_large", "MPS_TO_CPU"),
        ]:
            rows = [
                r for r in all_summary
                if int(r["process_repeat"]) == number
                and r["model"] == model
            ]

            assert len(rows) == 2
            full = next(
                r for r in rows
                if r["mapping"] == "FULL_MPS"
            )
            candidate = next(
                r for r in rows
                if r["mapping"] == hybrid
            )

            full_ms = float(full["median_ms"])
            hybrid_ms = float(candidate["median_ms"])

            pairs.append({
                "process_repeat": number,
                "model": model,
                "full_mps_ms": full_ms,
                "hybrid_ms": hybrid_ms,
                "delta_ms": hybrid_ms - full_ms,
                "delta_percent": (
                    100 * (hybrid_ms / full_ms - 1)
                ),
                "hybrid_faster": int(
                    hybrid_ms < full_ms
                ),
                "full_mps_order": int(
                    full["execution_order"]
                ),
                "hybrid_order": int(
                    candidate["execution_order"]
                ),
            })

    raw_path = outdir / "all_raw.csv"
    summary_path = outdir / "all_summary.csv"
    pairs_path = outdir / "paired_comparison.csv"

    write_csv(raw_path, RAW_FIELDS, all_raw)
    write_csv(summary_path, SUMMARY_FIELDS, all_summary)
    write_csv(
        pairs_path, list(pairs[0].keys()), pairs
    )

    print("\n=== INDEPENDENT PROCESS COMPARISON ===")

    for model in ("convnext_base", "convnext_large"):
        rows = [
            r for r in pairs if r["model"] == model
        ]
        deltas = [r["delta_percent"] for r in rows]
        wins = sum(r["hybrid_faster"] for r in rows)

        print(f"\n{model}:")
        print(
            f"  Hybrid faster in {wins}/{REPEATS} "
            "independent processes"
        )
        print(
            "  Median paired latency difference: "
            f"{statistics.median(deltas):+.3f}%"
        )
        print(
            "  Paired difference range: "
            f"{min(deltas):+.3f}% to "
            f"{max(deltas):+.3f}%"
        )

    metadata = {
        "protocol": "V3_paired_process_repeats_v1",
        "independent_processes": REPEATS,
        "models": [
            "convnext_base", "convnext_large"
        ],
        "configs_per_process": 4,
        "warmups_per_config": WARMUPS,
        "timed_calls_per_config": TIMED,
        "configuration_count": REPEATS * 4,
        "total_timed_samples": len(all_raw),
        "pairing": (
            "Full MPS vs selected hybrid within "
            "each independent Python process"
        ),
        "order_counterbalance": (
            "4 Full-MPS-first and 4 hybrid-first "
            "per model"
        ),
        "interpretation": (
            "Exploratory reproducibility test; "
            "independent process rounds are the "
            "comparison units, not individual calls."
        ),
        "script_sha256": script_hash,
    }

    metadata_path = outdir / "campaign_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    final_files = [
        Path(__file__), raw_path, summary_path,
        pairs_path, metadata_path,
    ]
    (outdir / "checksums.json").write_text(
        json.dumps(
            {p.name: sha(p) for p in final_files},
            indent=2
        )
    )

    print("\n==============================")
    print("REPEATED HYBRID PILOT: PASS")
    print(f"INDEPENDENT PROCESSES: {REPEATS}")
    print(f"CONFIGURATION RUNS: {REPEATS * 4}")
    print(f"TIMED SAMPLES: {len(all_raw)}")
    print("RESULT DIRECTORY:", outdir)
    print("==============================")

def main():
    outdir = Path(sys.argv[1]).resolve()
    assert outdir.is_dir()

    if len(sys.argv) == 4 and sys.argv[2] == "--worker":
        worker(outdir, int(sys.argv[3]))
        return

    print(
        "Starting independent-process paired campaign.",
        flush=True
    )

    for number in range(1, REPEATS + 1):
        print(
            f"\n=== PROCESS {number}/{REPEATS} ===",
            flush=True
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(outdir),
                "--worker",
                str(number),
            ],
            check=True,
        )

    aggregate(outdir)

if __name__ == "__main__":
    main()
