import csv, gc, hashlib, json, math, os, platform
import statistics, subprocess, sys, time, uuid
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(sys.argv[1]).resolve()
VAL = Path("results/benchmark_v3/validation/t05_20261015T173344Z_2438/metadata.json")
INV = Path("results/benchmark_v3/inventory/inv_20261015T175122Z_2438/structural_inventory.csv")
IMGROOT = Path("data/imagenetv2/imagenetv2-matched-frequency-format-val")
MANROOT = Path("configs/partition_manifests_v2")

THREADS = [1, 2, 4, 6, 8, 10]
ROUNDS = 3
WARMUPS = 3
TIMED = 10

RAW_FIELDS = [
    "process_number", "round", "threads", "model",
    "run_id", "config_id", "mapping", "cut_after",
    "iteration", "elapsed_ns", "elapsed_ms"
]
SUMMARY_FIELDS = [
    "process_number", "round", "threads", "model",
    "run_id", "config_id", "mapping", "cut_after",
    "placement_order", "device_sequence", "transfer_events",
    "logical_transfer_bytes", "numerical_pass",
    "top1_agreement", "max_abs_error", "warmups",
    "repetitions", "median_ms", "mean_ms", "min_ms", "max_ms"
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

def loadcsv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def savecsv(path, fields, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def checksum_files(directory, files):
    (directory / "checksums.json").write_text(json.dumps(
        {p.name: sha(p) for p in files}, indent=2
    ))

def verify(directory):
    checks = json.loads((directory / "checksums.json").read_text())
    for name, expected in checks.items():
        assert sha(directory / name) == expected, f"SHA mismatch: {name}"

def stages_of(m):
    from torch import nn
    return [
        m.features[0],
        m.features[1],
        nn.Sequential(m.features[2], m.features[3]),
        nn.Sequential(m.features[4], m.features[5]),
        nn.Sequential(m.features[6], m.features[7]),
        nn.Sequential(m.avgpool, m.classifier),
    ]

def worker(number, round_no, threads):
    import torch, torchvision
    from PIL import Image
    from torchvision.models import (
        convnext_base, convnext_large,
        ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
    )

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    assert torch.backends.mps.is_available()

    torch.set_num_threads(threads)
    assert torch.get_num_threads() == threads
    torch.manual_seed(2026)

    directory = ROOT / f"process_{number:02d}"
    directory.mkdir(parents=True, exist_ok=False)

    validation = json.loads(VAL.read_text())
    image_info = validation["images"][0]
    image_path = IMGROOT / image_info["path"]
    assert sha(image_path) == image_info["sha256"]

    inventory = loadcsv(INV)
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
    expected_ranges = [
        "G1-G1", "G2-G4", "G5-G8",
        "G9-G36", "G37-G40", "G41-G41"
    ]

    specs = [
        ("convnext_base", convnext_base,
         ConvNeXt_Base_Weights.IMAGENET1K_V1,
         "convnext_base__fine41.json"),
        ("convnext_large", convnext_large,
         ConvNeXt_Large_Weights.IMAGENET1K_V1,
         "convnext_large__fine41.json"),
    ]
    if number % 2 == 0:
        specs.reverse()

    raw = []
    summaries = []
    sources = {}

    for name, factory, weights, filename in specs:
        manifest_path = MANROOT / filename
        manifest_sha = sha(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["group_count"] == 41
        assert [g["modules"] for g in manifest["groups"]] == expected_paths

        inv = sorted(
            [r for r in inventory if r["model"] == name],
            key=lambda r: int(r["stage_id"][1:])
        )
        assert len(inv) == 6
        assert [r["source_groups"] for r in inv] == expected_ranges
        assert all(r["manifest_sha256"] == manifest_sha for r in inv)

        checkpoint = (
            Path(torch.hub.get_dir()) / "checkpoints"
            / Path(urlparse(weights.url).path).name
        )
        weight_sha = validation["models"][name]["weight_sha256"]
        assert checkpoint.is_file()
        assert sha(checkpoint) == weight_sha

        sources[name] = {
            "weight_sha256": weight_sha,
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
        with Image.open(image_path) as image:
            x_cpu = weights.transforms()(
                image.convert("RGB")
            ).unsqueeze(0)

        assert tuple(x_cpu.shape) == (1, 3, 224, 224)
        with torch.inference_mode():
            reference = model(x_cpu).detach().clone()

        assert tuple(reference.shape) == (1, 1000)
        assert bool(torch.isfinite(reference).all())

        layouts = [
            ("FULL_MPS", None, ["mps"] * 6),
            ("CPU_TO_MPS", 1, ["cpu"] + ["mps"] * 5),
            ("MPS_TO_CPU", 5, ["mps"] * 5 + ["cpu"]),
        ]
        shift = round_no - 1
        layouts = layouts[shift:] + layouts[:shift]

        def events_for(devices):
            events = []
            if devices[0] == "mps":
                events.append({
                    "at": "input", "from": "cpu", "to": "mps",
                    "logical_bytes": x_cpu.numel() * x_cpu.element_size()
                })
            for i in range(1, 6):
                if devices[i] != devices[i-1]:
                    events.append({
                        "at": f"after_S{i-1}",
                        "from": devices[i-1], "to": devices[i],
                        "logical_bytes": int(inv[i-1]["output_activation_bytes"])
                    })
            if devices[-1] == "mps":
                events.append({
                    "at": "output", "from": "mps", "to": "cpu",
                    "logical_bytes": reference.numel() * reference.element_size()
                })
            assert len(events) == 2
            return events

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

        model_results = {}

        for position, (mapping, cut, devices) in enumerate(layouts, 1):
            for stage, device in zip(stages, devices):
                stage.to(device)
            torch.mps.synchronize()

            with torch.inference_mode():
                actual = forward(devices)

            assert actual.shape == reference.shape
            assert bool(torch.isfinite(actual).all())

            numeric_ok = bool(torch.allclose(
                reference, actual, atol=1e-3, rtol=1e-3
            ))
            top1_ok = (
                reference.argmax().item() == actual.argmax().item()
            )
            assert numeric_ok and top1_ok
            error = float((reference - actual).abs().max().item())

            config = {
                "model": name,
                "threads": threads,
                "mapping": mapping,
                "cut_after": cut,
                "devices": devices,
                "image_sha256": image_info["sha256"],
                "weight_sha256": weight_sha,
                "manifest_sha256": manifest_sha,
                "precision": "float32",
                "batch_size": 1,
                "input_device": "cpu",
                "output_device": "cpu",
                "torch": torch.__version__,
            }
            config_id = "cfg_" + digest(config)
            run_id = "run_" + uuid.uuid4().hex
            values = []

            with torch.inference_mode():
                for _ in range(WARMUPS):
                    warmup = forward(devices)
                    del warmup

                for iteration in range(1, TIMED + 1):
                    torch.mps.synchronize()
                    start = time.perf_counter_ns()
                    output = forward(devices)
                    elapsed = time.perf_counter_ns() - start
                    del output
                    assert elapsed > 0

                    values.append(elapsed / 1e6)
                    raw.append({
                        "process_number": number,
                        "round": round_no,
                        "threads": threads,
                        "model": name,
                        "run_id": run_id,
                        "config_id": config_id,
                        "mapping": mapping,
                        "cut_after": cut,
                        "iteration": iteration,
                        "elapsed_ns": elapsed,
                        "elapsed_ms": elapsed / 1e6,
                    })

            events = events_for(devices)
            med = statistics.median(values)
            model_results[mapping] = med

            summaries.append({
                "process_number": number,
                "round": round_no,
                "threads": threads,
                "model": name,
                "run_id": run_id,
                "config_id": config_id,
                "mapping": mapping,
                "cut_after": cut,
                "placement_order": position,
                "device_sequence": json.dumps(devices),
                "transfer_events": json.dumps(events),
                "logical_transfer_bytes": sum(e["logical_bytes"] for e in events),
                "numerical_pass": int(numeric_ok),
                "top1_agreement": int(top1_ok),
                "max_abs_error": error,
                "warmups": WARMUPS,
                "repetitions": TIMED,
                "median_ms": med,
                "mean_ms": statistics.mean(values),
                "min_ms": min(values),
                "max_ms": max(values),
            })

        print(
            f"  {name}: "
            f"MPS={model_results['FULL_MPS']:.3f} ms | "
            f"Cut1={model_results['CPU_TO_MPS']:.3f} ms | "
            f"Cut5={model_results['MPS_TO_CPU']:.3f} ms | PASS",
            flush=True
        )

        del model, stages, x_cpu, reference, actual
        gc.collect()
        torch.mps.empty_cache()

    assert len(summaries) == 6
    assert len(raw) == 6 * TIMED

    raw_path = directory / "raw.csv"
    summary_path = directory / "summary.csv"
    metadata_path = directory / "metadata.json"

    savecsv(raw_path, RAW_FIELDS, raw)
    savecsv(summary_path, SUMMARY_FIELDS, summaries)

    metadata_path.write_text(json.dumps({
        "protocol": "T11_convnext_threads_partition_v1",
        "process_number": number,
        "process_id": uuid.uuid4().hex,
        "round": round_no,
        "threads": threads,
        "observed_threads": torch.get_num_threads(),
        "script_sha256": sha(__file__),
        "validation_sha256": sha(VAL),
        "inventory_sha256": sha(INV),
        "image": image_info,
        "sources": sources,
        "warmups": WARMUPS,
        "timed_iterations": TIMED,
        "batch_size": 1,
        "precision": "float32",
        "input_device": "cpu",
        "output_device": "cpu",
        "device_transfers_timed": True,
        "model_placement_timed": False,
        "preprocessing_timed": False,
        "execution": "sequential; no pipeline overlap",
        "mps_synchronization": "before and after timed inference",
        "transfer_bytes_note": "Logical payload, not physical unified-memory traffic",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": sys.version.split()[0],
        "macos": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }, indent=2))

    checksum_files(
        directory, [raw_path, summary_path, metadata_path]
    )
    print(f"PROCESS {number}/18: PASS", flush=True)

def aggregate():
    all_raw, all_summary = [], []
    process_ids = set()
    for number in range(1, 19):
        directory = ROOT / f"process_{number:02d}"
        verify(directory)
        metadata = json.loads(
            (directory / "metadata.json").read_text()
        )
        assert metadata["process_number"] == number
        assert metadata["script_sha256"] == sha(__file__)
        assert metadata["validation_sha256"] == sha(VAL)
        assert metadata["inventory_sha256"] == sha(INV)
        assert metadata["observed_threads"] == metadata["threads"]
        assert metadata["process_id"] not in process_ids
        process_ids.add(metadata["process_id"])
        for source in metadata["sources"].values():
            assert sha(source["manifest"]) == source["manifest_sha256"]

        r = loadcsv(directory / "raw.csv")
        s = loadcsv(directory / "summary.csv")
        assert len(r) == 60 and len(s) == 6
        all_raw.extend(r)
        all_summary.extend(s)

    assert len(process_ids) == 18
    assert len(all_summary) == 108
    assert len(all_raw) == 1080
    assert len({r["run_id"] for r in all_summary}) == 108

    config_counts = Counter(r["config_id"] for r in all_summary)
    assert len(config_counts) == 36
    assert set(config_counts.values()) == {3}

    by_run = defaultdict(list)
    for sample in all_raw:
        by_run[sample["run_id"]].append(sample)

    assert len(by_run) == 108

    for record in all_summary:
        values = by_run[record["run_id"]]
        assert len(values) == 10
        assert sorted(int(x["iteration"]) for x in values) == list(
            range(1, 11)
        )
        assert int(record["numerical_pass"]) == 1
        assert int(record["top1_agreement"]) == 1
        assert len(json.loads(record["transfer_events"])) == 2
        assert all(int(x["elapsed_ns"]) > 0 for x in values)
        assert math.isclose(
            statistics.median(float(x["elapsed_ms"]) for x in values),
            float(record["median_ms"]), abs_tol=1e-9
        )

    comparisons = []
    for name in ("convnext_base", "convnext_large"):
        for threads in THREADS:
            rows = [
                r for r in all_summary
                if r["model"] == name and int(r["threads"]) == threads
            ]
            assert len(rows) == 9

            for mapping in ("FULL_MPS", "CPU_TO_MPS", "MPS_TO_CPU"):
                positions = sorted(
                    int(r["placement_order"]) for r in rows
                    if r["mapping"] == mapping
                )
                assert positions == [1, 2, 3]

            for round_no in range(1, 4):
                group = [
                    r for r in rows if int(r["round"]) == round_no
                ]
                assert len(group) == 3
                baseline = next(
                    float(r["median_ms"]) for r in group
                    if r["mapping"] == "FULL_MPS"
                )

                for mapping in ("CPU_TO_MPS", "MPS_TO_CPU"):
                    candidate = next(
                        float(r["median_ms"]) for r in group
                        if r["mapping"] == mapping
                    )
                    comparisons.append({
                        "model": name,
                        "threads": threads,
                        "round": round_no,
                        "mapping": mapping,
                        "full_mps_ms": baseline,
                        "hybrid_ms": candidate,
                        "delta_percent": 100 * (candidate / baseline - 1),
                        "hybrid_faster": int(candidate < baseline),
                    })

    assert len(comparisons) == 72

    print("\n=== T11 PAIRED THREAD RESULTS ===")
    for name in ("convnext_base", "convnext_large"):
        print(f"\n{name}:")
        print(
            f"{'Threads':>7} {'Cut1 wins':>11} "
            f"{'Cut1 Δ':>10} {'Cut5 wins':>11} {'Cut5 Δ':>10}"
        )
        for threads in THREADS:
            groups = []
            for mapping in ("CPU_TO_MPS", "MPS_TO_CPU"):
                selected = [
                    r for r in comparisons
                    if r["model"] == name
                    and r["threads"] == threads
                    and r["mapping"] == mapping
                ]
                assert len(selected) == 3
                groups.append((
                    sum(r["hybrid_faster"] for r in selected),
                    statistics.median(
                        r["delta_percent"] for r in selected
                    ),
                ))
            print(
                f"{threads:>7} "
                f"{str(groups[0][0])+'/3':>11} "
                f"{groups[0][1]:>+9.3f}% "
                f"{str(groups[1][0])+'/3':>11} "
                f"{groups[1][1]:>+9.3f}%"
            )

    raw_path = ROOT / "all_raw.csv"
    summary_path = ROOT / "all_summary.csv"
    pairs_path = ROOT / "paired_comparison.csv"
    meta_path = ROOT / "campaign_metadata.json"

    savecsv(raw_path, RAW_FIELDS, all_raw)
    savecsv(summary_path, SUMMARY_FIELDS, all_summary)
    savecsv(pairs_path, list(comparisons[0]), comparisons)

    meta_path.write_text(json.dumps({
        "protocol": "T11_convnext_threads_partition_v1",
        "models": ["convnext_base", "convnext_large"],
        "thread_counts": THREADS,
        "rounds_per_thread": ROUNDS,
        "independent_processes": 18,
        "configuration_runs": len(all_summary),
        "timed_samples": len(all_raw),
        "paired_comparisons": len(comparisons),
        "order_balance": "Each mapping occupies each position once per thread",
        "script_sha256": sha(__file__),
        "validation_sha256": sha(VAL),
        "inventory_sha256": sha(INV),
        "limitations": [
            "One fixed ImageNetV2 image",
            "Three independent processes per thread setting",
            "CPU threads fixed within each process",
            "Only Full MPS and two selected single-cut hybrids",
            "No energy or memory measurement",
            "No pipeline overlap",
            "System load and thermal effects not controlled",
            "Exploratory comparison, not inferential optimization",
        ],
    }, indent=2))

    checksum_files(
        ROOT, [
            Path(__file__), raw_path, summary_path,
            pairs_path, meta_path
        ]
    )

    print("\n==============================")
    print("T11 THREAD × PARTITION CAMPAIGN: PASS")
    print("INDEPENDENT PROCESSES: 18")
    print("CONFIGURATION RUNS: 108")
    print("TIMED SAMPLES: 1080")
    print("PAIRED COMPARISONS: 72")
    print("RESULT DIRECTORY:", ROOT)
    print("==============================")

def main():
    assert ROOT.is_dir()
    if len(sys.argv) == 6 and sys.argv[2] == "--worker":
        worker(
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5])
        )
        return

    number = 0
    for round_no in range(1, ROUNDS + 1):
        shift = round_no - 1
        order = THREADS[shift:] + THREADS[:shift]
        print(f"\n=== ROUND {round_no}/3 ===", flush=True)

        for threads in order:
            number += 1
            print(
                f"\nPROCESS {number}/18 THREADS={threads}",
                flush=True
            )
            subprocess.run([
                sys.executable, "-u",
                str(Path(__file__).resolve()),
                str(ROOT), "--worker",
                str(number), str(round_no), str(threads),
            ], check=True)

    aggregate()

if __name__ == "__main__":
    main()
