import ctypes
import hashlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"
G4 = BASE / "production_campaigns/g4_campaign_mac_v1"
OUT = G4 / "pipeline_core_smoke_v1"
PROTOCOL = "G4_LONG_PERFORMANCE_V1"
QOS = 0x15

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def atomic_json(path, data):
    assert not path.exists(), path
    tmp = path.with_suffix(".tmp")
    assert not tmp.exists(), tmp
    with tmp.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def source_module(name):
    path = G4 / "source_snapshots" / name
    spec = importlib.util.spec_from_file_location(
        "g4_" + name.replace(".", "_"), path
    )
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def set_qos():
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    setter = lib.pthread_set_qos_class_self_np
    getter = lib.qos_class_self
    setter.argtypes = [ctypes.c_uint, ctypes.c_int]
    setter.restype = ctypes.c_int
    getter.restype = ctypes.c_uint
    rc = setter(QOS, 0)
    actual = int(getter())
    if rc != 0 or actual != QOS:
        raise RuntimeError(
            f"QOS_FAILURE rc={rc} actual={actual}"
        )
    return actual

def load_vision(case):
    import torch
    from torch import nn
    from PIL import Image
    from urllib.parse import urlparse
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        convnext_base, ConvNeXt_Base_Weights,
        convnext_large, ConvNeXt_Large_Weights,
    )

    src = source_module("vision_pilot.py")
    choices = {
        "resnet50": (
            resnet50, ResNet50_Weights.IMAGENET1K_V2,
            "resnet50__stage6.json",
        ),
        "convnext_base": (
            convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1,
            "convnext_base__fine41.json",
        ),
        "convnext_large": (
            convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1,
            "convnext_large__fine41.json",
        ),
    }
    make, weights, manifest_name = choices[case["model"]]

    checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )
    expected = json.loads(
        src.T05.read_text()
    )["models"][case["model"]]["weight_sha256"]
    assert checkpoint.is_file()
    assert digest(checkpoint) == expected

    records = json.loads(src.HOLDOUT.read_text())
    assert len(records) >= 8

    images = []
    hashes = []
    transform = weights.transforms()
    for record in records[:8]:
        path = src.IMAGES / record["path"]
        assert digest(path) == record["sha256"]
        with Image.open(path) as im:
            images.append(transform(im.convert("RGB")))
        hashes.append(record["sha256"])

    inputs = torch.stack(images)

    model = make(weights=None).eval().cpu()
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    del state

    with torch.inference_mode():
        reference = model(inputs).float().cpu()
    assert bool(torch.isfinite(reference).all())

    manifest_path = src.MANIFESTS / manifest_name
    groups = json.loads(manifest_path.read_text())["groups"]

    if case["model"] == "resnet50":
        stages = [
            nn.Sequential(
                model.conv1, model.bn1,
                model.relu, model.maxpool,
            ),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.Sequential(
                model.avgpool, nn.Flatten(1), model.fc
            ),
        ]
    else:
        stages = [
            nn.Sequential(*[
                model.get_submodule(p) for p in group["modules"]
            ])
            for group in groups
        ]

    assert len(stages) == len(groups)
    cut = case["cut"]
    assert 0 < cut < len(stages)

    first, second = case["devices"]
    for i, stage in enumerate(stages):
        stage.to(first if i < cut else second)

    def stage1(x):
        x = x.to(first)
        for stage in stages[:cut]:
            x = stage(x)
        return x

    def stage2(x):
        for stage in stages[cut:]:
            x = stage(x)
        return x

    return {
        "inputs": inputs,
        "reference": reference,
        "stage1": stage1,
        "stage2": stage2,
        "batch_size": 8,
        "context_tokens": None,
        "throughput_unit": "images_per_second",
        "stage_count": len(stages),
        "provenance": {
            "checkpoint_sha256": digest(checkpoint),
            "manifest_sha256": digest(manifest_path),
            "image_sha256_list": hashes,
        },
    }

def load_text(case):
    import torch
    import pyarrow.parquet as pq
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer
    )

    src = source_module("text_pilot.py")
    repo, revision, _ = src.SNAPSHOTS[case["model"]]
    snapshot = (
        Path.home() / ".cache/huggingface/hub"
        / repo / "snapshots" / revision
    )
    assert (snapshot / "config.json").is_file()

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    texts = pq.read_table(
        src.CORPUS, columns=["text"]
    )["text"].to_pylist()
    excerpt = "\n".join(
        t.strip() for t in texts
        if isinstance(t, str) and t.strip()
    )[:4000]
    tokens = tokenizer.encode(
        excerpt, add_special_tokens=False
    )
    assert len(tokens) >= 256

    ids = torch.tensor([
        tokens[:128], tokens[128:256],
        tokens[:128], tokens[128:256],
    ], dtype=torch.long)

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval().cpu()

    with torch.inference_mode():
        ref_two = model(
            input_ids=ids[:2],
            use_cache=False,
            return_dict=True,
        ).logits.detach().cpu()
        reference = ref_two.repeat(2, 1, 1)
    assert bool(torch.isfinite(reference).all())

    distil = case["model"] == "distilgpt2"
    backbone = (
        model.transformer if distil else model.model
    )
    blocks = list(
        backbone.h if distil else backbone.layers
    )
    embedding = (
        backbone.wte if distil else backbone.embed_tokens
    )
    head = model.lm_head

    assert embedding.weight.data_ptr() == head.weight.data_ptr()
    head.weight = torch.nn.Parameter(
        head.weight.detach().clone(),
        requires_grad=False,
    )
    assert embedding.weight.data_ptr() != head.weight.data_ptr()

    cut = case["cut"]
    assert 0 < cut < len(blocks)
    first, second = case["devices"]

    for i, block in enumerate(blocks):
        block.to(first if i < cut else second)

    if distil:
        backbone.wte.to(first)
        backbone.wpe.to(first)
        backbone.drop.to(first)
        backbone.ln_f.to(second)
    else:
        backbone.embed_tokens.to(first)
        backbone.norm.to(second)

    head.to(second)

    positions = {
        d: torch.arange(
            128, dtype=torch.long, device=d
        ).unsqueeze(0)
        for d in ("cpu", "mps")
    }

    future = torch.triu(
        torch.ones(128, 128, dtype=torch.bool),
        diagonal=1,
    )
    mask_cpu = torch.zeros(
        (1, 1, 128, 128), dtype=torch.float32
    ).masked_fill(
        future, torch.finfo(torch.float32).min
    )
    masks = {
        "cpu": mask_cpu,
        "mps": mask_cpu.to("mps"),
    }

    rotary = {}
    if not distil:
        for d in ("cpu", "mps"):
            dummy = torch.zeros(
                1, 128, model.config.hidden_size,
                dtype=torch.float32, device=d
            )
            with torch.inference_mode():
                rotary[d] = backbone.rotary_emb(
                    dummy, positions[d]
                )
            del dummy

    def block_forward(hidden, selection):
        device = hidden.device.type
        for block in selection:
            if distil:
                item = block(
                    hidden,
                    attention_mask=masks[device],
                    use_cache=False,
                )
            else:
                item = block(
                    hidden_states=hidden,
                    attention_mask=masks[device],
                    position_ids=positions[device],
                    position_embeddings=rotary[device],
                    use_cache=False,
                )
            hidden = (
                item[0] if isinstance(item, (tuple, list))
                else item
            )
        return hidden

    def stage1(x):
        x = x.to(first)
        if distil:
            hidden = backbone.drop(
                backbone.wte(x)
                + backbone.wpe(positions[first])
            )
        else:
            hidden = backbone.embed_tokens(x)
        return block_forward(hidden, blocks[:cut])

    def stage2(hidden):
        hidden = block_forward(hidden, blocks[cut:])
        if distil:
            return head(backbone.ln_f(hidden))
        return head(backbone.norm(hidden))

    return {
        "inputs": ids,
        "reference": reference,
        "stage1": stage1,
        "stage2": stage2,
        "batch_size": 4,
        "context_tokens": 128,
        "throughput_unit": "prefill_tokens_per_second",
        "stage_count": len(blocks),
        "provenance": {
            "snapshot_revision": revision,
            "input_token_sha256": hashlib.sha256(
                ids.numpy().tobytes()
            ).hexdigest(),
            "attention": "eager",
            "use_cache": False,
        },
    }

def execute(case):
    import psutil
    import torch

    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()
    assert case["precision"] == "fp32"
    assert case["direction"] in ("cpu_mps", "mps_cpu")
    assert case["microbatch"] in (1, 2)

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    case["devices"] = (
        ("cpu", "mps")
        if case["direction"] == "cpu_mps"
        else ("mps", "cpu")
    )

    model = (
        load_vision(case)
        if case["model"] in (
            "resnet50", "convnext_base", "convnext_large"
        )
        else load_text(case)
    )

    first, second = case["devices"]
    chunks = list(
        model["inputs"].split(case["microbatch"])
    )
    assert len(chunks) >= 2

    stage1_fn = model["stage1"]
    stage2_fn = model["stage2"]
    reference = model["reference"]

    torch.mps.synchronize()
    process = psutil.Process()

    def first_part(x):
        begin = time.perf_counter_ns()

        with torch.inference_mode():
            hidden = stage1_fn(x)
            if first == "mps":
                torch.mps.synchronize()

            compute_end = time.perf_counter_ns()
            logical_bytes = hidden.numel() * hidden.element_size()
            activation_shape = list(hidden.shape)

            transfer_ms = None
            if first == "mps":
                transfer_start = time.perf_counter_ns()
                hidden = hidden.to("cpu", copy=True)
                torch.mps.synchronize()
                transfer_ms = (
                    time.perf_counter_ns() - transfer_start
                ) / 1e6

        return hidden, {
            "stage1_ms": (compute_end - begin) / 1e6,
            "transfer_ms": transfer_ms,
            "logical_activation_bytes": logical_bytes,
            "activation_shape": activation_shape,
        }

    def second_part(hidden, event):
        with torch.inference_mode():
            if second == "mps":
                transfer_start = time.perf_counter_ns()
                hidden = hidden.to("mps", copy=True)
                torch.mps.synchronize()
                event["transfer_ms"] = (
                    time.perf_counter_ns() - transfer_start
                ) / 1e6

            start = time.perf_counter_ns()
            result = stage2_fn(hidden)

            if second == "mps":
                torch.mps.synchronize()

            finish = time.perf_counter_ns()
            event["stage2_ms"] = (finish - start) / 1e6

            result = result.float().cpu()
            torch.mps.synchronize()

        assert event["transfer_ms"] is not None
        return result, event

    def round_trip(mode):
        events = [None] * len(chunks)
        outputs = [None] * len(chunks)
        qos_proof = {}
        failures = []
        stop = threading.Event()
        channel = queue.Queue(maxsize=2)

        def producer():
            try:
                qos_proof["producer"] = set_qos()
                with torch.inference_mode():
                    for i, x in enumerate(chunks):
                        if stop.is_set():
                            return
                        hidden, event = first_part(x)
                        while not stop.is_set():
                            try:
                                channel.put(
                                    (i, hidden, event), timeout=0.2
                                )
                                break
                            except queue.Full:
                                pass
                    while not stop.is_set():
                        try:
                            channel.put(None, timeout=0.2)
                            break
                        except queue.Full:
                            pass
            except BaseException:
                failures.append(traceback.format_exc())
                stop.set()

        def consumer():
            try:
                qos_proof["consumer"] = set_qos()
                with torch.inference_mode():
                    while True:
                        try:
                            item = channel.get(timeout=0.2)
                        except queue.Empty:
                            if stop.is_set():
                                return
                            continue
                        if item is None:
                            return
                        i, hidden, event = item
                        outputs[i], events[i] = second_part(
                            hidden, event
                        )
            except BaseException:
                failures.append(traceback.format_exc())
                stop.set()

        start = time.perf_counter_ns()

        if mode == "sequential":
            for i, x in enumerate(chunks):
                hidden, event = first_part(x)
                outputs[i], events[i] = second_part(
                    hidden, event
                )
        else:
            a = threading.Thread(
                target=producer, daemon=True
            )
            b = threading.Thread(
                target=consumer, daemon=True
            )
            b.start()
            a.start()
            a.join(timeout=900)
            b.join(timeout=900)
            assert not a.is_alive() and not b.is_alive(), (
                "PIPELINE_THREAD_TIMEOUT"
            )
            assert not failures, "\n".join(failures)
            assert qos_proof == {
                "producer": QOS,
                "consumer": QOS,
            }, qos_proof

        torch.mps.synchronize()
        wall_s = (time.perf_counter_ns() - start) / 1e9

        assert all(x is not None for x in outputs)
        assert all(x is not None for x in events)

        output = torch.cat(outputs)
        assert output.shape == reference.shape
        assert bool(torch.isfinite(output).all()), (
            "NONFINITE_OUTPUT"
        )
        assert torch.allclose(
            output, reference, rtol=1e-3, atol=1e-3
        ), (
            "FP32_NUMERICAL_MISMATCH",
            float((output - reference).abs().max()),
        )
        assert bool(torch.equal(
            output.argmax(-1), reference.argmax(-1)
        )), "TASK_OUTPUT_MISMATCH"

        return {
            "wall_s": wall_s,
            "events": events,
            "qos_evidence": qos_proof,
            "max_abs_error": float(
                (output - reference).abs().max()
            ),
        }

    results = {}

    for mode in ("sequential", "pipeline"):
        print("MODE:", mode, flush=True)
        warmup = round_trip(mode)
        del warmup


        observer_stop = threading.Event()
        observer_errors = []
        peaks = {
            "rss": process.memory_info().rss,
            "mps_current": torch.mps.current_allocated_memory(),
            "mps_driver": torch.mps.driver_allocated_memory(),
        }

        def observe_memory():
            while not observer_stop.is_set():
                try:
                    peaks["rss"] = max(
                        peaks["rss"], process.memory_info().rss
                    )
                    peaks["mps_current"] = max(
                        peaks["mps_current"],
                        torch.mps.current_allocated_memory()
                    )
                    peaks["mps_driver"] = max(
                        peaks["mps_driver"],
                        torch.mps.driver_allocated_memory()
                    )
                except BaseException as exc:
                    observer_errors.append(repr(exc))
                    observer_stop.set()
                observer_stop.wait(0.05)

        observer = threading.Thread(
            target=observe_memory, daemon=True
        )
        rounds = []
        observer.start()
        window_start_ns = time.monotonic_ns()

        try:
            while (
                len(rounds) < 3
                or (
                    time.monotonic_ns() - window_start_ns
                ) < 16000000000
            ):
                rounds.append(round_trip(mode))
                assert len(rounds) < 100000
        finally:
            window_end_ns = time.monotonic_ns()
            observer_stop.set()
            observer.join(timeout=5)

        assert not observer_errors, observer_errors
        assert not observer.is_alive()
        total_wall = sum(r["wall_s"] for r in rounds)
        batch = model["batch_size"]
        work = (
            batch if model["context_tokens"] is None
            else batch * model["context_tokens"]
        )
        all_events = [
            event
            for result in rounds
            for event in result["events"]
        ]
        import statistics

        logical = {
            e["logical_activation_bytes"]
            for e in all_events
        }
        shapes = {
            tuple(e["activation_shape"])
            for e in all_events
        }
        assert len(logical) == 1
        assert len(shapes) == 1

        results[mode] = {
            "window_start_monotonic_ns": window_start_ns,
            "window_end_monotonic_ns": window_end_ns,
            "rss_peak_sampled_bytes": peaks["rss"],
            "mps_current_peak_sampled_bytes":
                peaks["mps_current"],
            "mps_driver_peak_sampled_bytes":
                peaks["mps_driver"],
            "memory_sampling_interval_s": 0.05,
            "rounds": len(rounds),
            "total_work_items": work * len(rounds),
            "work_unit": model["throughput_unit"],
            "wall_seconds": total_wall,
            "throughput_per_second":
                work * len(rounds) / total_wall,
            "stage1_median_ms": statistics.median(
                e["stage1_ms"] for e in all_events
            ),
            "transfer_median_ms": statistics.median(
                e["transfer_ms"] for e in all_events
            ),
            "stage2_median_ms": statistics.median(
                e["stage2_ms"] for e in all_events
            ),
            "logical_activation_bytes_per_microbatch":
                next(iter(logical)),
            "activation_shape": list(next(iter(shapes))),
            "numerical_status": "numerical_pass",
            "max_abs_error": max(
                r["max_abs_error"] for r in rounds
            ),
            "power_measured": False,
            "energy_measured": False,
            "rss_snapshot_bytes":
                process.memory_info().rss,
            "mps_current_snapshot_bytes":
                torch.mps.current_allocated_memory(),
            "mps_driver_snapshot_bytes":
                torch.mps.driver_allocated_memory(),
            "peak_memory_measured": False,
            "gpu_kernel_overlap_proven": False,
            "qos_thread_evidence":
                rounds[-1]["qos_evidence"],
        }

        print(
            "THROUGHPUT:",
            round(results[mode]["throughput_per_second"], 3),
            model["throughput_unit"],
            "TRANSFER:",
            round(results[mode]["transfer_median_ms"], 4),
            "ms",
            flush=True,
        )

    ratio = (
        results["pipeline"]["throughput_per_second"]
        / results["sequential"]["throughput_per_second"]
    )

    return {
        "protocol": PROTOCOL,
        "case": case,
        "stage_count": model["stage_count"],
        "model_provenance": model["provenance"],
        "modes": results,
        "pipeline_speedup": ratio,
        "concurrent_pipeline": True,
        "producer_consumer_queue_maxsize": 2,
        "power_measured": False,
        "energy_measured": False,
        "publication_ready": False,
        "note": (
            "Functional pipeline acceptance: "
            "two timed rounds per mode, not production "
            "power or confidence interval measurement"
        ),
    }

SMOKE_CASES = [
    ("resnet50", 2, "cpu_mps"),
    ("convnext_base", 20, "mps_cpu"),
    ("convnext_large", 20, "cpu_mps"),
    ("distilgpt2", 1, "mps_cpu"),
    ("qwen2_5_1_5b", 10, "cpu_mps"),
    ("qwen3_1_7b", 10, "mps_cpu"),
]

def smoke():
    OUT.mkdir(parents=True, exist_ok=True)
    print("G4.7 PIPELINE ACCEPTANCE", flush=True)

    for index, (model, cut, direction) in enumerate(
        SMOKE_CASES, 1
    ):
        name = f"{model}_cut{cut}_{direction}"
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        complete = folder / "result.json"

        if complete.exists():
            record = json.loads(complete.read_text())
            assert record["protocol"] == PROTOCOL
            assert record["case"]["model"] == model
            assert record["case"]["cut"] == cut
            assert record["case"]["direction"] == direction
            print("SKIP VERIFIED:", name)
            continue

        case = {
            "model": model,
            "cut": cut,
            "direction": direction,
            "precision": "fp32",
            "microbatch": 2,
        }

        attempt = 1
        while (
            folder / f"attempt_{attempt:03d}"
        ).exists():
            attempt += 1

        attempt_dir = folder / f"attempt_{attempt:03d}"
        attempt_dir.mkdir()
        config = attempt_dir / "case.json"
        result = attempt_dir / "result.json"

        atomic_json(config, case)
        print(
            f"[{index}/{len(SMOKE_CASES)}]",
            name, flush=True
        )

        env = {
            **os.environ,
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }

        with (attempt_dir / "stdout.txt").open("x") as out, \
             (attempt_dir / "stderr.txt").open("x") as err:
            try:
                proc = subprocess.run(
                    [
                        sys.executable, "-u",
                        str(Path(__file__).resolve()),
                        "worker", str(config), str(result),
                    ],
                    stdout=out,
                    stderr=err,
                    env=env,
                    timeout=1800,
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124

        if rc != 0 or not result.exists():
            tail = (
                attempt_dir / "stderr.txt"
            ).read_text(errors="replace")[-2000:]
            print("STOP: WORKER FAILED", name)
            print("EXIT CODE:", rc)
            print(tail)
            raise SystemExit(1)

        obj = json.loads(result.read_text())
        assert obj["protocol"] == PROTOCOL
        assert obj["case"]["model"] == model
        assert obj["case"]["cut"] == cut
        assert obj["case"]["direction"] == direction
        assert set(obj["modes"]) == {
            "sequential", "pipeline"
        }

        # No overwriting previous successful results.
        os.link(result, complete)

        print(
            "PASS:", name,
            "SPEEDUP:",
            round(obj["pipeline_speedup"], 4),
            flush=True,
        )

    print("\nG4.7 SIX-MODEL PIPELINE ACCEPTANCE: PASS")
    print("OUTPUT:", OUT)
    print("POWER/ENERGY MEASURED: NO")
    print("OLD BENCHMARKS MODIFIED: NO")
    print("GRAPH SQLITE MODIFIED: NO")

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "smoke":
        smoke()
    elif len(sys.argv) == 4 and sys.argv[1] == "worker":
        case = json.loads(Path(sys.argv[2]).read_text())
        result = execute(case)
        atomic_json(Path(sys.argv[3]), result)
    else:
        raise SystemExit(
            "Usage: g4_pipeline_core.py smoke"
        )

if __name__ == "__main__":
    main()
