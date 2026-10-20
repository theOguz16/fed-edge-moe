import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"
CAMPAIGN = BASE / "production_campaigns/g4_campaign_mac_v1"
SNAP = CAMPAIGN / "source_snapshots"
RUN = CAMPAIGN / "numerical_v1"
DB = RUN / "numerical.sqlite"
REGISTRY = (
    BASE / "production_campaigns/g2_cut_graph_mac"
    / "cut_registry_144.csv"
)

PROTOCOL = "G4_NUMERICAL_SCREEN_V1"
PILOT_CUTS = {
    "resnet50": 2,
    "convnext_base": 20,
    "convnext_large": 20,
    "distilgpt2": 1,
    "qwen2_5_1_5b": 10,
    "qwen3_1_7b": 10,
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_source(name):
    path = SNAP / name
    spec = importlib.util.spec_from_file_location(
        "g4_source_" + name.replace(".", "_"), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify_sources():
    manifest = json.loads(
        (CAMPAIGN / "source_manifest.json").read_text()
    )
    for entry in manifest["sources"]:
        filename = Path(entry["snapshot"]).name
        assert digest(SNAP / filename) == entry["sha256"], filename

def save_atomic(path, obj):
    assert not path.exists(), path
    tmp = path.with_suffix(path.suffix + ".tmp")
    assert not tmp.exists(), tmp
    with tmp.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def sync():
    import torch
    torch.mps.synchronize()

def measure_split(first, second, cut, initial, advance):
    import torch
    sync()
    start = time.perf_counter_ns()
    hidden = initial(first)
    hidden = advance(hidden, 0, cut)
    sync()
    stage1_end = time.perf_counter_ns()

    logical_bytes = hidden.numel() * hidden.element_size()
    shape = list(hidden.shape)
    source_dtype = str(hidden.dtype)

    hidden = hidden.to(second, copy=True)
    sync()
    transfer_end = time.perf_counter_ns()

    hidden = advance(hidden, cut, None)
    sync()
    stage2_end = time.perf_counter_ns()

    return hidden, {
        "logical_activation_bytes": logical_bytes,
        "activation_shape": shape,
        "activation_dtype": source_dtype,
        "stage1_elapsed_ms": (stage1_end - start) / 1e6,
        "transfer_elapsed_ms": (transfer_end - stage1_end) / 1e6,
        "stage2_elapsed_ms": (stage2_end - transfer_end) / 1e6,
        "timing_scope": "single_forward_diagnostic_not_sustained",
        "transfer_scope": "tensor_to_copy_true_mps_synchronized",
        "physical_bus_bytes_measured": False,
    }

def numerical_compare(actual, reference, kind):
    import torch
    same_shape = actual.shape == reference.shape
    finite = bool(torch.isfinite(actual).all())

    result = {
        "output_shape_match": same_shape,
        "output_finite": finite,
        "strict_allclose_1e_3": None,
        "max_abs_logit_error": None,
        "top1_agreement": None,
        "token_agreement": None,
        "kl_fp32_to_candidate_nats": None,
    }

    if not same_shape or not finite:
        result["numerical_status"] = "numerical_invalid"
        return result

    actual = actual.float()
    reference = reference.float()

    error = float((actual - reference).abs().max())
    strict = bool(torch.allclose(
        actual, reference, atol=1e-3, rtol=1e-3
    ))

    result["max_abs_logit_error"] = error
    result["strict_allclose_1e_3"] = strict

    if kind == "vision":
        agreement = float(
            (actual.argmax(-1) == reference.argmax(-1))
            .float().mean()
        )
        result["top1_agreement"] = agreement
    else:
        agreement = float(
            (actual.argmax(-1) == reference.argmax(-1))
            .float().mean()
        )
        result["token_agreement"] = agreement

        p = torch.log_softmax(reference, dim=-1)
        q = torch.log_softmax(actual, dim=-1)
        kl = float(
            (p.exp() * (p - q)).sum(dim=-1).mean()
        )
        assert math.isfinite(kl)
        result["kl_fp32_to_candidate_nats"] = kl

    result["numerical_status"] = (
        "numerical_pass"
        if strict and agreement == 1.0
        else "numerical_mismatch"
    )
    return result

def run_vision(model_name, cut, precision, first, second):
    import psutil
    import torch
    from torch import nn
    from PIL import Image
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        convnext_base, ConvNeXt_Base_Weights,
        convnext_large, ConvNeXt_Large_Weights,
    )
    from urllib.parse import urlparse

    source = load_source("vision_pilot.py")
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
    factory, weights, manifest_file = choices[model_name]

    checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )
    expected = json.loads(
        source.T05.read_text()
    )["models"][model_name]["weight_sha256"]

    assert checkpoint.is_file()
    assert digest(checkpoint) == expected

    model = factory(weights=None).eval()
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    del state

    records = json.loads(source.HOLDOUT.read_text())
    tensors = []
    image_hashes = []

    for item in records[:2]:
        path = source.IMAGES / item["path"]
        assert digest(path) == item["sha256"]
        with Image.open(path) as image:
            tensors.append(
                weights.transforms()(image.convert("RGB"))
            )
        image_hashes.append(item["sha256"])

    images = torch.stack(tensors)

    with torch.inference_mode():
        reference = model(images).detach().cpu()
    assert bool(torch.isfinite(reference).all())

    manifest_path = source.MANIFESTS / manifest_file
    manifest = json.loads(manifest_path.read_text())
    groups = manifest["groups"]

    if model_name == "resnet50":
        stages = [
            nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool
            ),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.Sequential(model.avgpool, nn.Flatten(1), model.fc),
        ]
    else:
        stages = [
            nn.Sequential(*[
                model.get_submodule(path)
                for path in group["modules"]
            ])
            for group in groups
        ]

    assert len(stages) == len(groups)
    assert 0 < cut < len(stages)

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
    }[precision]

    for index, stage in enumerate(stages):
        stage.to(
            device=first if index < cut else second,
            dtype=dtype,
        )

    def initial(device):
        return images.to(device=device, dtype=dtype)

    def advance(hidden, start, stop):
        upper = len(stages) if stop is None else stop
        for stage in stages[start:upper]:
            hidden = stage(hidden)
        return hidden

    process = psutil.Process()
    rss_before = process.memory_info().rss

    with torch.inference_mode():
        hidden, metrics = measure_split(
            first, second, cut, initial, advance
        )
        actual = hidden.float().cpu()
        sync()

    rss_after = process.memory_info().rss

    metrics.update(
        numerical_compare(actual, reference, "vision")
    )
    metrics.update({
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": digest(manifest_path),
        "input_image_sha256_list": image_hashes,
        "rss_before_forward_bytes": rss_before,
        "rss_after_forward_bytes": rss_after,
        "rss_peak_measured": False,
        "mps_current_after_bytes":
            torch.mps.current_allocated_memory(),
        "mps_driver_after_bytes":
            torch.mps.driver_allocated_memory(),
    })
    return metrics

def run_text(model_name, cut, precision, first, second):
    import psutil
    import torch
    import pyarrow.parquet as pq
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer
    )

    source = load_source("text_pilot.py")
    repo, revision, _ = source.SNAPSHOTS[model_name]

    snapshot = (
        Path.home() / ".cache/huggingface/hub"
        / repo / "snapshots" / revision
    )
    assert (snapshot / "config.json").is_file()
    assert source.CORPUS.is_file()

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
    )
    texts = pq.read_table(
        source.CORPUS, columns=["text"]
    )["text"].to_pylist()
    excerpt = "\n".join(
        t.strip() for t in texts
        if isinstance(t, str) and t.strip()
    )[:4000]

    tokens = tokenizer.encode(
        excerpt, add_special_tokens=False
    )
    assert len(tokens) >= 256

    ids = torch.tensor(
        [tokens[:128], tokens[128:256]],
        dtype=torch.long,
    )
    token_hash = hashlib.sha256(
        ids.numpy().tobytes()
    ).hexdigest()

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval().cpu()

    with torch.inference_mode():
        reference = model(
            input_ids=ids,
            use_cache=False,
            return_dict=True,
        ).logits.detach().cpu()
    assert bool(torch.isfinite(reference).all())

    is_distil = model_name == "distilgpt2"

    if is_distil:
        backbone = model.transformer
        blocks = list(backbone.h)
        embedding = backbone.wte
    else:
        backbone = model.model
        blocks = list(backbone.layers)
        embedding = backbone.embed_tokens

    assert 0 < cut < len(blocks)

    head = model.lm_head
    assert (
        embedding.weight.data_ptr()
        == head.weight.data_ptr()
    ), "Expected tied input/output weights"

    head.weight = torch.nn.Parameter(
        head.weight.detach().clone(),
        requires_grad=False,
    )

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
    }[precision]

    for index, layer in enumerate(blocks):
        layer.to(
            device=first if index < cut else second,
            dtype=dtype,
        )

    if is_distil:
        for module in (
            backbone.wte, backbone.wpe, backbone.drop
        ):
            module.to(device=first, dtype=dtype)
        backbone.ln_f.to(device=second, dtype=dtype)
    else:
        backbone.embed_tokens.to(
            device=first, dtype=dtype
        )
        backbone.norm.to(device=second, dtype=dtype)

    head.to(device=second, dtype=dtype)

    positions = {
        device: torch.arange(
            128, dtype=torch.long, device=device
        ).unsqueeze(0)
        for device in ("cpu", "mps")
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
    if not is_distil:
        for device in ("cpu", "mps"):
            dummy = torch.zeros(
                1, 128, model.config.hidden_size,
                dtype=dtype, device=device
            )
            with torch.inference_mode():
                rotary[device] = backbone.rotary_emb(
                    dummy, positions[device]
                )

    def initial(device):
        if is_distil:
            return backbone.drop(
                backbone.wte(ids.to(device))
                + backbone.wpe(positions[device])
            )
        return backbone.embed_tokens(ids.to(device))

    def advance(hidden, start, stop):
        upper = len(blocks) if stop is None else stop

        for layer in blocks[start:upper]:
            device = hidden.device.type
            if is_distil:
                result = layer(
                    hidden,
                    attention_mask=masks[device],
                    use_cache=False,
                )
            else:
                result = layer(
                    hidden_states=hidden,
                    attention_mask=masks[device],
                    position_ids=positions[device],
                    position_embeddings=rotary[device],
                    use_cache=False,
                )

            hidden = (
                result[0]
                if isinstance(result, (tuple, list))
                else result
            )

        return hidden

    process = psutil.Process()
    rss_before = process.memory_info().rss

    with torch.inference_mode():
        hidden, metrics = measure_split(
            first, second, cut, initial, advance
        )

        if is_distil:
            logits = head(backbone.ln_f(hidden))
        else:
            logits = head(backbone.norm(hidden))

        sync()
        actual = logits.float().cpu()

    rss_after = process.memory_info().rss

    metrics.update(
        numerical_compare(actual, reference, "text")
    )
    metrics.update({
        "snapshot_revision": revision,
        "input_token_sha256": token_hash,
        "context_tokens": 128,
        "batch_size": 2,
        "rss_before_forward_bytes": rss_before,
        "rss_after_forward_bytes": rss_after,
        "rss_peak_measured": False,
        "mps_current_after_bytes":
            torch.mps.current_allocated_memory(),
        "mps_driver_after_bytes":
            torch.mps.driver_allocated_memory(),
    })
    return metrics

def worker(model, cut, precision, direction, target):
    import torch

    verify_sources()
    assert os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK"
    ) == "0"
    assert torch.backends.mps.is_available()

    torch.manual_seed(2026)
    torch.set_num_threads(4)

    first, second = {
        "cpu_mps": ("cpu", "mps"),
        "mps_cpu": ("mps", "cpu"),
    }[direction]

    if model in (
        "resnet50", "convnext_base", "convnext_large"
    ):
        result = run_vision(
            model, cut, precision, first, second
        )
    else:
        result = run_text(
            model, cut, precision, first, second
        )

    result.update({
        "protocol": PROTOCOL,
        "model": model,
        "cut_after_stage": cut,
        "precision": precision,
        "direction": direction,
        "first_device": first,
        "second_device": second,
        "concurrent_pipeline": False,
        "sustained_throughput_measured": False,
        "power_measured": False,
        "energy_measured": False,
        "quality_generalization_established": False,
    })

    save_atomic(target, result)

def identity(model, cut, precision, direction):
    obj = {
        "protocol": PROTOCOL,
        "model": model,
        "cut_after_stage": cut,
        "precision": precision,
        "direction": direction,
    }
    encoded = json.dumps(
        obj, sort_keys=True,
        separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]

def initialize():
    verify_sources()
    assert REGISTRY.is_file()

    with REGISTRY.open(newline="") as f:
        registry = list(csv.DictReader(f))

    assert len(registry) == 144
    assert len({
        (row["model"], int(row["cut_after_stage"]))
        for row in registry
    }) == 144

    RUN.mkdir(exist_ok=True)

    if DB.exists():
        with sqlite3.connect(
            f"file:{DB}?mode=ro", uri=True
        ) as conn:
            assert conn.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"
            assert conn.execute(
                "SELECT COUNT(*) FROM cases"
            ).fetchone()[0] == 576
        print("PLAN: EXISTS AND VALIDATED")
        return

    conn = sqlite3.connect(DB)

    try:
        conn.execute("""
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                cut_after_stage INTEGER NOT NULL,
                precision TEXT NOT NULL,
                direction TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                result_path TEXT,
                result_sha256 TEXT,
                note TEXT
            )
        """)

        for row in registry:
            model = row["model"]
            cut = int(row["cut_after_stage"])

            for direction in ("cpu_mps", "mps_cpu"):
                for precision in ("fp32", "fp16"):
                    excluded = (
                        model == "qwen2_5_1_5b"
                        and direction == "cpu_mps"
                        and precision == "fp16"
                    )

                    conn.execute("""
                        INSERT INTO cases (
                            case_id, model, cut_after_stage,
                            precision, direction, status, note
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        identity(model, cut, precision, direction),
                        model, cut, precision, direction,
                        "excluded_not_run" if excluded else "pending",
                        (
                            "G2 confirmed first CPU block FP16 QK "
                            "overflow; other cuts not remeasured"
                            if excluded else None
                        ),
                    ))

        conn.commit()

        assert conn.execute(
            "SELECT COUNT(*) FROM cases"
        ).fetchone()[0] == 576

        assert conn.execute(
            "SELECT COUNT(*) FROM cases "
            "WHERE status='excluded_not_run'"
        ).fetchone()[0] == 27

    finally:
        conn.close()

    print("PLAN: CREATED")

def status():
    assert DB.is_file(), "Run plan first"

    with sqlite3.connect(
        f"file:{DB}?mode=ro", uri=True
    ) as conn:
        assert conn.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        print("\nG4 NUMERICAL STATUS")
        for state, count in conn.execute("""
            SELECT status, COUNT(*)
            FROM cases GROUP BY status
            ORDER BY status
        """):
            print(state, count)

        print("TOTAL:", conn.execute(
            "SELECT COUNT(*) FROM cases"
        ).fetchone()[0])

def stop_process(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

def execute(mode, limit):
    initialize()
    assert limit > 0

    lock_path = RUN / ".runner.lock"
    lock_fd = os.open(
        lock_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)

    try:
        conn = sqlite3.connect(DB)
        conn.row_factory = sqlite3.Row

        try:
            if mode == "smoke":
                target_query = """
                    SELECT * FROM cases
                    WHERE status='pending'
                      AND direction='mps_cpu'
                      AND (
                        (model='resnet50' AND cut_after_stage=2)
                        OR (model LIKE 'convnext_%'
                            AND cut_after_stage=20)
                        OR (model='distilgpt2'
                            AND cut_after_stage=1)
                        OR (model LIKE 'qwen%'
                            AND cut_after_stage=10)
                      )
                    ORDER BY model, precision
                    LIMIT ?
                """
            else:
                target_query = """
                    SELECT * FROM cases
                    WHERE status='pending'
                    ORDER BY model, direction,
                             precision, cut_after_stage
                    LIMIT ?
                """

            queue = conn.execute(
                target_query, (limit,)
            ).fetchall()

            print(
                "G4 QUEUE:", mode,
                "SELECTED:", len(queue),
                flush=True,
            )

            for index, case in enumerate(queue, 1):
                case_id = case["case_id"]
                attempt = case["attempts"] + 1

                folder = (
                    RUN / "runs" / case_id
                    / f"attempt_{attempt:03d}"
                )
                folder.mkdir(parents=True, exist_ok=False)

                result = folder / "result.json"
                stdout = folder / "stdout.txt"
                stderr = folder / "stderr.txt"

                conn.execute("""
                    UPDATE cases SET attempts=?
                    WHERE case_id=? AND status='pending'
                """, (attempt, case_id))
                conn.commit()

                print(
                    f"[{index}/{len(queue)}]",
                    case["model"],
                    "cut", case["cut_after_stage"],
                    case["precision"],
                    case["direction"],
                    flush=True,
                )

                env = {
                    **os.environ,
                    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }

                with stdout.open("x") as out, \
                     stderr.open("x") as err:
                    proc = subprocess.Popen(
                        [
                            sys.executable, "-u", str(
                                Path(__file__).resolve()
                            ),
                            "worker",
                            case["model"],
                            str(case["cut_after_stage"]),
                            case["precision"],
                            case["direction"],
                            str(result),
                        ],
                        stdout=out,
                        stderr=err,
                        env=env,
                        start_new_session=True,
                    )

                    try:
                        rc = proc.wait(timeout=1200)
                    except KeyboardInterrupt:
                        stop_process(proc)
                        print("INTERRUPTED: RESUMABLE")
                        break
                    except subprocess.TimeoutExpired:
                        stop_process(proc)
                        rc = 124

                if rc != 0 or not result.is_file():
                    err_tail = stderr.read_text(
                        errors="replace"
                    )[-1200:]

                    conn.execute("""
                        UPDATE cases
                        SET status='execution_failed', note=?
                        WHERE case_id=?
                    """, (f"exit={rc}; {err_tail}", case_id))
                    conn.commit()

                    print("FAILED — CAMPAIGN PAUSED")
                    print(err_tail[-600:])
                    break

                obj = json.loads(result.read_text())
                assert (
                    obj["protocol"] == PROTOCOL
                    and obj["model"] == case["model"]
                    and obj["cut_after_stage"]
                        == case["cut_after_stage"]
                    and obj["precision"] == case["precision"]
                    and obj["direction"] == case["direction"]
                ), ("Case/result mismatch", case_id)

                verdict = obj["numerical_status"]
                assert verdict in (
                    "numerical_pass",
                    "numerical_mismatch",
                    "numerical_invalid",
                )

                conn.execute("""
                    UPDATE cases
                    SET status=?, result_path=?,
                        result_sha256=?, note=NULL
                    WHERE case_id=?
                """, (
                    verdict,
                    str(result.relative_to(ROOT)),
                    digest(result),
                    case_id,
                ))
                conn.commit()

                print(
                    "COMMITTED:", verdict,
                    "logical_bytes=",
                    obj["logical_activation_bytes"],
                    flush=True,
                )
        finally:
            conn.close()
    finally:
        lock_path.unlink(missing_ok=True)

    status()

def main():
    assert ROOT.is_dir()
    command = sys.argv[1] if len(sys.argv) > 1 else ""

    if command == "plan":
        initialize()
        status()
    elif command == "status":
        status()
    elif command in ("smoke", "run"):
        limit = (
            int(sys.argv[2])
            if len(sys.argv) > 2
            else (12 if command == "smoke" else 549)
        )
        execute(command, limit)
    elif command == "worker":
        worker(
            sys.argv[2],
            int(sys.argv[3]),
            sys.argv[4],
            sys.argv[5],
            Path(sys.argv[6]).resolve(),
        )
    else:
        raise SystemExit(
            "Usage: g4_numeric.py "
            "plan | status | smoke | run [limit]"
        )

if __name__ == "__main__":
    main()
