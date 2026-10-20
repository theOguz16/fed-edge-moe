import ctypes
import hashlib
import json
import math
import os
import queue
import re
import signal
import statistics
import subprocess
import threading
import time
import traceback
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("results/benchmark_v3").resolve()
ROOT = BASE / "production_campaigns/t26c_text_pipeline"
CORPUS = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)
HUB = Path.home() / ".cache/huggingface/hub"
SNAPSHOTS = {
    "distilgpt2": (
        "models--distilgpt2",
        "2290a62682d06624634c1f46a6ad5be0f47f38aa",
    ),
    "qwen2_5_1_5b": (
        "models--Qwen--Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    ),
    "qwen3_1_7b": (
        "models--Qwen--Qwen3-1.7B",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    ),
}
QOS = {
    "background": 0x09,
    "utility": 0x11,
    "default": 0x15,
    "user_initiated": 0x19,
}
PROTOCOL = "T26C_UNIFIED_TEXT_PIPELINE_FP32_V2"
RUN_DIR = Path(os.environ["T26C_V2_RUN_DIR"]).resolve()
CASE = json.loads((RUN_DIR / "case.json").read_text())
DEST = RUN_DIR / "result.json"
CONTEXT = 128
BATCH = 4
MIN_ROUNDS = 3
MIN_WINDOW_SECONDS = 10.0

assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
assert torch.backends.mps.is_available()
assert CASE["protocol"] == PROTOCOL
assert CASE["precision"] == "fp32"
assert CASE["batch_size"] == BATCH
assert CASE["context_tokens"] == CONTEXT
assert CASE["microbatch_size"] in (1, 2)
assert CASE["direction"] in ("cpu_mps", "mps_cpu")
assert CASE["qos"] in QOS
assert CASE["model"] in SNAPSHOTS
assert CASE["repeat"] in (1, 2)
assert CASE["cpu_threads"] == 4
assert CORPUS.is_file()
assert not DEST.exists(), "Existing result: refusing overwrite"

model_name = CASE["model"]
cut = CASE["cut"]
micro = CASE["microbatch_size"]
first, second = (
    ("cpu", "mps")
    if CASE["direction"] == "cpu_mps"
    else ("mps", "cpu")
)
torch.set_num_threads(4)
torch.manual_seed(2026)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def set_qos():
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    setter = lib.pthread_set_qos_class_self_np
    getter = lib.qos_class_self
    setter.argtypes = [ctypes.c_uint, ctypes.c_int]
    setter.restype = ctypes.c_int
    getter.restype = ctypes.c_uint
    expected = QOS[CASE["qos"]]
    rc = setter(expected, 0)
    actual = getter()
    if rc != 0 or actual != expected:
        raise RuntimeError(
            f"QOS_FAILED expected={expected} "
            f"actual={actual} rc={rc}"
        )
    return int(actual)

def save(path, obj):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

def power_pairs(text):
    cpu = list(re.finditer(
        r"CPU Power:\s*([\d.]+)\s*mW", text, re.I
    ))
    pairs = []
    for i, match in enumerate(cpu):
        end = cpu[i + 1].start() if i + 1 < len(cpu) else len(text)
        gpu = re.search(
            r"GPU Power:\s*([\d.]+)\s*mW",
            text[match.end():end], re.I
        )
        if gpu:
            pairs.append((
                float(match.group(1)) / 1000,
                float(gpu.group(1)) / 1000,
            ))
    return pairs

main_qos = set_qos()
snapshot = HUB / SNAPSHOTS[model_name][0] / "snapshots" / SNAPSHOTS[model_name][1]
assert (snapshot / "config.json").is_file()

print("LOADING:", model_name, CASE["direction"],
      "cut=", cut, "micro=", micro, "qos=", CASE["qos"],
      flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(snapshot), local_files_only=True,
    trust_remote_code=False
)
texts = pq.read_table(CORPUS, columns=["text"])["text"].to_pylist()
excerpt = "\n".join(
    s.strip() for s in texts
    if isinstance(s, str) and s.strip()
)[:4000]
tokens = tokenizer.encode(excerpt, add_special_tokens=False)
assert len(tokens) >= 2 * CONTEXT

ids = torch.tensor([
    tokens[:CONTEXT],
    tokens[CONTEXT:2 * CONTEXT],
    tokens[:CONTEXT],
    tokens[CONTEXT:2 * CONTEXT],
], dtype=torch.long)
input_sha = digest(ids.numpy().tobytes())
chunks = list(ids.split(micro))

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
del ref_two
assert bool(torch.isfinite(reference).all())

# Separate the tied embedding/output weights before placing
# model stages on different devices.
if model_name == "distilgpt2":
    backbone = model.transformer
    blocks = list(backbone.h)
    head = model.lm_head
    embedding = backbone.wte
    assert len(blocks) == 6
    assert cut in (2, 3, 4)
else:
    backbone = model.model
    blocks = list(backbone.layers)
    head = model.lm_head
    embedding = backbone.embed_tokens
    assert len(blocks) == 28
    assert cut in (7, 14, 21)

assert embedding.weight.data_ptr() == head.weight.data_ptr()
head.weight = torch.nn.Parameter(
    head.weight.detach().clone(), requires_grad=False
)
assert embedding.weight.data_ptr() != head.weight.data_ptr()

for i, block in enumerate(blocks):
    block.to(first if i < cut else second)

if model_name == "distilgpt2":
    backbone.wte.to(first)
    backbone.wpe.to(first)
    backbone.drop.to(first)
    backbone.ln_f.to(second)
else:
    backbone.embed_tokens.to(first)
    backbone.norm.to(second)

head.to(second)

positions = {
    device: torch.arange(
        CONTEXT, dtype=torch.long, device=device
    ).unsqueeze(0)
    for device in ("cpu", "mps")
}

future = torch.triu(
    torch.ones(CONTEXT, CONTEXT, dtype=torch.bool),
    diagonal=1,
)
mask_cpu = torch.zeros(
    (1, 1, CONTEXT, CONTEXT),
    dtype=torch.float32,
).masked_fill(
    future, torch.finfo(torch.float32).min
)
masks = {
    "cpu": mask_cpu,
    "mps": mask_cpu.to("mps"),
}

rotary = {}
if model_name != "distilgpt2":
    for device in ("cpu", "mps"):
        dummy = torch.zeros(
            1, CONTEXT, model.config.hidden_size,
            dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            rotary[device] = backbone.rotary_emb(
                dummy, positions[device]
            )
        del dummy

torch.mps.synchronize()
assert embedding.weight.device.type == first
assert head.weight.device.type == second
assert all(
    next(layer.parameters()).device.type
    == (first if i < cut else second)
    for i, layer in enumerate(blocks)
)

def forward_blocks(hidden, selected):
    device = hidden.device.type
    for block in selected:
        if model_name == "distilgpt2":
            result = block(
                hidden,
                attention_mask=masks[device],
                use_cache=False,
            )
        else:
            result = block(
                hidden_states=hidden,
                attention_mask=masks[device],
                position_ids=positions[device],
                position_embeddings=rotary[device],
                use_cache=False,
            )
        hidden = (
            result[0] if isinstance(result, (tuple, list))
            else result
        )
    return hidden

def stage1(i, x):
    start = time.perf_counter_ns()
    assert x.device.type == "cpu"
    inp = x if first == "cpu" else x.to("mps")

    if model_name == "distilgpt2":
        pos = torch.arange(
            CONTEXT, device=first, dtype=torch.long
        ).unsqueeze(0)
        hidden = backbone.drop(
            backbone.wte(inp) + backbone.wpe(pos)
        )
    else:
        hidden = backbone.embed_tokens(inp)

    hidden = forward_blocks(hidden, blocks[:cut])

    if first == "mps":
        torch.mps.synchronize()
        hidden = hidden.to("cpu")
        torch.mps.synchronize()

    end = time.perf_counter_ns()
    return hidden, {
        "index": i,
        "stage1_start_ns": start,
        "stage1_end_ns": end,
    }

def stage2(i, hidden, event):
    assert hidden.device.type == "cpu"
    start = time.perf_counter_ns()
    if second == "mps":
        hidden = hidden.to("mps")

    hidden = forward_blocks(hidden, blocks[cut:])
    if model_name == "distilgpt2":
        output = head(backbone.ln_f(hidden))
    else:
        output = head(backbone.norm(hidden))

    output = output.to("cpu").detach()
    if second == "mps":
        torch.mps.synchronize()

    end = time.perf_counter_ns()
    return output, {
        **event,
        "stage2_start_ns": start,
        "stage2_end_ns": end,
    }

def sequential():
    outputs = []
    start = time.perf_counter_ns()
    with torch.inference_mode():
        for i, x in enumerate(chunks):
            hidden, event = stage1(i, x)
            out, event = stage2(i, hidden, event)
            outputs.append(out)
    end = time.perf_counter_ns()
    return torch.cat(outputs), (end - start) / 1e9, {}

def pipeline():
    channel = queue.Queue(maxsize=2)
    stop = threading.Event()
    errors = []
    outputs = [None] * len(chunks)
    qos_actual = {}

    def producer():
        try:
            qos_actual["producer"] = set_qos()
            with torch.inference_mode():
                for i, x in enumerate(chunks):
                    if stop.is_set():
                        break
                    item = (i, *stage1(i, x))
                    while not stop.is_set():
                        try:
                            channel.put(item, timeout=0.2)
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
            errors.append(traceback.format_exc())
            stop.set()

    def consumer():
        try:
            qos_actual["consumer"] = set_qos()
            with torch.inference_mode():
                while True:
                    try:
                        item = channel.get(timeout=0.2)
                    except queue.Empty:
                        if stop.is_set():
                            break
                        continue
                    if item is None:
                        break
                    i, hidden, event = item
                    outputs[i], _ = stage2(i, hidden, event)
        except BaseException:
            errors.append(traceback.format_exc())
            stop.set()

    a = threading.Thread(target=producer, daemon=True)
    b = threading.Thread(target=consumer, daemon=True)
    start = time.perf_counter_ns()
    b.start()
    a.start()
    a.join(timeout=900)
    b.join(timeout=900)
    end = time.perf_counter_ns()

    assert not a.is_alive() and not b.is_alive(), (
        "PIPELINE_THREAD_TIMEOUT"
    )
    if errors:
        raise RuntimeError("\n".join(errors))

    assert qos_actual == {
        "producer": QOS[CASE["qos"]],
        "consumer": QOS[CASE["qos"]],
    }
    assert all(x is not None for x in outputs)
    return (
        torch.cat(outputs),
        (end - start) / 1e9,
        qos_actual,
    )

def validate_output(output, mode):
    if output.shape != reference.shape:
        raise RuntimeError("NUMERICAL_MISMATCH shape " + mode)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("NUMERICAL_MISMATCH nonfinite " + mode)

    difference = float(
        (output - reference).abs().max().item()
    )
    agreement = float(
        (
            output.argmax(-1) == reference.argmax(-1)
        ).float().mean().item()
    )
    if not torch.allclose(
        output, reference, atol=1e-3, rtol=1e-3
    ):
        raise RuntimeError(
            f"NUMERICAL_MISMATCH {mode} max_abs={difference}"
        )
    if agreement != 1.0:
        raise RuntimeError(
            f"TOKEN_AGREEMENT_MISMATCH {mode}={agreement}"
        )
    return difference, agreement

measurements = {}
order = (
    ("sequential", "pipeline")
    if CASE["repeat"] == 1
    else ("pipeline", "sequential")
)

for mode in order:
    fn = sequential if mode == "sequential" else pipeline

    print("VALIDATING:", mode, flush=True)
    warm, _, actual_qos = fn()
    warm_error, warm_agreement = validate_output(warm, mode)
    del warm

    latencies = []
    power_path = RUN_DIR / f"powermetrics_{mode}_power_v2.txt"

    with power_path.open("w") as power_file:
        sensor = subprocess.Popen(
            [
                "sudo", "-n", "powermetrics",
                "--samplers", "cpu_power,gpu_power",
                "-i", "1000", "-n", "2000",
            ],
            stdout=power_file,
            stderr=subprocess.STDOUT,
            start_new_session=False,
        )
        started = time.monotonic()
        last_output = None

        try:
            while (
                len(latencies) < MIN_ROUNDS
                or time.monotonic() - started < MIN_WINDOW_SECONDS
            ):
                last_output, seconds, actual_qos = fn()
                assert math.isfinite(seconds) and seconds > 0
                latencies.append(seconds)
        finally:
            elapsed = time.monotonic() - started
            if sensor.poll() is None:
                sensor.send_signal(signal.SIGINT)
            try:
                sensor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sensor.kill()
                sensor.wait()

    last_error, last_agreement = validate_output(
        last_output, mode
    )
    del last_output

    raw = power_path.read_text(errors="replace")
    samples = power_pairs(raw)
    if len(samples) < 3:
        raise RuntimeError(
            f"POWER_MISSING mode={mode} "
            f"count={len(samples)} tail={raw[-500:]!r}"
        )

    cpu_w = statistics.mean(x[0] for x in samples)
    gpu_w = statistics.mean(x[1] for x in samples)
    assert cpu_w >= 0 and gpu_w >= 0
    assert cpu_w + gpu_w > 0
    assert elapsed >= MIN_WINDOW_SECONDS
    assert len(latencies) >= MIN_ROUNDS

    throughput = (
        len(latencies) * BATCH * CONTEXT / elapsed
    )
    assert math.isfinite(throughput) and throughput > 0

    measurements[mode] = {
        "numerical_status": "numerical_pass",
        "max_abs_error": max(warm_error, last_error),
        "token_agreement": min(warm_agreement, last_agreement),
        "timed_round_count": len(latencies),
        "timing_samples_ms": [x * 1000 for x in latencies],
        "median_wall_ms": statistics.median(latencies) * 1000,
        "sustained_prefill_tokens_s": throughput,
        "power_window_s": elapsed,
        "power_sample_count": len(samples),
        "cpu_power_w": cpu_w,
        "gpu_power_w": gpu_w,
        "gross_energy_j_token": (cpu_w + gpu_w) / throughput,
        "power_file_sha256": sha(power_path),
        "pipeline_thread_qos_verified": mode == "pipeline",
    }

    print(
        "PASS:", mode,
        f"rounds={len(latencies)}",
        f"tokens/s={throughput:.2f}",
        f"energy_J/token={measurements[mode]['gross_energy_j_token']:.6f}",
        flush=True,
    )

result = {
    "protocol": PROTOCOL,
    "case": CASE,
    "model": model_name,
    "precision": "fp32",
    "input_tokens_sha256": input_sha,
    "corpus_sha256": sha(CORPUS),
    "model_config_sha256": sha(snapshot / "config.json"),
    "worker_sha256": sha(Path(__file__)),
    "main_qos_class": main_qos,
    "pipeline_thread_qos_verified": True,
    "native_pytorch_thread_qos_verified": False,
    "gpu_kernel_overlap_proven": False,
    "power_scope": "estimated_cpu_gpu_soc_not_wall",
    "power_time_alignment": "same_run_untimestamped_samples",
    "mode_order": list(order),
    "measurements": measurements,
    "sustained_speedup": (
        measurements["pipeline"]["sustained_prefill_tokens_s"]
        / measurements["sequential"]["sustained_prefill_tokens_s"]
    ),
    "pipeline_to_sequential_energy_ratio": (
        measurements["pipeline"]["gross_energy_j_token"]
        / measurements["sequential"]["gross_energy_j_token"]
    ),
}
save(DEST, result)

print("T26C UNIFIED WORKER: PASS", flush=True)
print("RESULT:", DEST, flush=True)
