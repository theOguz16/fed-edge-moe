import ctypes
import hashlib
import json
import os
import queue
import statistics
import re
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("results/benchmark_v3").resolve()
OUT = BASE / "production_campaigns/t26c_text_pipeline"
SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub"
    / "models--distilgpt2/snapshots"
    / "2290a62682d06624634c1f46a6ad5be0f47f38aa"
)
CORPUS = BASE / (
    "text_quality_recovery/"
    "t22_2r_20261017T210547Z_2438/"
    "wikitext2_raw_test.parquet"
)
RUN_DIR = Path(os.environ["T26C_RUN_DIR"]).resolve()
assert RUN_DIR.is_dir()
DEST = RUN_DIR / "result.json"

CONTEXT = 128
BATCH = 4
MICRO = 1
CUT = 3
QOS = 0x15  # default
ROUNDS = 10

def digest(data):
    return hashlib.sha256(data).hexdigest()

def set_qos():
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    setter = lib.pthread_set_qos_class_self_np
    getter = lib.qos_class_self
    setter.argtypes = [ctypes.c_uint, ctypes.c_int]
    setter.restype = ctypes.c_int
    getter.restype = ctypes.c_uint
    actual = getter()
    rc = setter(QOS, 0)
    actual = getter()
    assert rc == 0 and actual == QOS, (
        "QOS_FAILED", rc, actual
    )
    return int(actual)

def make_causal_mask(device):
    upper_triangle = torch.triu(
        torch.ones(
            CONTEXT, CONTEXT,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )
    mask = torch.zeros(
        (1, 1, CONTEXT, CONTEXT),
        dtype=torch.float32,
        device=device,
    )
    mask.masked_fill_(
        upper_triangle,
        torch.finfo(torch.float32).min,
    )
    return mask

CAUSAL_MASK_CPU = make_causal_mask("cpu")
CAUSAL_MASK_MPS = None

def block_forward(block, hidden):
    mask = (
        CAUSAL_MASK_CPU
        if hidden.device.type == "cpu"
        else CAUSAL_MASK_MPS
    )
    assert mask is not None
    assert mask.device == hidden.device
    assert hidden.shape[1] == CONTEXT
    result = block(
        hidden,
        attention_mask=mask,
        use_cache=False,
    )
    if isinstance(result, (tuple, list)):
        return result[0]
    if hasattr(result, "last_hidden_state"):
        return result.last_hidden_state
    assert isinstance(result, torch.Tensor), type(result)
    return result

assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
assert torch.backends.mps.is_available()
assert SNAPSHOT.is_dir(), SNAPSHOT
assert CORPUS.is_file(), CORPUS
assert not DEST.exists(), (
    "Pilot result already exists; refusing overwrite"
)

torch.set_num_threads(4)
torch.manual_seed(2026)
main_qos = set_qos()

print("T26C PILOT: LOADING DISTILGPT2 FP32", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(SNAPSHOT),
    local_files_only=True,
    trust_remote_code=False,
)

raw = pq.read_table(
    CORPUS, columns=["text"]
)["text"].to_pylist()
excerpt = "\n".join(
    x.strip() for x in raw
    if isinstance(x, str) and x.strip()
)[:4000]
tokens = tokenizer.encode(
    excerpt, add_special_tokens=False
)
assert len(tokens) >= 2 * CONTEXT

seq = [
    tokens[0:CONTEXT],
    tokens[CONTEXT:2 * CONTEXT],
]
ids = torch.tensor(
    [seq[0], seq[1], seq[0], seq[1]],
    dtype=torch.long,
)
token_sha = digest(ids.numpy().tobytes())

model = AutoModelForCausalLM.from_pretrained(
    str(SNAPSHOT),
    local_files_only=True,
    trust_remote_code=False,
    torch_dtype=torch.float32,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
).eval()

assert len(model.transformer.h) == 6

with torch.inference_mode():
    ref_two = model(
        input_ids=ids[:2],
        use_cache=False,
        return_dict=True,
    ).logits.detach().cpu()
    reference = ref_two.repeat(2, 1, 1)
assert torch.isfinite(reference).all().item()

blocks = list(model.transformer.h)
wte = model.transformer.wte
wpe = model.transformer.wpe
drop = model.transformer.drop
norm = model.transformer.ln_f
head = model.lm_head

def embed(x):
    position = torch.arange(
        x.shape[1], dtype=torch.long
    ).unsqueeze(0)
    return drop(wte(x) + wpe(position))

def cpu_first(x):
    y = embed(x)
    for block in blocks[:CUT]:
        y = block_forward(block, y)
    return y

def cpu_second(y):
    for block in blocks[CUT:]:
        y = block_forward(block, y)
    return head(norm(y))

print("T26C PILOT: CPU GRAPH CHECK", flush=True)
with torch.inference_mode():
    cpu_manual = cpu_second(
        cpu_first(ids[:2])
    )
    assert torch.allclose(
        cpu_manual, ref_two,
        atol=1e-5, rtol=1e-5
    ), "CPU_MANUAL_MODEL_MISMATCH"

print("CPU GRAPH: PASS", flush=True)

# GPT-2 ties token embeddings to the LM head.
# Break this tie before moving the output head to MPS.
assert wte.weight is head.weight
head.weight = torch.nn.Parameter(
    head.weight.detach().clone(),
    requires_grad=False,
)

for block in blocks[CUT:]:
    block.to("mps")
norm.to("mps")
head.to("mps")
CAUSAL_MASK_MPS = CAUSAL_MASK_CPU.to("mps")
torch.mps.synchronize()

assert wte.weight.device.type == "cpu"
assert head.weight.device.type == "mps"
assert all(
    p.device.type == "cpu"
    for block in blocks[:CUT]
    for p in block.parameters()
)
assert all(
    p.device.type == "mps"
    for block in blocks[CUT:]
    for p in block.parameters()
)

chunks = list(ids.split(MICRO))

def stage1(i, x):
    start = time.perf_counter_ns()
    y = cpu_first(x)
    end = time.perf_counter_ns()
    return y, {
        "index": i,
        "stage1_start_ns": start,
        "stage1_end_ns": end,
    }

def stage2(i, y, event):
    # GPU transfer and every MPS operation have one owner.
    y = y.to("mps")
    start = time.perf_counter_ns()
    for block in blocks[CUT:]:
        y = block_forward(block, y)
    y = head(norm(y))
    torch.mps.synchronize()
    end = time.perf_counter_ns()
    output = y.cpu().detach().clone()
    return output, {
        **event,
        "stage2_start_ns": start,
        "stage2_end_ns": end,
        "output_ready_ns": time.perf_counter_ns(),
    }

def sequential():
    outputs = []
    events = []
    start = time.perf_counter_ns()
    with torch.inference_mode():
        for i, x in enumerate(chunks):
            y, event = stage1(i, x)
            z, event = stage2(i, y, event)
            outputs.append(z)
            events.append(event)
    end = time.perf_counter_ns()
    return torch.cat(outputs), events, start, end, {}

def pipeline():
    channel = queue.Queue(maxsize=2)
    stop = threading.Event()
    errors = []
    outputs = [None] * len(chunks)
    events = [None] * len(chunks)
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
                    i, y, event = item
                    outputs[i], events[i] = stage2(
                        i, y, event
                    )
        except BaseException:
            errors.append(traceback.format_exc())
            stop.set()

    a = threading.Thread(target=producer, daemon=True)
    b = threading.Thread(target=consumer, daemon=True)
    start = time.perf_counter_ns()
    b.start()
    a.start()
    a.join(timeout=120)
    b.join(timeout=120)
    end = time.perf_counter_ns()

    assert not a.is_alive() and not b.is_alive(), (
        "PIPELINE_THREAD_TIMEOUT"
    )
    assert not errors, "\n".join(errors)
    assert qos_actual == {
        "producer": QOS,
        "consumer": QOS,
    }, qos_actual
    assert all(x is not None for x in outputs)
    return (
        torch.cat(outputs),
        events, start, end, qos_actual
    )

def overlap_ms(events):
    ns = 0
    for a in events:
        for b in events:
            if a["index"] == b["index"]:
                continue
            left = max(
                a["stage1_start_ns"],
                b["stage2_start_ns"]
            )
            right = min(
                a["stage1_end_ns"],
                b["stage2_end_ns"]
            )
            ns += max(0, right - left)
    return ns / 1e6

measurements = {}

for mode, fn in (
    ("sequential", sequential),
    ("pipeline", pipeline),
):
    print("VALIDATING:", mode, flush=True)
    latencies = []
    overlaps = []
    max_error = 0.0

    for _ in range(2):
        warm_actual, _, _, _, _ = fn()
        assert torch.allclose(
            warm_actual, reference,
            atol=1e-3, rtol=1e-3
        ), "WARMUP_NUMERICAL_MISMATCH"

    power_path = DEST.with_name(
        "powermetrics_" + mode + "_power_pilot.txt"
    )

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
        sensor_start = time.monotonic()
        try:
            while (
                len(latencies) < ROUNDS
                or time.monotonic() - sensor_start < 10.0
            ):
                iteration = len(latencies)
                actual, events, start, end, qos_threads = fn()
                assert actual.shape == reference.shape
                assert torch.isfinite(actual).all().item()
                error = float(
                    (actual - reference).abs().max().item()
                )
                max_error = max(max_error, error)

                assert torch.allclose(
                    actual, reference,
                    atol=1e-3, rtol=1e-3
                ), f"NUMERICAL_MISMATCH {mode} max_abs={error}"

                assert torch.equal(
                    actual.argmax(dim=-1),
                    reference.argmax(dim=-1)
                ), f"TOKEN_AGREEMENT_MISMATCH {mode}"

                seconds = (end - start) / 1e9
                latencies.append(seconds)
                overlaps.append(overlap_ms(events))
        finally:
            window_seconds = time.monotonic() - sensor_start
            if sensor.poll() is None:
                sensor.send_signal(signal.SIGINT)
            try:
                sensor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sensor.kill()
                sensor.wait()

    raw = power_path.read_text(errors="replace")
    cpu_matches = list(re.finditer(
        r"CPU Power:\s*([\d.]+)\s*mW", raw, re.I
    ))
    samples = []

    for i, match in enumerate(cpu_matches):
        boundary = (
            cpu_matches[i + 1].start()
            if i + 1 < len(cpu_matches) else len(raw)
        )
        gpu = re.search(
            r"GPU Power:\s*([\d.]+)\s*mW",
            raw[match.end():boundary], re.I
        )
        if gpu:
            samples.append((
                float(match.group(1)) / 1000,
                float(gpu.group(1)) / 1000,
            ))

    if len(samples) < 2:
        raise RuntimeError(
            f"POWER_MISSING mode={mode} "
            f"samples={len(samples)} "
            f"sensor_exit={sensor.returncode} "
            f"tail={raw[-500:]!r}"
        )

    cpu_power = statistics.mean(x[0] for x in samples)
    gpu_power = statistics.mean(x[1] for x in samples)
    sustained_tokens_s = (
        BATCH * CONTEXT * len(latencies)
        / window_seconds
    )
    assert sustained_tokens_s > 0

    measurements[mode] = {
        "rounds": ROUNDS,
        "median_wall_ms": (
            statistics.median(latencies) * 1000
        ),
        "median_prefill_tokens_s": (
            BATCH * CONTEXT
            / statistics.median(latencies)
        ),
        "max_abs_error": max_error,
        "median_stage_overlap_ms": (
            statistics.median(overlaps)
        ),
        "numerical_status": "numerical_pass",
    }

    measurements[mode].update({
        "sustained_prefill_tokens_s": sustained_tokens_s,
        "cpu_power_w": cpu_power,
        "gpu_power_w": gpu_power,
        "gross_energy_j_token": (
            (cpu_power + gpu_power) / sustained_tokens_s
        ),
        "power_sample_count": len(samples),
        "power_window_s": window_seconds,
        "power_file_sha256": digest(
            power_path.read_bytes()
        ),
    })

    print(
        "PASS:", mode,
        "median_ms=",
        round(
            measurements[mode]["median_wall_ms"], 2
        ),
        flush=True
    )

speedup = (
    measurements["pipeline"]["median_prefill_tokens_s"]
    / measurements["sequential"]["median_prefill_tokens_s"]
)

result = {
    "protocol": "T26C_DISTILGPT2_PIPELINE_PILOT_V1",
    "model": "distilgpt2",
    "precision": "fp32",
    "mapping": "cpu_to_mps",
    "cut_after_block": CUT,
    "context_tokens": CONTEXT,
    "batch_size": BATCH,
    "microbatch_size": MICRO,
    "corpus": str(CORPUS),
    "input_tokens_sha256": token_sha,
    "main_qos_class": main_qos,
    "pipeline_thread_qos_verified": True,
    "native_pytorch_thread_qos_verified": False,
    "independent_process_repeats": 1,
    "power_measured": True,
    "gpu_kernel_overlap_proven": False,
    "measurements": measurements,
    "pilot_speedup": speedup,
    "sustained_speedup": (
        measurements["pipeline"]["sustained_prefill_tokens_s"]
        / measurements["sequential"]["sustained_prefill_tokens_s"]
    ),
    "pipeline_to_sequential_energy_ratio": (
        measurements["pipeline"]["gross_energy_j_token"]
        / measurements["sequential"]["gross_energy_j_token"]
    ),
}
OUT.mkdir(parents=True, exist_ok=True)
temporary = DEST.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(result, indent=2) + "\n"
)
os.replace(temporary, DEST)

print("T26C PILOT: PASS")
print("NUMERICAL: PASS sequential + pipeline")
print("PIPELINE THREAD QOS: PASS")
print(f"PILOT SPEEDUP: {speedup:.3f}x")
print("POWER: MEASURED — estimated CPU+GPU SoC")
print("SQLITE MODIFIED: NO")
print("RESULT:", DEST)
