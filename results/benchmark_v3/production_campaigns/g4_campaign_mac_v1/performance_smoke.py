import hashlib
import json
import os
import queue
import re
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"
OUT = (
    BASE / "production_campaigns/g4_campaign_mac_v1"
    / "performance_smoke_resnet50_v1"
)
SOURCE = (
    BASE / "production_campaigns/g4_campaign_mac_v1"
    / "source_snapshots/vision_pilot.py"
)

assert not OUT.exists(), "Previous smoke exists; refusing overwrite"
assert SOURCE.is_file()

import psutil
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from urllib.parse import urlparse

assert torch.backends.mps.is_available()
assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
assert subprocess.run(
    ["sudo", "-n", "true"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
).returncode == 0, "Run sudo -v before this test"

torch.set_num_threads(4)
torch.manual_seed(2026)
OUT.mkdir(parents=True)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    assert not path.exists()
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)

t05 = BASE / "validation/t05_20261015T173344Z_2438/metadata.json"
holdout = (
    BASE / "holdout_thread_confirmation"
    / "t12_20261015T192923Z_2438/holdout_images.json"
)
images_dir = (
    ROOT / "data/imagenetv2"
    / "imagenetv2-matched-frequency-format-val"
)

weights = ResNet50_Weights.IMAGENET1K_V2
checkpoint = (
    Path(torch.hub.get_dir()) / "checkpoints"
    / Path(urlparse(weights.url).path).name
)

assert checkpoint.is_file()
assert sha(checkpoint) == json.loads(
    t05.read_text()
)["models"]["resnet50"]["weight_sha256"]

records = json.loads(holdout.read_text())
tensors = []
image_hashes = []

for item in records[:2]:
    path = images_dir / item["path"]
    assert sha(path) == item["sha256"]
    with Image.open(path) as image:
        tensors.append(
            weights.transforms()(image.convert("RGB"))
        )
    image_hashes.append(item["sha256"])

inputs = torch.stack(tensors)

model = resnet50(weights=None).eval()
state = torch.load(
    checkpoint, map_location="cpu", weights_only=True
)
model.load_state_dict(state, strict=True)
del state

with torch.inference_mode():
    reference = model(inputs).detach().cpu()

stage1 = nn.Sequential(
    model.conv1, model.bn1, model.relu,
    model.maxpool, model.layer1
).eval().cpu()

stage2 = nn.Sequential(
    model.layer2, model.layer3, model.layer4,
    model.avgpool, nn.Flatten(1), model.fc
).eval().to("mps")

torch.mps.synchronize()

def compute_first():
    start = time.perf_counter_ns()
    with torch.inference_mode():
        hidden = stage1(inputs)
    end = time.perf_counter_ns()
    return hidden, (end - start) / 1e6

def compute_second(hidden):
    start = time.perf_counter_ns()
    with torch.inference_mode():
        transfer_start = time.perf_counter_ns()
        hidden = hidden.to("mps", copy=True)
        torch.mps.synchronize()
        transfer_end = time.perf_counter_ns()

        output = stage2(hidden)
        torch.mps.synchronize()
        finish = time.perf_counter_ns()

        output = output.float().cpu()
        torch.mps.synchronize()

    return output, (
        (transfer_end - transfer_start) / 1e6,
        (finish - transfer_end) / 1e6,
    )

def validate(out):
    assert tuple(out.shape) == tuple(reference.shape)
    assert bool(torch.isfinite(out).all())
    assert torch.allclose(
        out, reference, atol=1e-3, rtol=1e-3
    )
    assert bool(
        torch.equal(out.argmax(-1), reference.argmax(-1))
    )

# Warm both execution stages before power measurement.
for _ in range(2):
    h, _ = compute_first()
    o, _ = compute_second(h)
    validate(o)

def one_round(mode, count=4):
    first_ms, transfer_ms, second_ms = [], [], []
    checked = [0]
    failures = []

    start = time.perf_counter_ns()

    if mode == "sequential":
        for _ in range(count):
            h, a = compute_first()
            o, (b, c) = compute_second(h)
            validate(o)
            checked[0] += 1
            first_ms.append(a)
            transfer_ms.append(b)
            second_ms.append(c)
    else:
        q = queue.Queue(maxsize=2)
        cancelled = threading.Event()
        sentinel = object()

        def producer():
            try:
                with torch.inference_mode():
                    for _ in range(count):
                        if cancelled.is_set():
                            break
                        h, a = compute_first()
                        while not cancelled.is_set():
                            try:
                                q.put((h, a), timeout=0.2)
                                break
                            except queue.Full:
                                continue
            except BaseException as exc:
                failures.append(repr(exc))
                cancelled.set()
            finally:
                while True:
                    try:
                        q.put(sentinel, timeout=0.2)
                        break
                    except queue.Full:
                        if cancelled.is_set():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                pass

        def consumer():
            try:
                with torch.inference_mode():
                    while True:
                        item = q.get(timeout=120)
                        if item is sentinel:
                            break
                        h, a = item
                        o, (b, c) = compute_second(h)
                        validate(o)
                        checked[0] += 1
                        first_ms.append(a)
                        transfer_ms.append(b)
                        second_ms.append(c)
            except BaseException as exc:
                failures.append(repr(exc))
                cancelled.set()

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join(timeout=180)
        t2.join(timeout=180)

        assert not t1.is_alive() and not t2.is_alive(), (
            "Worker threads timed out"
        )
        assert not failures, failures

    torch.mps.synchronize()
    wall_s = (time.perf_counter_ns() - start) / 1e9
    assert checked[0] == count, checked

    return {
        "images": checked[0] * len(inputs),
        "wall_s": wall_s,
        "first_ms": first_ms,
        "transfer_ms": transfer_ms,
        "second_ms": second_ms,
    }

def measure(mode):
    power_path = OUT / ("powermetrics_" + mode + ".txt")
    stop = threading.Event()
    rss_peak = [0]
    mps_current_peak = [0]
    mps_driver_peak = [0]
    observer_errors = []

    process = psutil.Process()

    def observe():
        while not stop.is_set():
            try:
                rss_peak[0] = max(
                    rss_peak[0], process.memory_info().rss
                )
                mps_current_peak[0] = max(
                    mps_current_peak[0],
                    torch.mps.current_allocated_memory()
                )
                mps_driver_peak[0] = max(
                    mps_driver_peak[0],
                    torch.mps.driver_allocated_memory()
                )
            except Exception as exc:
                observer_errors.append(repr(exc))
                stop.set()
            stop.wait(0.05)

    rounds = []
    with power_path.open("x") as power_file:
        sensor = subprocess.Popen(
            [
                "sudo", "-n", "/usr/bin/powermetrics",
                "--samplers", "cpu_power,gpu_power",
                "-i", "1000", "-n", "2000",
            ],
            stdout=power_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        monitor = threading.Thread(target=observe)
        monitor.start()

        try:
            time.sleep(1.2)
            elapsed = 0.0
            while elapsed < 7.0:
                round_data = one_round(mode)
                rounds.append(round_data)
                elapsed += round_data["wall_s"]
                assert elapsed < 120, "Measurement too slow"
            time.sleep(0.5)
        finally:
            stop.set()
            monitor.join(timeout=5)
            if sensor.poll() is None:
                os.killpg(sensor.pid, signal.SIGTERM)
            try:
                sensor.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(sensor.pid, signal.SIGKILL)
                sensor.wait()

    assert not observer_errors, observer_errors

    raw = power_path.read_text(errors="replace")
    cpu_samples = [
        float(x) / 1000
        for x in re.findall(
            r"(?m)^\s*CPU Power:\s*([\d.]+)\s*mW",
            raw,
        )
    ]
    gpu_samples = [
        float(x) / 1000
        for x in re.findall(
            r"(?m)^\s*GPU Power:\s*([\d.]+)\s*mW",
            raw,
        )
    ]
    assert len(cpu_samples) >= 3, (
        "Insufficient CPU power samples", len(cpu_samples)
    )
    assert len(gpu_samples) >= 3, (
        "Insufficient GPU power samples", len(gpu_samples)
    )

    total_images = sum(r["images"] for r in rounds)
    total_wall = sum(r["wall_s"] for r in rounds)
    throughput = total_images / total_wall
    cpu_w = statistics.mean(cpu_samples)
    gpu_w = statistics.mean(gpu_samples)

    return {
        "mode": mode,
        "pipeline_concurrent_threads":
            mode == "pipeline",
        "producer_consumer_queue_maxsize":
            2 if mode == "pipeline" else None,
        "rounds": len(rounds),
        "images": total_images,
        "measurement_wall_seconds": total_wall,
        "throughput_images_per_second": throughput,
        "round_wall_median_ms": statistics.median(
            r["wall_s"] * 1000 for r in rounds
        ),
        "stage1_median_ms": statistics.median(
            x for r in rounds for x in r["first_ms"]
        ),
        "transfer_median_ms": statistics.median(
            x for r in rounds for x in r["transfer_ms"]
        ),
        "stage2_median_ms": statistics.median(
            x for r in rounds for x in r["second_ms"]
        ),
        "cpu_power_w": cpu_w,
        "gpu_power_w": gpu_w,
        "estimated_soc_energy_j_per_image":
            (cpu_w + gpu_w) / throughput,
        "cpu_power_samples": len(cpu_samples),
        "gpu_power_samples": len(gpu_samples),
        "rss_peak_sampled_bytes": rss_peak[0],
        "mps_current_peak_sampled_bytes":
            mps_current_peak[0],
        "mps_driver_peak_sampled_bytes":
            mps_driver_peak[0],
        "power_file": str(power_path.relative_to(ROOT)),
        "power_sha256": sha(power_path),
        "numerical_status": "numerical_pass",
    }

results = []

for mode in ("sequential", "pipeline"):
    print("MEASURING:", mode, flush=True)
    results.append(measure(mode))
    print(
        "THROUGHPUT:",
        round(results[-1]["throughput_images_per_second"], 3),
        "images/s",
        "TRANSFER MEDIAN:",
        round(results[-1]["transfer_median_ms"], 4),
        "ms",
        flush=True,
    )

sequential, pipeline = results
speedup = (
    pipeline["throughput_images_per_second"]
    / sequential["throughput_images_per_second"]
)
energy_ratio = (
    pipeline["estimated_soc_energy_j_per_image"]
    / sequential["estimated_soc_energy_j_per_image"]
)

report = {
    "protocol": "G4_RESNET50_PERFORMANCE_SMOKE_V1",
    "model": "resnet50",
    "precision": "fp32",
    "cut_after_stage": 2,
    "direction": "cpu_mps",
    "microbatch_images": 2,
    "checkpoint_sha256": sha(checkpoint),
    "image_sha256_list": image_hashes,
    "synchronization": "torch.mps.synchronize",
    "transfer_scope": "explicit_tensor_copy_cpu_to_mps",
    "activation_bytes": None,
    "power_scope": "estimated_cpu_gpu_soc_not_wall",
    "energy_method":
        "average_sampled_cpu_plus_gpu_power_over_throughput",
    "power_window_exactly_aligned": False,
    "memory_scope": "sampled_process_rss_and_mps_allocator",
    "physical_bus_bytes_measured": False,
    "gpu_kernel_overlap_proven": False,
    "qos_applied": False,
    "modes": results,
    "pipeline_speedup": speedup,
    "pipeline_over_sequential_energy_ratio":
        energy_ratio,
    "publication_ready": False,
    "notes": [
        "This is an end-to-end instrumentation smoke, not a publication measurement.",
        "Fixed two-image microbatch; repeated execution.",
        "Both modes checked against full FP32 CPU reference.",
        "Power sampling windows include setup and tail time.",
        "No QoS experiment or GPU kernel overlap proof."
    ],
}

write_json(OUT / "result.json", report)

print("\nG4 PERFORMANCE SMOKE: PASS")
print("SEQUENTIAL IMAGES/S:",
      round(sequential["throughput_images_per_second"], 3))
print("PIPELINE IMAGES/S:",
      round(pipeline["throughput_images_per_second"], 3))
print("SPEEDUP:", round(speedup, 4))
print("ENERGY RATIO:", round(energy_ratio, 4))
print("POWER SAMPLES:", [
    (r["cpu_power_samples"], r["gpu_power_samples"])
    for r in results
])
print("OUTPUT:", OUT)
print("GRAPH SQLITE MODIFIED: NO")
print("OLD BENCHMARKS MODIFIED: NO")
