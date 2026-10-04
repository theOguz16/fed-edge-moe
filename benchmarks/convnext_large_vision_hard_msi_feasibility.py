import csv
import gc
import statistics
import time

import torch
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

RESOLUTION = 224
BATCH = 1
WARMUP = 3
REPEATS = 5
CPU_THREADS = 4

OUT = "results/convnext_large_vision_hard_msi_feasibility.csv"

torch.set_num_threads(CPU_THREADS)

weights = ConvNeXt_Large_Weights.DEFAULT

def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_case(backend, dtype):
    cleanup()

    device = torch.device(backend)

    if backend == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = convnext_large(weights=weights)
    model.eval()
    model = model.to(device=device, dtype=dtype)

    x = torch.randn(
        BATCH, 3, RESOLUTION, RESOLUTION,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(x)

        sync(backend)

        latencies = []

        for _ in range(REPEATS):
            sync(backend)

            t0 = time.perf_counter()
            _ = model(x)
            sync(backend)

            latencies.append(time.perf_counter() - t0)

    med_sec = statistics.median(latencies)

    alloc_mb = ""
    peak_mb = ""

    if backend == "cuda":
        alloc_mb = torch.cuda.memory_allocated() / 1024**2
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    result = {
        "model": "convnext_large",
        "params": sum(p.numel() for p in model.parameters()),
        "backend": backend,
        "precision": str(dtype).replace("torch.", ""),
        "resolution": RESOLUTION,
        "batch": BATCH,
        "cpu_threads": CPU_THREADS if backend == "cpu" else "",
        "median_latency_ms": med_sec * 1000,
        "throughput_img_s": BATCH / med_sec,
        "cuda_allocated_mb": alloc_mb,
        "cuda_peak_mb": peak_mb,
        "status": "PASS",
    }

    del x
    del model
    cleanup()

    return result

cases = [
    ("cpu", torch.float32),
]

if torch.cuda.is_available():
    cases += [
        ("cuda", torch.float32),
        ("cuda", torch.float16),
    ]

rows = []

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()

for backend, dtype in cases:
    name = f"{backend.upper()} / {str(dtype).replace('torch.', '')}"

    try:
        r = run_case(backend, dtype)
        rows.append(r)

        mem = ""

        if backend == "cuda":
            mem = (
                f" | alloc {r['cuda_allocated_mb']:.1f} MB"
                f" | peak {r['cuda_peak_mb']:.1f} MB"
            )

        print(
            f"{name:16} | "
            f"{r['median_latency_ms']:9.3f} ms | "
            f"{r['throughput_img_s']:8.2f} img/s"
            f"{mem}"
        )

    except Exception as e:
        rows.append({
            "model": "convnext_large",
            "params": "",
            "backend": backend,
            "precision": str(dtype).replace("torch.", ""),
            "resolution": RESOLUTION,
            "batch": BATCH,
            "cpu_threads": CPU_THREADS if backend == "cpu" else "",
            "median_latency_ms": "",
            "throughput_img_s": "",
            "cuda_allocated_mb": "",
            "cuda_peak_mb": "",
            "status": f"FAIL: {type(e).__name__}: {e}",
        })

        print(
            f"{name:16} | FAIL | "
            f"{type(e).__name__}: {e}"
        )

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nSaved:", OUT)
