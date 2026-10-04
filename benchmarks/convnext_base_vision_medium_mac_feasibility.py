import csv
import gc
import statistics
import time

import torch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

RESOLUTION = 224
BATCH = 1
WARMUP = 5
REPEATS = 5
CPU_THREADS = 4

OUT = "results/convnext_base_vision_medium_mac_feasibility.csv"

torch.set_num_threads(CPU_THREADS)

weights = ConvNeXt_Base_Weights.DEFAULT

def sync(device):
    if device == "mps":
        torch.mps.synchronize()

def mps_mem_mb():
    if not torch.backends.mps.is_available():
        return None, None
    return (
        torch.mps.current_allocated_memory() / 1024**2,
        torch.mps.driver_allocated_memory() / 1024**2,
    )

def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()

def run_case(backend, dtype):
    cleanup()

    device = torch.device(backend)

    model = convnext_base(weights=weights)
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
            t1 = time.perf_counter()

            latencies.append(t1 - t0)

    med_sec = statistics.median(latencies)
    throughput = BATCH / med_sec

    alloc_mb = None
    driver_mb = None

    if backend == "mps":
        alloc_mb, driver_mb = mps_mem_mb()

    result = {
        "model": "convnext_base",
        "params": sum(p.numel() for p in model.parameters()),
        "backend": backend,
        "precision": str(dtype).replace("torch.", ""),
        "resolution": RESOLUTION,
        "batch": BATCH,
        "cpu_threads": CPU_THREADS if backend == "cpu" else "",
        "median_latency_ms": med_sec * 1000,
        "throughput_img_s": throughput,
        "mps_allocated_mb": alloc_mb,
        "mps_driver_mb": driver_mb,
        "status": "PASS",
    }

    del x
    del model
    cleanup()

    return result

cases = [
    ("cpu", torch.float32),
]

if torch.backends.mps.is_available():
    cases += [
        ("mps", torch.float32),
        ("mps", torch.float16),
    ]

rows = []

print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print()

for backend, dtype in cases:
    name = f"{backend.upper()} / {str(dtype).replace('torch.', '')}"

    try:
        r = run_case(backend, dtype)
        rows.append(r)

        mem = ""
        if r["mps_allocated_mb"] is not None:
            mem = (
                f" | alloc {r['mps_allocated_mb']:.1f} MB"
                f" | driver {r['mps_driver_mb']:.1f} MB"
            )

        print(
            f"{name:16} | "
            f"{r['median_latency_ms']:8.3f} ms | "
            f"{r['throughput_img_s']:8.2f} img/s"
            f"{mem}"
        )

    except Exception as e:
        rows.append({
            "model": "convnext_base",
            "params": "",
            "backend": backend,
            "precision": str(dtype).replace("torch.", ""),
            "resolution": RESOLUTION,
            "batch": BATCH,
            "cpu_threads": CPU_THREADS if backend == "cpu" else "",
            "median_latency_ms": "",
            "throughput_img_s": "",
            "mps_allocated_mb": "",
            "mps_driver_mb": "",
            "status": f"FAIL: {type(e).__name__}: {e}",
        })

        print(f"{name:16} | FAIL | {type(e).__name__}: {e}")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nSaved:", OUT)
