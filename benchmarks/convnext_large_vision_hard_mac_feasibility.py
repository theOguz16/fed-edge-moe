import csv
import gc
import statistics
import time

import torch
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

RESOLUTION = 224
BATCH = 1
CPU_THREADS = 4
WARMUP = 3
REPEATS = 5

OUT = "results/convnext_large_vision_hard_mac_feasibility.csv"

torch.set_num_threads(CPU_THREADS)

weights = ConvNeXt_Large_Weights.DEFAULT


def sync(backend):
    if backend == "mps":
        torch.mps.synchronize()


def cleanup():
    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


def run_case(backend, dtype):
    cleanup()

    device = torch.device(backend)

    model = convnext_large(weights=weights)
    model.eval()
    model = model.to(
        device=device,
        dtype=dtype,
    )

    params = sum(p.numel() for p in model.parameters())

    x = torch.randn(
        BATCH,
        3,
        RESOLUTION,
        RESOLUTION,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():

        for _ in range(WARMUP):
            _ = model(x)

        sync(backend)

        times = []

        for _ in range(REPEATS):
            sync(backend)

            t0 = time.perf_counter()
            _ = model(x)
            sync(backend)

            times.append(
                time.perf_counter() - t0
            )

    med_sec = statistics.median(times)

    alloc_mb = ""
    driver_mb = ""

    if backend == "mps":
        alloc_mb = (
            torch.mps.current_allocated_memory()
            / 1024**2
        )

        driver_mb = (
            torch.mps.driver_allocated_memory()
            / 1024**2
        )

    result = {
        "model": "convnext_large",
        "params": params,
        "backend": backend,
        "precision": str(dtype).replace("torch.", ""),
        "resolution": RESOLUTION,
        "batch": BATCH,
        "cpu_threads": (
            CPU_THREADS if backend == "cpu" else ""
        ),
        "median_latency_ms": med_sec * 1000,
        "throughput_img_s": BATCH / med_sec,
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

    name = (
        f"{backend.upper()} / "
        f"{str(dtype).replace('torch.', '')}"
    )

    try:
        r = run_case(backend, dtype)
        rows.append(r)

        mem = ""

        if backend == "mps":
            mem = (
                f" | alloc {r['mps_allocated_mb']:.1f} MB"
                f" | driver {r['mps_driver_mb']:.1f} MB"
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
            "cpu_threads": (
                CPU_THREADS if backend == "cpu" else ""
            ),
            "median_latency_ms": "",
            "throughput_img_s": "",
            "mps_allocated_mb": "",
            "mps_driver_mb": "",
            "status": (
                f"FAIL: {type(e).__name__}: {e}"
            ),
        })

        print(
            f"{name:16} | FAIL | "
            f"{type(e).__name__}: {e}"
        )


with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )
    w.writeheader()
    w.writerows(rows)

print("\nSaved:", OUT)
