import csv
import gc
import statistics
import time

import torch
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

RESOLUTIONS = [160, 224, 320]
BATCHES = [1, 4, 16]
PRECISIONS = [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
]

WARMUP = 5
REPEATS = 3

OUT = "results/convnext_large_vision_hard_mac_mps_sweep.csv"

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS unavailable")

device = torch.device("mps")
weights = ConvNeXt_Large_Weights.DEFAULT

rows = []

def sync():
    torch.mps.synchronize()

def cleanup():
    gc.collect()
    torch.mps.empty_cache()
    sync()

for precision_name, dtype in PRECISIONS:

    cleanup()

    model = convnext_large(weights=weights)
    model.eval()
    model = model.to(device=device, dtype=dtype)

    params = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 72}")
    print(f"MPS {precision_name.upper()}")
    print("=" * 72)

    for res in RESOLUTIONS:
        for batch in BATCHES:

            try:
                x = torch.randn(
                    batch,
                    3,
                    res,
                    res,
                    device=device,
                    dtype=dtype,
                )

                with torch.inference_mode():

                    for _ in range(WARMUP):
                        _ = model(x)

                    sync()

                    times = []

                    for repeat in range(1, REPEATS + 1):

                        sync()
                        t0 = time.perf_counter()

                        _ = model(x)

                        sync()
                        elapsed = time.perf_counter() - t0

                        throughput = batch / elapsed
                        times.append(elapsed)

                        allocated_mb = (
                            torch.mps.current_allocated_memory()
                            / 1024**2
                        )

                        driver_mb = (
                            torch.mps.driver_allocated_memory()
                            / 1024**2
                        )

                        rows.append({
                            "model": "convnext_large",
                            "params": params,
                            "backend": "mps",
                            "precision": precision_name,
                            "resolution": res,
                            "batch": batch,
                            "repeat": repeat,
                            "latency_sec": elapsed,
                            "latency_ms": elapsed * 1000,
                            "ms_per_image": elapsed * 1000 / batch,
                            "throughput_img_s": throughput,
                            "mps_allocated_mb": allocated_mb,
                            "mps_driver_mb": driver_mb,
                            "status": "PASS",
                        })

                med = statistics.median(times)

                print(
                    f"{res:3} B{batch:<2} | "
                    f"{batch / med:8.2f} img/s | "
                    f"{med * 1000:8.2f} ms | "
                    f"{med * 1000 / batch:7.2f} ms/img | "
                    f"{allocated_mb:7.1f} MB"
                )

                del x
                cleanup()

            except Exception as e:

                print(
                    f"{res:3} B{batch:<2} | "
                    f"FAIL: {type(e).__name__}: {e}"
                )

                rows.append({
                    "model": "convnext_large",
                    "params": params,
                    "backend": "mps",
                    "precision": precision_name,
                    "resolution": res,
                    "batch": batch,
                    "repeat": "",
                    "latency_sec": "",
                    "latency_ms": "",
                    "ms_per_image": "",
                    "throughput_img_s": "",
                    "mps_allocated_mb": "",
                    "mps_driver_mb": "",
                    "status": f"FAIL: {type(e).__name__}: {e}",
                })

                cleanup()

    del model
    cleanup()

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nSaved:", OUT)
