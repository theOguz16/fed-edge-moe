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

OUT = "results/convnext_large_vision_hard_msi_cuda_sweep.csv"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

device = torch.device("cuda")
weights = ConvNeXt_Large_Weights.DEFAULT

rows = []

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

for precision_name, dtype in PRECISIONS:

    cleanup()

    model = convnext_large(weights=weights)
    model.eval()
    model = model.to(device=device, dtype=dtype)

    params = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 72}")
    print(f"CUDA {precision_name.upper()}")
    print("=" * 72)

    for res in RESOLUTIONS:
        for batch in BATCHES:

            try:
                torch.cuda.reset_peak_memory_stats()

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

                    torch.cuda.synchronize()

                    times = []

                    for repeat in range(1, REPEATS + 1):

                        torch.cuda.synchronize()
                        t0 = time.perf_counter()

                        _ = model(x)

                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - t0

                        throughput = batch / elapsed
                        times.append(elapsed)

                        alloc_mb = (
                            torch.cuda.memory_allocated()
                            / 1024**2
                        )

                        peak_mb = (
                            torch.cuda.max_memory_allocated()
                            / 1024**2
                        )

                        rows.append({
                            "model": "convnext_large",
                            "params": params,
                            "backend": "cuda",
                            "precision": precision_name,
                            "resolution": res,
                            "batch": batch,
                            "repeat": repeat,
                            "latency_sec": elapsed,
                            "latency_ms": elapsed * 1000,
                            "ms_per_image": elapsed * 1000 / batch,
                            "throughput_img_s": throughput,
                            "cuda_allocated_mb": alloc_mb,
                            "cuda_peak_mb": peak_mb,
                            "status": "PASS",
                        })

                med = statistics.median(times)

                print(
                    f"{res:3} B{batch:<2} | "
                    f"{batch / med:8.2f} img/s | "
                    f"{med * 1000:8.2f} ms | "
                    f"{med * 1000 / batch:7.2f} ms/img | "
                    f"peak {peak_mb:7.1f} MB"
                )

                del x
                cleanup()

            except torch.cuda.OutOfMemoryError as e:

                print(
                    f"{res:3} B{batch:<2} | OOM"
                )

                rows.append({
                    "model": "convnext_large",
                    "params": params,
                    "backend": "cuda",
                    "precision": precision_name,
                    "resolution": res,
                    "batch": batch,
                    "repeat": "",
                    "latency_sec": "",
                    "latency_ms": "",
                    "ms_per_image": "",
                    "throughput_img_s": "",
                    "cuda_allocated_mb": "",
                    "cuda_peak_mb": "",
                    "status": "OOM",
                })

                cleanup()

            except Exception as e:

                print(
                    f"{res:3} B{batch:<2} | "
                    f"FAIL: {type(e).__name__}: {e}"
                )

                rows.append({
                    "model": "convnext_large",
                    "params": params,
                    "backend": "cuda",
                    "precision": precision_name,
                    "resolution": res,
                    "batch": batch,
                    "repeat": "",
                    "latency_sec": "",
                    "latency_ms": "",
                    "ms_per_image": "",
                    "throughput_img_s": "",
                    "cuda_allocated_mb": "",
                    "cuda_peak_mb": "",
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

