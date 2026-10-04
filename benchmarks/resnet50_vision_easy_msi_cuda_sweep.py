import csv
import time
import statistics
import torch
from torchvision.models import resnet50, ResNet50_Weights

RESOLUTIONS = [160, 224, 320]
BATCHES = [1, 4, 16]
REPEATS = 3
WARMUP = 10
ITERATIONS = 50

CONFIGS = [
    ("cuda_fp32", torch.float32),
    ("cuda_fp16", torch.float16),
]

rows = []

for config_name, dtype in CONFIGS:

    print(f"\n{'='*70}")
    print(config_name)
    print("="*70)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to(device="cuda", dtype=dtype)
    model.eval()

    for resolution in RESOLUTIONS:
        for batch in BATCHES:

            try:
                x = torch.randn(
                    batch, 3, resolution, resolution,
                    device="cuda",
                    dtype=dtype
                )

                with torch.inference_mode():
                    for _ in range(WARMUP):
                        model(x)

                torch.cuda.synchronize()

                repeat_values = []

                for repeat in range(1, REPEATS + 1):
                    torch.cuda.reset_peak_memory_stats()
                    times = []

                    with torch.inference_mode():
                        for _ in range(ITERATIONS):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()

                            model(x)

                            torch.cuda.synchronize()
                            times.append(time.perf_counter() - t0)

                    latency = statistics.median(times)
                    throughput = batch / latency
                    ms_image = latency * 1000 / batch

                    alloc_mb = torch.cuda.memory_allocated() / 1024**2
                    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

                    rows.append({
                        "config": config_name,
                        "precision": str(dtype),
                        "resolution": resolution,
                        "batch": batch,
                        "repeat": repeat,
                        "latency_sec": latency,
                        "ms_per_image": ms_image,
                        "throughput_images_sec": throughput,
                        "vram_allocated_mb": alloc_mb,
                        "vram_peak_mb": peak_mb,
                        "status": "PASS",
                    })

                    repeat_values.append(throughput)

                    print(
                        f"{resolution:3} B{batch:2} R{repeat} | "
                        f"{throughput:8.2f} img/s | "
                        f"{ms_image:7.3f} ms/img | "
                        f"{peak_mb:7.1f} MB"
                    )

                print(
                    f"MEDIAN {resolution:3} B{batch:2} | "
                    f"{statistics.median(repeat_values):.2f} img/s"
                )

                del x
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"OOM {resolution} B{batch}")

                rows.append({
                    "config": config_name,
                    "precision": str(dtype),
                    "resolution": resolution,
                    "batch": batch,
                    "repeat": 0,
                    "latency_sec": "",
                    "ms_per_image": "",
                    "throughput_images_sec": "",
                    "vram_allocated_mb": "",
                    "vram_peak_mb": "",
                    "status": "OOM",
                })

                torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()


outfile = "results/resnet50_vision_easy_msi_cuda_sweep.csv"

with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", outfile)
