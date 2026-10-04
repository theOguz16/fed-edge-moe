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
    ("mps_fp32", torch.float32),
    ("mps_fp16", torch.float16),
]

weights = ResNet50_Weights.DEFAULT
rows = []

def sync():
    torch.mps.synchronize()

for config_name, dtype in CONFIGS:

    print(f"\n{'='*70}")
    print(config_name)
    print("="*70)

    model = resnet50(weights=weights)
    model = model.to(device="mps", dtype=dtype)
    model.eval()

    for resolution in RESOLUTIONS:
        for batch in BATCHES:

            x = torch.randn(
                batch, 3, resolution, resolution,
                device="mps",
                dtype=dtype
            )

            with torch.inference_mode():
                for _ in range(WARMUP):
                    model(x)

            sync()

            repeat_values = []

            for repeat in range(1, REPEATS + 1):
                times = []

                with torch.inference_mode():
                    for _ in range(ITERATIONS):
                        sync()
                        start = time.perf_counter()

                        model(x)

                        sync()
                        times.append(time.perf_counter() - start)

                median_latency = statistics.median(times)
                throughput = batch / median_latency
                ms_image = median_latency * 1000 / batch

                try:
                    mem_mb = (
                        torch.mps.current_allocated_memory()
                        / 1024**2
                    )
                except Exception:
                    mem_mb = float("nan")

                rows.append({
                    "config": config_name,
                    "precision": str(dtype),
                    "resolution": resolution,
                    "batch": batch,
                    "repeat": repeat,
                    "latency_sec": median_latency,
                    "ms_per_image": ms_image,
                    "throughput_images_sec": throughput,
                    "mps_allocated_mb": mem_mb,
                })

                repeat_values.append(throughput)

                print(
                    f"{resolution:3} B{batch:2} R{repeat} | "
                    f"{throughput:8.2f} img/s | "
                    f"{ms_image:7.3f} ms/img | "
                    f"{mem_mb:7.1f} MB"
                )

            print(
                f"MEDIAN {resolution:3} B{batch:2} | "
                f"{statistics.median(repeat_values):.2f} img/s"
            )

            del x
            torch.mps.empty_cache()

    del model
    torch.mps.empty_cache()

outfile = "results/resnet50_vision_easy_mps_sweep.csv"

with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", outfile)
