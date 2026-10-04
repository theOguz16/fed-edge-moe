import csv
import statistics
import time

import torch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

THREADS = [1, 2, 4, 8, 16]
REPEATS = 3
WARMUP = 2

WORKLOADS = {
    "highres_single": (320, 1),
    "batched_load":   (224, 4),
}

OUT = "results/convnext_base_vision_medium_msi_cpu_scaling.csv"

model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
model.eval()

rows = []

with torch.inference_mode():

    for workload, (res, batch) in WORKLOADS.items():

        print(f"\n{'='*70}")
        print(f"{workload}: {res}x{res} / B{batch}")
        print("="*70)

        for threads in THREADS:

            torch.set_num_threads(threads)

            x = torch.randn(batch, 3, res, res)

            for _ in range(WARMUP):
                _ = model(x)

            vals = []
            latencies = []

            for repeat in range(1, REPEATS + 1):

                t0 = time.perf_counter()
                _ = model(x)
                dt = time.perf_counter() - t0

                throughput = batch / dt

                vals.append(throughput)
                latencies.append(dt)

                rows.append({
                    "workload": workload,
                    "resolution": res,
                    "batch": batch,
                    "threads": threads,
                    "repeat": repeat,
                    "latency_sec": dt,
                    "throughput_img_s": throughput,
                })

            print(
                f"{threads:2}t | "
                f"{statistics.median(vals):7.2f} img/s | "
                f"{statistics.median(latencies)*1000:8.2f} ms"
            )

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nSaved:", OUT)
