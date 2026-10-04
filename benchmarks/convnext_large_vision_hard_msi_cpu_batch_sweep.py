import csv
import statistics
import time

import torch
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

BATCHES = [1, 2, 4, 8]
THREADS = 16
REPEATS = 3
WARMUP = 2
RES = 224

OUT = "results/convnext_large_vision_hard_msi_cpu_batch_sweep.csv"

torch.set_num_threads(THREADS)

model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
model.eval()

rows = []

with torch.inference_mode():
    for batch in BATCHES:
        x = torch.randn(batch, 3, RES, RES)

        for _ in range(WARMUP):
            _ = model(x)

        vals = []

        for repeat in range(1, REPEATS + 1):
            t0 = time.perf_counter()
            _ = model(x)
            dt = time.perf_counter() - t0

            thr = batch / dt
            vals.append(thr)

            rows.append({
                "resolution": RES,
                "batch": batch,
                "threads": THREADS,
                "repeat": repeat,
                "latency_sec": dt,
                "throughput_img_s": thr,
            })

        print(
            f"B{batch:<2} | "
            f"{statistics.median(vals):7.2f} img/s"
        )

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nSaved:", OUT)
