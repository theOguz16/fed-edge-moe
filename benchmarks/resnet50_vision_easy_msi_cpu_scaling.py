import csv
import time
import statistics
import torch
from torchvision.models import resnet50, ResNet50_Weights

THREADS = [1, 2, 4, 8, 16]
REPEATS = 3
WARMUP = 5
ITERATIONS = 20

WORKLOADS = {
    "highres_single": (320, 1),
    "batch_workload": (224, 10),
}

rows = []

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

for name, (res, batch) in WORKLOADS.items():
    print(f"\n{name} {res} B{batch}")

    for threads in THREADS:
        torch.set_num_threads(threads)

        x = torch.randn(batch, 3, res, res)

        with torch.inference_mode():
            for _ in range(WARMUP):
                model(x)

        vals = []

        for repeat in range(1, REPEATS + 1):
            times = []

            with torch.inference_mode():
                for _ in range(ITERATIONS):
                    t0 = time.perf_counter()
                    model(x)
                    times.append(time.perf_counter() - t0)

            latency = statistics.median(times)
            throughput = batch / latency
            vals.append(throughput)

            rows.append({
                "workload": name,
                "resolution": res,
                "batch": batch,
                "threads": threads,
                "repeat": repeat,
                "latency_sec": latency,
                "throughput_images_sec": throughput,
            })

        print(
            f"MEDIAN {threads:2}t | "
            f"{statistics.median(vals):7.2f} img/s"
        )

        del x

outfile = "results/resnet50_vision_easy_msi_cpu_scaling.csv"

with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", outfile)
