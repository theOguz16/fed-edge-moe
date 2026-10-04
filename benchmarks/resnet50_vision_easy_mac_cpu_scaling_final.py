import csv
import time
import statistics
import torch
from torchvision.models import resnet50, ResNet50_Weights

THREADS = [1, 2, 4, 6, 8, 10]
REPEATS = 3
WARMUP = 5
ITERATIONS = 20

WORKLOADS = {
    "highres_single": (320, 1),
    "batch_optimal":  (224, 10),
}

rows = []

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

for name, (resolution, batch) in WORKLOADS.items():

    print(f"\n{name} {resolution} B{batch}")

    for threads in THREADS:
        torch.set_num_threads(threads)

        x = torch.randn(
            batch, 3, resolution, resolution,
            dtype=torch.float32
        )

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
            ms_img = latency * 1000 / batch

            rows.append({
                "workload": name,
                "resolution": resolution,
                "batch": batch,
                "threads": threads,
                "repeat": repeat,
                "latency_sec": latency,
                "ms_per_image": ms_img,
                "throughput_images_sec": throughput,
            })

            vals.append(throughput)

        print(
            f"MEDIAN {threads:2}t | "
            f"{statistics.median(vals):7.2f} img/s"
        )

        del x

outfile = "results/resnet50_vision_easy_mac_cpu_scaling_final.csv"

with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", outfile)
