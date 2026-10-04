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
    "batch_heavy":    (224, 16),
    "highres_batch":  (320, 16),
}

rows = []

for name, (resolution, batch) in WORKLOADS.items():

    print(f"\n{'='*70}")
    print(name, resolution, batch)
    print("="*70)

    for threads in THREADS:

        torch.set_num_threads(threads)

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()

        x = torch.randn(
            batch, 3, resolution, resolution,
            dtype=torch.float32
        )

        with torch.inference_mode():
            for _ in range(WARMUP):
                model(x)

        repeat_values = []

        for repeat in range(1, REPEATS + 1):

            times = []

            with torch.inference_mode():
                for _ in range(ITERATIONS):
                    start = time.perf_counter()
                    model(x)
                    times.append(time.perf_counter() - start)

            latency = statistics.median(times)
            throughput = batch / latency
            ms_image = latency * 1000 / batch

            rows.append({
                "workload": name,
                "resolution": resolution,
                "batch": batch,
                "threads": threads,
                "repeat": repeat,
                "latency_sec": latency,
                "ms_per_image": ms_image,
                "throughput_images_sec": throughput,
            })

            repeat_values.append(throughput)

            print(
                f"{threads:2}t R{repeat} | "
                f"{throughput:8.2f} img/s | "
                f"{ms_image:7.3f} ms/img"
            )

        print(
            f"MEDIAN {threads:2}t | "
            f"{statistics.median(repeat_values):.2f} img/s"
        )

        del model, x

outfile = "results/resnet50_vision_easy_mac_cpu_scaling.csv"

with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", outfile)
