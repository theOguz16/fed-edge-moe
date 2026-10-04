import os
import time
import statistics
import subprocess
import torch
from torchvision.models import resnet50, ResNet50_Weights

torch.set_num_threads(4)

RES = 224
BATCHES = [8, 10, 12, 14, 16]
REPEATS = 3
ITERATIONS = 10
WARMUP = 3

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

def rss_mb():
    out = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        text=True
    )
    return int(out.strip()) / 1024

for batch in BATCHES:
    x = torch.randn(batch, 3, RES, RES)

    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)

    vals = []

    for r in range(1, REPEATS + 1):
        times = []

        with torch.inference_mode():
            for _ in range(ITERATIONS):
                t0 = time.perf_counter()
                model(x)
                times.append(time.perf_counter() - t0)

        latency = statistics.median(times)
        throughput = batch / latency
        vals.append(throughput)

        print(
            f"B{batch:2} R{r} | "
            f"{throughput:7.2f} img/s | "
            f"{latency*1000/batch:7.2f} ms/img | "
            f"RSS {rss_mb():7.0f} MB"
        )

    print(
        f"MEDIAN B{batch:2} | "
        f"{statistics.median(vals):.2f} img/s"
    )

    del x
