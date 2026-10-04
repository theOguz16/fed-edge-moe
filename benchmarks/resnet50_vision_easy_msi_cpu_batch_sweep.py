import time
import statistics
import torch
from torchvision.models import resnet50, ResNet50_Weights

torch.set_num_threads(16)

RESOLUTIONS = [224, 320]
BATCHES = [1, 2, 4, 8, 10, 12, 14, 16]
REPEATS = 3
WARMUP = 3
ITERATIONS = 10

model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

for res in RESOLUTIONS:
    print(f"\nRESOLUTION {res}")

    for batch in BATCHES:
        x = torch.randn(batch, 3, res, res)

        with torch.inference_mode():
            for _ in range(WARMUP):
                model(x)

        vals = []

        for _ in range(REPEATS):
            times = []

            with torch.inference_mode():
                for _ in range(ITERATIONS):
                    t0 = time.perf_counter()
                    model(x)
                    times.append(time.perf_counter() - t0)

            latency = statistics.median(times)
            vals.append(batch / latency)

        print(
            f"MEDIAN B{batch:2} | "
            f"{statistics.median(vals):.2f} img/s"
        )

        del x
