import time
import statistics
import torch
from torchvision.models import resnet50, ResNet50_Weights

torch.set_num_threads(4)

RESOLUTIONS = [224, 320]
BATCHES = [1, 2, 4, 8, 16]
WARMUP = 5
REPEATS = 3
ITERATIONS = 10

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

for res in RESOLUTIONS:
    print(f"\nRESOLUTION {res}")

    for batch in BATCHES:
        x = torch.randn(batch, 3, res, res)

        with torch.inference_mode():
            for _ in range(WARMUP):
                model(x)

        vals = []

        for r in range(1, REPEATS + 1):
            times = []

            with torch.inference_mode():
                for _ in range(ITERATIONS):
                    start = time.perf_counter()
                    model(x)
                    times.append(time.perf_counter() - start)

            latency = statistics.median(times)
            throughput = batch / latency
            ms_img = latency * 1000 / batch

            vals.append(throughput)

            print(
                f"B{batch:2} R{r} | "
                f"{throughput:7.2f} img/s | "
                f"{ms_img:7.2f} ms/img"
            )

        print(
            f"MEDIAN B{batch:2} | "
            f"{statistics.median(vals):.2f} img/s"
        )
