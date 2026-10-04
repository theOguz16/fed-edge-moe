import time, statistics, torch
from torchvision.models import resnet50, ResNet50_Weights

THREADS = [1, 4, 16]
REPEATS = 3
DURATION = 30

WORKLOADS = {
    "highres_single": (320, 1),
    "batch_optimal":  (224, 4),
}

model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

for name, (res, batch) in WORKLOADS.items():
    for threads in THREADS:
        torch.set_num_threads(threads)
        x = torch.randn(batch, 3, res, res)

        with torch.inference_mode():
            for _ in range(10):
                model(x)

        vals = []

        for r in range(1, REPEATS + 1):
            images = 0
            start = time.perf_counter()

            with torch.inference_mode():
                while time.perf_counter() - start < DURATION:
                    model(x)
                    images += batch

            elapsed = time.perf_counter() - start
            tp = images / elapsed
            vals.append(tp)

            print(f"{name:15} {threads:2}t R{r} | {tp:.2f} img/s")

        print(
            f"MEDIAN {name:15} {threads:2}t | "
            f"{statistics.median(vals):.2f} img/s"
        )

        del x
