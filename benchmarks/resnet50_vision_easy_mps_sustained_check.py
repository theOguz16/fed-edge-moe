import time
import statistics
import torch
from torchvision.models import resnet50, ResNet50_Weights

TESTS = [
    ("fp32_light", torch.float32, 160, 1),
    ("fp16_heavy", torch.float16, 320, 16),
]

DURATION = 20
REPEATS = 3

def sync():
    torch.mps.synchronize()

for name, dtype, res, batch in TESTS:
    vals = []

    for repeat in range(1, REPEATS + 1):
        torch.mps.empty_cache()

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = model.to("mps", dtype=dtype).eval()

        x = torch.randn(
            batch, 3, res, res,
            device="mps", dtype=dtype
        )

        with torch.inference_mode():
            for _ in range(20):
                model(x)
        sync()

        images = 0
        start = time.perf_counter()

        with torch.inference_mode():
            while time.perf_counter() - start < DURATION:
                model(x)
                sync()
                images += batch

        elapsed = time.perf_counter() - start
        throughput = images / elapsed
        vals.append(throughput)

        print(f"{name:12} R{repeat} | {throughput:.2f} img/s")

        del model, x
        torch.mps.empty_cache()

    print(f"MEDIAN {name:12} | {statistics.median(vals):.2f} img/s")
