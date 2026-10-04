import time
import statistics
import torch
from torchvision.models import resnet50, ResNet50_Weights

DURATION = 30
REPEATS = 3

PROFILES = {
    "light":  (160, 1),
    "medium": (224, 4),
    "heavy":  (320, 16),
}

PRECISIONS = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}

for pname, dtype in PRECISIONS.items():

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to("cuda", dtype=dtype).eval()

    for profile, (res, batch) in PROFILES.items():

        x = torch.randn(
            batch, 3, res, res,
            device="cuda", dtype=dtype
        )

        with torch.inference_mode():
            for _ in range(20):
                model(x)

        torch.cuda.synchronize()

        vals = []

        for r in range(1, REPEATS + 1):
            images = 0

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                while time.perf_counter() - start < DURATION:
                    model(x)
                    torch.cuda.synchronize()
                    images += batch

            elapsed = time.perf_counter() - start
            tp = images / elapsed
            vals.append(tp)

            print(
                f"{pname:4} {profile:6} R{r} | "
                f"{tp:.2f} img/s"
            )

        print(
            f"MEDIAN {pname:4} {profile:6} | "
            f"{statistics.median(vals):.2f} img/s"
        )

        del x
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
