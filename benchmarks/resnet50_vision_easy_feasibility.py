import time
import statistics
import torch
from torchvision.models import resnet50, ResNet50_Weights

MODEL = "ResNet50"
RESOLUTION = 224
BATCH = 1
REPEATS = 100
WARMUP = 20

configs = [
    ("cpu_fp32", "cpu", torch.float32),
]

if torch.backends.mps.is_available():
    configs += [
        ("mps_fp32", "mps", torch.float32),
        ("mps_fp16", "mps", torch.float16),
    ]

weights = ResNet50_Weights.DEFAULT

def sync(device):
    if device == "mps":
        torch.mps.synchronize()

print("=" * 70)
print("VISION EASY - RESNET50 FEASIBILITY")
print("=" * 70)

for name, device, dtype in configs:
    print(f"\nCONFIG: {name}")

    model = resnet50(weights=weights)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    params = sum(p.numel() for p in model.parameters())

    x = torch.randn(
        BATCH, 3, RESOLUTION, RESOLUTION,
        device=device,
        dtype=dtype
    )

    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)

    sync(device)

    times = []

    with torch.inference_mode():
        for _ in range(REPEATS):
            sync(device)
            start = time.perf_counter()

            model(x)

            sync(device)
            times.append(time.perf_counter() - start)

    median = statistics.median(times)

    print(f"Model          : {MODEL}")
    print(f"Parameters     : {params / 1e6:.2f} M")
    print(f"Device         : {device}")
    print(f"Precision      : {dtype}")
    print(f"Resolution     : {RESOLUTION}x{RESOLUTION}")
    print(f"Batch          : {BATCH}")
    print(f"Median latency : {median * 1000:.3f} ms")
    print(f"ms/image       : {(median / BATCH) * 1000:.3f}")
    print(f"Throughput     : {BATCH / median:.2f} images/s")

    if device == "mps":
        try:
            mem = torch.mps.current_allocated_memory() / 1024**2
            print(f"MPS allocated  : {mem:.1f} MB")
        except Exception:
            pass

    print("Status         : PASS")

    del model, x
    if device == "mps":
        torch.mps.empty_cache()
