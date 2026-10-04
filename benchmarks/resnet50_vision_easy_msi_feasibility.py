import time
import statistics
import torch
from torchvision.models import resnet50, ResNet50_Weights

CONFIGS = [
    ("cpu_fp32", "cpu", torch.float32),
    ("cuda_fp32", "cuda", torch.float32),
    ("cuda_fp16", "cuda", torch.float16),
]

BATCH = 1
RES = 224
WARMUP = 20
REPEATS = 100

for name, device, dtype in CONFIGS:
    print(f"`nCONFIG: {name}")

    if device == "cuda" and not torch.cuda.is_available():
        print("Status: SKIP - CUDA unavailable")
        continue

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to(device=device, dtype=dtype).eval()

    x = torch.randn(
        BATCH, 3, RES, RES,
        device=device,
        dtype=dtype
    )

    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []

    with torch.inference_mode():
        for _ in range(REPEATS):
            if device == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            model(x)

            if device == "cuda":
                torch.cuda.synchronize()

            times.append(time.perf_counter() - t0)

    latency = statistics.median(times)
    throughput = BATCH / latency

    print("Model          : ResNet50")
    print(f"Parameters     : {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"Device         : {device}")
    print(f"Precision      : {dtype}")
    print(f"Resolution     : {RES}x{RES}")
    print(f"Batch          : {BATCH}")
    print(f"Median latency : {latency*1000:.3f} ms")
    print(f"Throughput     : {throughput:.2f} images/s")

    if device == "cuda":
        print(f"VRAM allocated : {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"VRAM peak      : {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
        torch.cuda.reset_peak_memory_stats()

    print("Status         : PASS")

    del model, x

    if device == "cuda":
        torch.cuda.empty_cache()
