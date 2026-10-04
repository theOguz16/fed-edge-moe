import time
import statistics
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

print("================================")
print("VISION EASY FEASIBILITY")
print("================================")
print("Model  : MobileNetV3-Small")
print("Device :", DEVICE)
print("Dtype  :", DTYPE)

weights = MobileNet_V3_Small_Weights.DEFAULT
model = mobilenet_v3_small(weights=weights)
model = model.to(device=DEVICE, dtype=DTYPE)
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"Params : {params/1e6:.2f} M")

x = torch.randn(
    1, 3, 224, 224,
    device=DEVICE,
    dtype=DTYPE
)

with torch.inference_mode():
    for _ in range(20):
        model(x)

sync()

times = []

with torch.inference_mode():
    for _ in range(100):
        sync()
        start = time.perf_counter()
        model(x)
        sync()
        times.append(time.perf_counter() - start)

median = statistics.median(times)

print("\nRESULT")
print(f"Median latency : {median*1000:.3f} ms")
print(f"Throughput     : {1/median:.2f} images/s")

if DEVICE == "cuda":
    print(f"VRAM allocated : {torch.cuda.memory_allocated()/1024**2:.1f} MB")
elif DEVICE == "mps":
    print(f"MPS allocated  : {torch.mps.current_allocated_memory()/1024**2:.1f} MB")

print("Status         : PASS")
