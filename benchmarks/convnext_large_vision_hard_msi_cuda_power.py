import csv
import gc
import statistics
import threading
import time

import torch
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

try:
    import pynvml
except ImportError:
    raise RuntimeError(
        "pynvml missing. Run: .\\.venv\\Scripts\\python.exe -m pip install nvidia-ml-py"
    )

PROFILES = {
    "light":  (160, 1),
    "medium": (224, 4),
    "heavy":  (320, 16),
}

PRECISIONS = [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
]

REPEATS = 3
DURATION_SEC = 10.0
POLL_SEC = 0.5

OUT = "results/convnext_large_vision_hard_msi_cuda_power.csv"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

device = torch.device("cuda")
weights = ConvNeXt_Large_Weights.DEFAULT

rows = []

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def nvml_sample():
    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
    temp = pynvml.nvmlDeviceGetTemperature(
        handle,
        pynvml.NVML_TEMPERATURE_GPU
    )

    return power_w, util, mem, temp

def run_for_duration(model, x, batch, seconds):
    images = 0

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        while True:
            _ = model(x)
            images += batch

            if images % (batch * 4) == 0:
                torch.cuda.synchronize()

                if time.perf_counter() - start >= seconds:
                    break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return images / elapsed

# Formal idle baseline
idle_samples = []

for _ in range(30):
    idle_samples.append(nvml_sample()[0])
    time.sleep(0.1)

print(
    f"NVML idle | median "
    f"{statistics.median(idle_samples):.2f} W | "
    f"mean {statistics.mean(idle_samples):.2f} W"
)

for precision, dtype in PRECISIONS:

    cleanup()

    model = convnext_large(weights=weights)
    model.eval()
    model = model.to(device=device, dtype=dtype)

    for profile, (res, batch) in PROFILES.items():

        torch.cuda.reset_peak_memory_stats()

        x = torch.randn(
            batch, 3, res, res,
            device=device,
            dtype=dtype,
        )

        with torch.inference_mode():
            for _ in range(5):
                _ = model(x)

        torch.cuda.synchronize()

        for repeat in range(1, REPEATS + 1):

            # A: no monitor
            no_thr = run_for_duration(
                model, x, batch, DURATION_SEC
            )

            time.sleep(1)

            powers = []
            utils = []
            mems = []
            temps = []

            stop = threading.Event()

            def poll_nvml():
                while not stop.is_set():
                    try:
                        p, u, m, t = nvml_sample()
                        powers.append(p)
                        utils.append(u)
                        mems.append(m)
                        temps.append(t)
                    except Exception:
                        pass

                    stop.wait(POLL_SEC)

            thread = threading.Thread(
                target=poll_nvml,
                daemon=True,
            )

            thread.start()

            mon_thr = run_for_duration(
                model, x, batch, DURATION_SEC
            )

            stop.set()
            thread.join(timeout=2)

            if not powers:
                raise RuntimeError("No NVML samples collected")

            mean_power = statistics.mean(powers)
            mean_util = statistics.mean(utils)
            peak_mem = max(mems)
            peak_temp = max(temps)

            energy_img = mean_power / mon_thr

            delta_pct = (
                (mon_thr - no_thr) / no_thr * 100.0
            )

            peak_alloc_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
            )

            rows.append({
                "precision": precision,
                "profile": profile,
                "resolution": res,
                "batch": batch,
                "repeat": repeat,
                "no_monitor_img_s": no_thr,
                "monitored_img_s": mon_thr,
                "monitor_delta_pct": delta_pct,
                "mean_power_w": mean_power,
                "mean_gpu_util_pct": mean_util,
                "nvml_peak_used_mb": peak_mem,
                "torch_peak_alloc_mb": peak_alloc_mb,
                "peak_temp_c": peak_temp,
                "energy_img_j": energy_img,
                "samples": len(powers),
            })

            print(
                f"{precision:4} {profile:6} R{repeat} | "
                f"NO {no_thr:7.2f} | "
                f"MON {mon_thr:7.2f} | "
                f"{delta_pct:+6.2f}% | "
                f"{mean_power:6.2f} W | "
                f"{mean_util:5.1f}% | "
                f"{energy_img:.4f} J/img"
            )

        del x
        cleanup()

    del model
    cleanup()

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS")
print("-" * 135)

for precision, _ in PRECISIONS:
    for profile in PROFILES:

        r = [
            x for x in rows
            if x["precision"] == precision
            and x["profile"] == profile
        ]

        print(
            f"{precision:4} {profile:6} | "
            f"NO {statistics.median(x['no_monitor_img_s'] for x in r):7.2f} | "
            f"MON {statistics.median(x['monitored_img_s'] for x in r):7.2f} | "
            f"delta {statistics.median(x['monitor_delta_pct'] for x in r):+6.2f}% | "
            f"{statistics.median(x['mean_power_w'] for x in r):6.2f} W | "
            f"util {statistics.median(x['mean_gpu_util_pct'] for x in r):5.1f}% | "
            f"{statistics.median(x['energy_img_j'] for x in r):.4f} J/img | "
            f"temp {statistics.median(x['peak_temp_c'] for x in r):.0f} C"
        )

print("\nSaved:", OUT)

pynvml.nvmlShutdown()

