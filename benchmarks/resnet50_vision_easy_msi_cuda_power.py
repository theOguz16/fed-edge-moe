import csv
import time
import threading
import statistics

import torch
import pynvml
from torchvision.models import resnet50, ResNet50_Weights

DURATION = 30
REPEATS = 3
POLL_SEC = 0.5

PROFILES = {
    "light":  (160, 1),
    "medium": (224, 4),
    "heavy":  (320, 16),
}

PRECISIONS = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)

# ---- Idle baseline ----
idle = []
for _ in range(30):
    idle.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
    time.sleep(1)

idle_median = statistics.median(idle)
idle_mean = statistics.mean(idle)

print(f"IDLE MEDIAN : {idle_median:.2f} W")
print(f"IDLE MEAN   : {idle_mean:.2f} W")

rows = []

for precision_name, dtype in PRECISIONS.items():

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to("cuda", dtype=dtype).eval()

    for profile, (res, batch) in PROFILES.items():

        x = torch.randn(
            batch, 3, res, res,
            device="cuda",
            dtype=dtype
        )

        with torch.inference_mode():
            for _ in range(20):
                model(x)
        torch.cuda.synchronize()

        for repeat in range(1, REPEATS + 1):

            powers = []
            utils = []
            nvml_vram = []
            temps = []
            stop = False

            torch.cuda.reset_peak_memory_stats()

            def monitor():
                while not stop:
                    try:
                        powers.append(
                            pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                        )
                        utils.append(
                            pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                        )
                        nvml_vram.append(
                            pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024**2
                        )
                        temps.append(
                            pynvml.nvmlDeviceGetTemperature(
                                h, pynvml.NVML_TEMPERATURE_GPU
                            )
                        )
                    except Exception:
                        pass

                    time.sleep(POLL_SEC)

            t = threading.Thread(target=monitor)
            t.start()

            images = 0
            runs = 0

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                while time.perf_counter() - start < DURATION:
                    model(x)
                    torch.cuda.synchronize()

                    runs += 1
                    images += batch

            elapsed = time.perf_counter() - start

            stop = True
            t.join()

            if not powers:
                raise RuntimeError("No NVML samples")

            throughput = images / elapsed
            avg_power = statistics.mean(powers)

            energy_total = avg_power * elapsed
            j_run = energy_total / runs
            j_image = energy_total / images

            torch_peak = (
                torch.cuda.max_memory_allocated() / 1024**2
            )

            rows.append({
                "precision": precision_name,
                "profile": profile,
                "resolution": res,
                "batch": batch,
                "repeat": repeat,
                "throughput_images_sec": throughput,
                "avg_gpu_power_w": avg_power,
                "dynamic_gpu_power_w": avg_power - idle_median,
                "energy_run_j": j_run,
                "energy_image_j": j_image,
                "gpu_util_percent": statistics.mean(utils),
                "nvml_vram_mb": statistics.mean(nvml_vram),
                "torch_peak_vram_mb": torch_peak,
                "gpu_temp_c": statistics.mean(temps),
                "samples": len(powers),
            })

            print(
                f"{precision_name:4} {profile:6} R{repeat} | "
                f"{throughput:7.2f} img/s | "
                f"{avg_power:5.2f} W | "
                f"{statistics.mean(utils):5.1f}% | "
                f"{j_image:.4f} J/img"
            )

        del x
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()


outfile = "results/resnet50_vision_easy_msi_cuda_power.csv"

with open(outfile, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)


print("\nMEDIANS")
print("-" * 110)

for precision_name in PRECISIONS:
    for profile in PROFILES:

        r = [
            x for x in rows
            if x["precision"] == precision_name
            and x["profile"] == profile
        ]

        print(
            f"{precision_name:4} {profile:6} | "
            f"{statistics.median(x['throughput_images_sec'] for x in r):7.2f} img/s | "
            f"{statistics.median(x['avg_gpu_power_w'] for x in r):5.2f} W | "
            f"{statistics.median(x['gpu_util_percent'] for x in r):5.1f}% | "
            f"{statistics.median(x['torch_peak_vram_mb'] for x in r):7.1f} MB | "
            f"{statistics.median(x['gpu_temp_c'] for x in r):4.1f} C | "
            f"{statistics.median(x['energy_image_j'] for x in r):.4f} J/image"
        )

print("\nSaved:", outfile)

pynvml.nvmlShutdown()
