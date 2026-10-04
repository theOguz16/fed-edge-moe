import csv
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path

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

POWER_RE = re.compile(
    r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW"
)

OUT = Path("results/resnet50_vision_easy_mps_power_500ms.csv")
rows = []

subprocess.run(["sudo", "-n", "-v"], check=True)

def sync():
    torch.mps.synchronize()

for precision_name, dtype in PRECISIONS.items():

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = model.to(device="mps", dtype=dtype)
    model.eval()

    for profile, (resolution, batch) in PROFILES.items():

        x = torch.randn(
            batch, 3, resolution, resolution,
            device="mps",
            dtype=dtype,
        )

        with torch.inference_mode():
            for _ in range(10):
                model(x)

        sync()

        for repeat in range(1, REPEATS + 1):

            power_file = Path(
                f"results/resnet50_mps_{precision_name}_{profile}_r{repeat}_power.txt"
            )

            with power_file.open("w") as pf:
                pm = subprocess.Popen(
                    [
                        "sudo", "-n", "powermetrics",
                        "--samplers", "cpu_power",
                        "-i", "500",
                    ],
                    stdout=pf,
                    stderr=subprocess.STDOUT,
                )

                time.sleep(0.3)

                runs = 0
                images = 0

                sync()
                start = time.perf_counter()

                with torch.inference_mode():
                    while time.perf_counter() - start < DURATION:
                        model(x)
                        sync()

                        runs += 1
                        images += batch

                elapsed = time.perf_counter() - start

                try:
                    pm.send_signal(signal.SIGINT)
                    pm.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pm.terminate()
                    pm.wait()

            powers = [
                float(v) / 1000.0
                for v in POWER_RE.findall(
                    power_file.read_text(errors="replace")
                )
            ]

            if len(powers) > 10:
                powers = powers[2:-2]

            if not powers:
                raise RuntimeError("No power samples parsed")

            throughput = images / elapsed
            avg_power = statistics.mean(powers)

            energy_total = avg_power * elapsed
            energy_run = energy_total / runs
            energy_image = energy_total / images

            try:
                memory_mb = (
                    torch.mps.current_allocated_memory()
                    / 1024**2
                )
            except Exception:
                memory_mb = float("nan")

            row = {
                "precision": precision_name,
                "profile": profile,
                "resolution": resolution,
                "batch": batch,
                "repeat": repeat,
                "throughput_images_sec": throughput,
                "avg_power_w": avg_power,
                "energy_run_j": energy_run,
                "energy_image_j": energy_image,
                "mps_allocated_mb": memory_mb,
                "samples": len(powers),
            }

            rows.append(row)

            print(
                f"{precision_name:4} {profile:6} R{repeat} | "
                f"{throughput:7.2f} img/s | "
                f"{avg_power:6.2f} W | "
                f"{energy_image:.4f} J/image | "
                f"{memory_mb:.1f} MB"
            )

        del x
        torch.mps.empty_cache()

    del model
    torch.mps.empty_cache()


with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


print("\nMEDIANS")
print("-" * 100)

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
            f"{statistics.median(x['avg_power_w'] for x in r):6.2f} W | "
            f"{statistics.median(x['energy_run_j'] for x in r):7.3f} J/run | "
            f"{statistics.median(x['energy_image_j'] for x in r):.4f} J/image | "
            f"{statistics.median(x['mps_allocated_mb'] for x in r):.1f} MB"
        )

print("\nSaved:", OUT)
