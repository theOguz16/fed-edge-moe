import csv
import os
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torchvision.models import resnet50, ResNet50_Weights

THREADS = [1, 4, 10]
REPEATS = 3
DURATION = 30

WORKLOADS = {
    "highres_single": (320, 1),
    "batch_optimal":  (224, 10),
}

POWER_RE = re.compile(
    r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW"
)

OUT = Path("results/resnet50_vision_easy_mac_cpu_power.csv")
rows = []

subprocess.run(["sudo", "-n", "-v"], check=True)

for workload, (resolution, batch) in WORKLOADS.items():

    for threads in THREADS:

        torch.set_num_threads(threads)

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.eval()

        x = torch.randn(
            batch, 3, resolution, resolution,
            dtype=torch.float32
        )

        with torch.inference_mode():
            for _ in range(10):
                model(x)

        for repeat in range(1, REPEATS + 1):

            power_file = Path(
                f"results/resnet50_cpu_{workload}_{threads}t_r{repeat}_power.txt"
            )

            with power_file.open("w") as pf:
                pm = subprocess.Popen(
                    [
                        "sudo", "-n", "powermetrics",
                        "--samplers", "cpu_power",
                        "-i", "100",
                    ],
                    stdout=pf,
                    stderr=subprocess.STDOUT,
                )

                time.sleep(0.3)

                runs = 0
                images = 0
                start = time.perf_counter()

                with torch.inference_mode():
                    while time.perf_counter() - start < DURATION:
                        model(x)
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

            row = {
                "workload": workload,
                "resolution": resolution,
                "batch": batch,
                "threads": threads,
                "repeat": repeat,
                "throughput_images_sec": throughput,
                "avg_power_w": avg_power,
                "energy_run_j": energy_run,
                "energy_image_j": energy_image,
                "samples": len(powers),
            }

            rows.append(row)

            print(
                f"{workload:15} {threads:2}t R{repeat} | "
                f"{throughput:7.2f} img/s | "
                f"{avg_power:6.2f} W | "
                f"{energy_image:.4f} J/image"
            )

        del model, x


with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


print("\nMEDIANS")
print("-" * 85)

for workload in WORKLOADS:
    for threads in THREADS:
        r = [
            x for x in rows
            if x["workload"] == workload
            and x["threads"] == threads
        ]

        print(
            f"{workload:15} {threads:2}t | "
            f"{statistics.median(x['throughput_images_sec'] for x in r):7.2f} img/s | "
            f"{statistics.median(x['avg_power_w'] for x in r):6.2f} W | "
            f"{statistics.median(x['energy_run_j'] for x in r):7.3f} J/run | "
            f"{statistics.median(x['energy_image_j'] for x in r):.4f} J/image"
        )

print("\nSaved:", OUT)
