import csv
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

THREADS = [1, 4, 8, 10]
REPEATS = 3
WARMUP = 2
INTERVAL_MS = 500

WORKLOADS = {
    "highres_single": (320, 1),
    "batch_workload": (224, 8),
}

OUT = Path("results/convnext_large_vision_hard_mac_cpu_power.csv")

POWER_RE = re.compile(
    r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW"
)

subprocess.run(["sudo", "-n", "-v"], check=True)

model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
model.eval()

rows = []

with torch.inference_mode():

    for workload, (res, batch) in WORKLOADS.items():
        for threads in THREADS:

            torch.set_num_threads(threads)
            x = torch.randn(batch, 3, res, res)

            for _ in range(WARMUP):
                _ = model(x)

            for repeat in range(1, REPEATS + 1):

                power_file = Path(
                    f"/tmp/convnext_{workload}_{threads}t_r{repeat}.txt"
                )

                with power_file.open("w") as pf:
                    pm = subprocess.Popen(
                        [
                            "sudo", "-n", "powermetrics",
                            "--samplers", "cpu_power",
                            "-i", str(INTERVAL_MS),
                        ],
                        stdout=pf,
                        stderr=subprocess.STDOUT,
                    )

                    time.sleep(0.15)

                    t0 = time.perf_counter()
                    _ = model(x)
                    dt = time.perf_counter() - t0

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

                if not powers:
                    raise RuntimeError(
                        f"No power samples: {workload} {threads}t R{repeat}"
                    )

                mean_power = statistics.mean(powers)
                throughput = batch / dt
                energy_run = mean_power * dt
                energy_img = energy_run / batch

                rows.append({
                    "workload": workload,
                    "resolution": res,
                    "batch": batch,
                    "threads": threads,
                    "repeat": repeat,
                    "latency_sec": dt,
                    "throughput_img_s": throughput,
                    "mean_power_w": mean_power,
                    "energy_run_j": energy_run,
                    "energy_img_j": energy_img,
                    "samples": len(powers),
                })

                print(
                    f"{workload:14} {threads:2}t R{repeat} | "
                    f"{throughput:6.2f} img/s | "
                    f"{mean_power:6.2f} W | "
                    f"{energy_img:.4f} J/img | "
                    f"{len(powers)} samples"
                )

                power_file.unlink(missing_ok=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS")
print("-" * 100)

for workload in WORKLOADS:
    for threads in THREADS:
        r = [
            x for x in rows
            if x["workload"] == workload and x["threads"] == threads
        ]

        print(
            f"{workload:14} {threads:2}t | "
            f"{statistics.median(x['throughput_img_s'] for x in r):6.2f} img/s | "
            f"{statistics.median(x['mean_power_w'] for x in r):6.2f} W | "
            f"{statistics.median(x['energy_run_j'] for x in r):7.2f} J/run | "
            f"{statistics.median(x['energy_img_j'] for x in r):.4f} J/img"
        )

print("\nSaved:", OUT)
