import csv
import gc
import json
import statistics
import threading
import time
import urllib.request

import torch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

THREADS = [1, 8, 16]
REPEATS = 3
DURATION_SEC = 12.0
POLL_SEC = 2.0

WORKLOADS = {
    "highres_single": (320, 1),
    "batched_load":   (224, 4),
}

OUT = "results/convnext_base_vision_medium_msi_cpu_power.csv"

LHM_URL = "http://localhost:8085/data.json"
CPU_POWER_ID = "/intelcpu/0/power/0"

def get_json():
    with urllib.request.urlopen(LHM_URL, timeout=3) as r:
        return json.load(r)

def find_sensor(node, sensor_id):
    if isinstance(node, dict):
        if node.get("SensorId") == sensor_id:
            return node
        for value in node.values():
            found = find_sensor(value, sensor_id)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_sensor(value, sensor_id)
            if found:
                return found
    return None

def cpu_power_w():
    node = find_sensor(get_json(), CPU_POWER_ID)

    if node is None:
        raise RuntimeError("CPU Package power sensor not found")

    value = str(node["Value"])
    return float(
        value.replace(" W", "").replace(",", ".").strip()
    )

def run_for_duration(model, x, batch, seconds):
    images = 0
    start = time.perf_counter()

    with torch.inference_mode():
        while True:
            _ = model(x)
            images += batch

            if time.perf_counter() - start >= seconds:
                break

    elapsed = time.perf_counter() - start
    return images / elapsed

print(f"LHM CPU Package: {cpu_power_w():.2f} W")

model = convnext_base(
    weights=ConvNeXt_Base_Weights.DEFAULT
)
model.eval()

rows = []

for workload, (res, batch) in WORKLOADS.items():

    for threads in THREADS:

        torch.set_num_threads(threads)

        x = torch.randn(
            batch, 3, res, res,
            dtype=torch.float32
        )

        with torch.inference_mode():
            for _ in range(2):
                _ = model(x)

        for repeat in range(1, REPEATS + 1):

            # A: no monitor
            no_thr = run_for_duration(
                model, x, batch, DURATION_SEC
            )

            time.sleep(1)

            # B: monitored
            powers = []
            stop = threading.Event()

            def poll_power():
                while not stop.is_set():
                    try:
                        powers.append(cpu_power_w())
                    except Exception:
                        pass

                    stop.wait(POLL_SEC)

            t = threading.Thread(
                target=poll_power,
                daemon=True
            )
            t.start()

            mon_thr = run_for_duration(
                model, x, batch, DURATION_SEC
            )

            stop.set()
            t.join(timeout=3)

            if not powers:
                raise RuntimeError("No LHM power samples")

            mean_power = statistics.mean(powers)
            energy_img = mean_power / mon_thr

            delta_pct = (
                (mon_thr - no_thr) / no_thr * 100.0
            )

            rows.append({
                "workload": workload,
                "resolution": res,
                "batch": batch,
                "threads": threads,
                "repeat": repeat,
                "no_monitor_img_s": no_thr,
                "monitored_img_s": mon_thr,
                "monitor_delta_pct": delta_pct,
                "mean_power_w": mean_power,
                "energy_img_j": energy_img,
                "samples": len(powers),
            })

            print(
                f"{workload:14} {threads:2}t R{repeat} | "
                f"NO {no_thr:6.2f} | "
                f"MON {mon_thr:6.2f} | "
                f"{delta_pct:+6.2f}% | "
                f"{mean_power:6.2f} W | "
                f"{energy_img:.4f} J/img | "
                f"{len(powers)} samples"
            )

        del x
        gc.collect()

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS")
print("-" * 115)

for workload in WORKLOADS:
    for threads in THREADS:

        r = [
            x for x in rows
            if x["workload"] == workload
            and x["threads"] == threads
        ]

        print(
            f"{workload:14} {threads:2}t | "
            f"NO {statistics.median(x['no_monitor_img_s'] for x in r):6.2f} | "
            f"MON {statistics.median(x['monitored_img_s'] for x in r):6.2f} | "
            f"delta {statistics.median(x['monitor_delta_pct'] for x in r):+6.2f}% | "
            f"{statistics.median(x['mean_power_w'] for x in r):6.2f} W | "
            f"{statistics.median(x['energy_img_j'] for x in r):.4f} J/img"
        )

print("\nSaved:", OUT)
