import csv, time, statistics, threading, urllib.request, json
import torch
from torchvision.models import resnet50, ResNet50_Weights

THREADS = [1, 4, 16]
REPEATS = 3
DURATION = 30
POLL_SEC = 1.0

WORKLOADS = {
    "highres_single": (320, 1),
    "batch_optimal":  (224, 4),
}

def get_json():
    with urllib.request.urlopen("http://localhost:8085/data.json", timeout=2) as r:
        return json.load(r)

def find_cpu_package_power(node):
    if isinstance(node, dict):
        if node.get("SensorId") == "/intelcpu/0/power/0":
            v = str(node.get("Value", ""))
            v = v.replace("W", "").replace(",", ".").strip()
            return float(v)

        for c in node.get("Children", []):
            x = find_cpu_package_power(c)
            if x is not None:
                return x

    return None

rows = []

model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()

for workload, (res, batch) in WORKLOADS.items():
    for threads in THREADS:
        torch.set_num_threads(threads)

        x = torch.randn(batch, 3, res, res)

        with torch.inference_mode():
            for _ in range(10):
                model(x)

        for repeat in range(1, REPEATS + 1):
            powers = []
            stop = False

            def monitor():
                while not stop:
                    try:
                        p = find_cpu_package_power(get_json())
                        if p is not None:
                            powers.append(p)
                    except:
                        pass
                    time.sleep(POLL_SEC)

            t = threading.Thread(target=monitor)
            t.start()

            images = 0
            runs = 0
            start = time.perf_counter()

            with torch.inference_mode():
                while time.perf_counter() - start < DURATION:
                    model(x)
                    runs += 1
                    images += batch

            elapsed = time.perf_counter() - start
            stop = True
            t.join()

            if not powers:
                raise RuntimeError("No CPU Package power samples found")

            throughput = images / elapsed
            avg_power = statistics.mean(powers)
            energy_total = avg_power * elapsed
            j_run = energy_total / runs
            j_image = energy_total / images

            rows.append({
                "workload": workload,
                "resolution": res,
                "batch": batch,
                "threads": threads,
                "repeat": repeat,
                "throughput_images_sec": throughput,
                "avg_cpu_package_power_w": avg_power,
                "energy_run_j": j_run,
                "energy_image_j": j_image,
                "samples": len(powers),
            })

            print(
                f"{workload:15} {threads:2}t R{repeat} | "
                f"{throughput:6.2f} img/s | "
                f"{avg_power:5.2f} W | "
                f"{j_image:.4f} J/image"
            )

        del x

outfile = "results/resnet50_vision_easy_msi_cpu_power.csv"

with open(outfile, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS")
print("-" * 85)

for workload in WORKLOADS:
    for threads in THREADS:
        r = [x for x in rows if x["workload"] == workload and x["threads"] == threads]

        print(
            f"{workload:15} {threads:2}t | "
            f"{statistics.median(x['throughput_images_sec'] for x in r):6.2f} img/s | "
            f"{statistics.median(x['avg_cpu_package_power_w'] for x in r):5.2f} W | "
            f"{statistics.median(x['energy_run_j'] for x in r):7.3f} J/run | "
            f"{statistics.median(x['energy_image_j'] for x in r):.4f} J/image"
        )

print("\nSaved:", outfile)

