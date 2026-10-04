import csv
import statistics
from collections import defaultdict

MAC = "results/resnet50_vision_easy_mps_sweep.csv"
MSI = "results/resnet50_vision_easy_msi_cuda_sweep.csv"
OUT = "results/resnet50_vision_easy_cross_device.csv"

def load(path, prefix):
    d = defaultdict(list)

    with open(path) as f:
        for r in csv.DictReader(f):
            cfg = r["config"]

            if "fp32" in cfg:
                precision = "fp32"
            elif "fp16" in cfg:
                precision = "fp16"
            else:
                continue

            if r.get("status", "PASS") != "PASS":
                continue

            key = (
                precision,
                int(r["resolution"]),
                int(r["batch"]),
            )

            d[key].append(float(r["throughput_images_sec"]))

    return {
        k: statistics.median(v)
        for k, v in d.items()
    }

mac = load(MAC, "mac")
msi = load(MSI, "msi")

rows = []
wins = {"Mac MPS": 0, "MSI CUDA": 0}

for key in sorted(mac):
    if key not in msi:
        continue

    precision, res, batch = key
    a = mac[key]
    b = msi[key]

    if a > b:
        winner = "Mac MPS"
        speedup = a / b
    else:
        winner = "MSI CUDA"
        speedup = b / a

    wins[winner] += 1

    rows.append({
        "precision": precision,
        "resolution": res,
        "batch": batch,
        "mac_mps_img_s": round(a, 2),
        "msi_cuda_img_s": round(b, 2),
        "winner": winner,
        "winner_speedup": round(speedup, 3),
    })

    print(
        f"{precision} {res}/B{batch:<2} | "
        f"Mac {a:7.2f} | MSI {b:7.2f} | "
        f"{winner} {speedup:.2f}x"
    )

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nWINS")
print("Mac MPS :", wins["Mac MPS"])
print("MSI CUDA:", wins["MSI CUDA"])
print("\nSaved:", OUT)
