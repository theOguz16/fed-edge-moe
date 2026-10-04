import csv
import gc
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

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
INTERVAL_MS = 500

OUT = Path("results/convnext_base_vision_medium_mac_mps_power.csv")

POWER_RE = re.compile(
    r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW"
)

if not torch.backends.mps.is_available():
    raise RuntimeError("MPS unavailable")

subprocess.run(["sudo", "-n", "-v"], check=True)

device = torch.device("mps")
weights = ConvNeXt_Base_Weights.DEFAULT

rows = []

def sync():
    torch.mps.synchronize()

def cleanup():
    gc.collect()
    torch.mps.empty_cache()
    sync()

def run_for_duration(model, x, batch, seconds):
    count = 0

    sync()
    start = time.perf_counter()

    with torch.inference_mode():
        while True:
            _ = model(x)
            count += batch

            if count % (batch * 4) == 0:
                sync()
                if time.perf_counter() - start >= seconds:
                    break

    sync()
    elapsed = time.perf_counter() - start

    return count / elapsed, elapsed, count

for precision, dtype in PRECISIONS:

    cleanup()

    model = convnext_base(weights=weights)
    model.eval()
    model = model.to(device=device, dtype=dtype)

    for profile, (res, batch) in PROFILES.items():

        x = torch.randn(
            batch, 3, res, res,
            device=device,
            dtype=dtype,
        )

        # Warm-up before both A/B measurements.
        with torch.inference_mode():
            for _ in range(5):
                _ = model(x)
        sync()

        for repeat in range(1, REPEATS + 1):

            # A: no monitor
            no_thr, _, _ = run_for_duration(
                model, x, batch, DURATION_SEC
            )

            # Small stabilization gap outside measurement.
            time.sleep(1)

            power_file = Path(
                f"/tmp/convnext_{precision}_{profile}_r{repeat}.txt"
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

                mon_thr, mon_elapsed, mon_images = run_for_duration(
                    model, x, batch, DURATION_SEC
                )

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
                    f"No power samples: {precision}/{profile}/R{repeat}"
                )

            mean_power = statistics.mean(powers)

            energy_img = mean_power / mon_thr
            energy_batch = energy_img * batch

            delta_pct = (
                (mon_thr - no_thr) / no_thr * 100.0
            )

            allocated_mb = (
                torch.mps.current_allocated_memory() / 1024**2
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
                "energy_batch_j": energy_batch,
                "energy_img_j": energy_img,
                "mps_allocated_mb": allocated_mb,
                "power_samples": len(powers),
            })

            print(
                f"{precision:4} {profile:6} R{repeat} | "
                f"NO {no_thr:7.2f} | "
                f"MON {mon_thr:7.2f} | "
                f"{delta_pct:+6.2f}% | "
                f"{mean_power:6.2f} W | "
                f"{energy_img:.4f} J/img"
            )

            power_file.unlink(missing_ok=True)

        del x
        cleanup()

    del model
    cleanup()

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS")
print("-" * 120)

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
            f"{statistics.median(x['energy_img_j'] for x in r):.4f} J/img | "
            f"{statistics.median(x['mps_allocated_mb'] for x in r):.1f} MB"
        )

print("\nSaved:", OUT)
