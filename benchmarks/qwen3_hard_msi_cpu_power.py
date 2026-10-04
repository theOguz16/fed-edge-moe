import csv
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

THREADS = [1, 4, 8, 16]
REPEATS = 3
POLL_SEC = 2.0

WORKLOADS = {
    "long_single": (128, 128, 1),
    "batch":       (128, 64, 4),
}

EXE = r"tools\llama-cuda\bin\llama-batched-bench.exe"
MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"
OUT = Path("results/qwen3_hard_q4_msi_cpu_power.csv")

LHM_URL = "http://localhost:8085/data.json"
CPU_POWER_ID = "/intelcpu/0/power/0"

ROW_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)

def get_json():
    with urllib.request.urlopen(LHM_URL, timeout=3) as r:
        return json.load(r)

def find_sensor(node, sensor_id):
    if isinstance(node, dict):
        if node.get("SensorId") == sensor_id:
            return node
        for v in node.values():
            found = find_sensor(v, sensor_id)
            if found:
                return found
    elif isinstance(node, list):
        for x in node:
            found = find_sensor(x, sensor_id)
            if found:
                return found
    return None

def cpu_power_w():
    node = find_sensor(get_json(), CPU_POWER_ID)
    if node is None:
        raise RuntimeError("CPU Package power sensor bulunamadı")

    value = str(node["Value"])
    value = value.replace(" W", "").replace(",", ".").strip()
    return float(value)

# LHM sanity
print(f"LHM CPU Package: {cpu_power_w():.2f} W")

rows = []

for workload, (pp, tg, npl) in WORKLOADS.items():
    for threads in THREADS:
        for repeat in range(1, REPEATS + 1):

            cmd = [
                EXE,
                "-hf", MODEL,
                "-ngl", "0",
                "-t", str(threads),
                "-tb", str(threads),
                "-npp", str(pp),
                "-ntg", str(tg),
                "-npl", str(npl),
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            captured = []
            powers = []
            stop = threading.Event()
            poll_thread = None

            def poll_power():
                # benchmark başlangıcından sonra ölç
                while not stop.is_set():
                    try:
                        powers.append(cpu_power_w())
                    except Exception as e:
                        print("Power poll warning:", e)

                    stop.wait(POLL_SEC)

            for line in proc.stdout:
                captured.append(line)

                # Model load bittikten sonra benchmark başlıyor.
                if poll_thread is None and "llama_batched_bench:" in line:
                    poll_thread = threading.Thread(
                        target=poll_power,
                        daemon=True
                    )
                    poll_thread.start()

            proc.wait()
            stop.set()

            if poll_thread:
                poll_thread.join(timeout=3)

            text = "".join(captured)
            matches = ROW_RE.findall(text)

            if not matches:
                print(text)
                raise RuntimeError("Benchmark row parse edilemedi")

            if not powers:
                raise RuntimeError("LHM power sample alınamadı")

            m = matches[-1]

            s_tg = float(m[7])
            t_total = float(m[8])

            median_power = statistics.median(powers)
            mean_power = statistics.mean(powers)

            generated_tokens = tg * npl

            energy_run = mean_power * t_total
            energy_token = energy_run / generated_tokens

            rows.append({
                "workload": workload,
                "threads": threads,
                "repeat": repeat,
                "s_tg_tok_s": s_tg,
                "t_total_sec": t_total,
                "median_power_w": median_power,
                "mean_power_w": mean_power,
                "energy_run_j": energy_run,
                "energy_token_j": energy_token,
                "samples": len(powers),
            })

            print(
                f"{workload:12} {threads:2}t R{repeat} | "
                f"{s_tg:7.2f} tok/s | "
                f"{mean_power:6.2f} W | "
                f"{energy_token:.4f} J/token | "
                f"{len(powers)} samples"
            )

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
            if x["workload"] == workload
            and x["threads"] == threads
        ]

        print(
            f"{workload:12} {threads:2}t | "
            f"{statistics.median(x['s_tg_tok_s'] for x in r):7.2f} tok/s | "
            f"{statistics.median(x['mean_power_w'] for x in r):6.2f} W | "
            f"{statistics.median(x['energy_run_j'] for x in r):7.2f} J/run | "
            f"{statistics.median(x['energy_token_j'] for x in r):.4f} J/token"
        )

print("\nSaved:", OUT)

