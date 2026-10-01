import csv
import re
import time
import threading
import statistics
import subprocess
from pathlib import Path
import pynvml

EXE = Path(r".\tools\llama-cuda\bin\llama-batched-bench.exe")
MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"

PROFILES = {
    "light":  (128, 64, 1, 105),
    "medium": (512, 128, 2, 42),
    "heavy":  (1024, 128, 4, 21),
}

REPEATS = 3

ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
)

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

rows_out = []

def run_one(profile, ctx, out, batch, count, repeat):
    npp = ",".join([str(ctx)] * count)

    cmd = [
        str(EXE),
        "-hf", MODEL,
        "-c", "8192",
        "-b", "2048",
        "-ub", "512",
        "-ngl", "99",
        "-npp", npp,
        "-ntg", str(out),
        "-npl", str(batch),
    ]

    powers = []
    utils = []
    vrams = []
    temps = []

    started = threading.Event()
    stopped = threading.Event()

    timing = {"start": None, "end": None}

    def monitor():
        started.wait()

        timing["start"] = time.perf_counter()

        while not stopped.is_set():
            try:
                powers.append(
                    pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                )

                utils.append(
                    pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                )

                vrams.append(
                    pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
                )

                temps.append(
                    pynvml.nvmlDeviceGetTemperature(
                        handle,
                        pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass

            time.sleep(0.1)

        timing["end"] = time.perf_counter()

    print(f"\n{profile.upper()} R{repeat}")

    thread = threading.Thread(target=monitor)
    thread.start()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    bench_rows = []

    for line in proc.stdout:
        if "llama_batched_bench:" in line:
            started.set()
            print("benchmark started...")

        m = ROW_RE.match(line.strip())

        if m:
            bench_rows.append({
                "pp": int(m.group(1)),
                "tg": int(m.group(2)),
                "batch": int(m.group(3)),
                "t_pp": float(m.group(5)),
                "s_pp": float(m.group(6)),
                "t_tg": float(m.group(7)),
                "s_tg": float(m.group(8)),
            })

    rc = proc.wait()

    stopped.set()
    thread.join()

    if rc != 0:
        raise RuntimeError(f"benchmark failed: rc={rc}")

    if not bench_rows or not powers:
        raise RuntimeError("no benchmark rows or NVML samples")

    wall_sec = timing["end"] - timing["start"]

    generated_tokens = sum(
        x["tg"] * x["batch"] for x in bench_rows
    )

    avg_power = statistics.mean(powers)
    avg_util = statistics.mean(utils)
    peak_vram = max(vrams)
    peak_temp = max(temps)

    energy_total = avg_power * wall_sec

    energy_run = energy_total / len(bench_rows)
    energy_token = energy_total / generated_tokens

    median_tg = statistics.median(
        x["s_tg"] for x in bench_rows
    )

    first_tg = bench_rows[0]["s_tg"]
    last_tg = bench_rows[-1]["s_tg"]

    result = {
        "profile": profile,
        "repeat": repeat,
        "context": ctx,
        "output": out,
        "batch": batch,
        "runs": len(bench_rows),
        "wall_sec": wall_sec,
        "median_generation_tok_sec": median_tg,
        "first_generation_tok_sec": first_tg,
        "last_generation_tok_sec": last_tg,
        "avg_power_w": avg_power,
        "avg_gpu_util_percent": avg_util,
        "peak_vram_mb": peak_vram,
        "peak_temp_c": peak_temp,
        "energy_run_j": energy_run,
        "energy_token_j": energy_token,
        "samples": len(powers),
    }

    print(
        f"{median_tg:.2f} tok/s | "
        f"{avg_power:.2f} W | "
        f"{avg_util:.1f}% GPU | "
        f"{peak_vram:.0f} MB | "
        f"{peak_temp:.0f} C | "
        f"{energy_token:.4f} J/token"
    )

    return result


for profile, (ctx, out, batch, count) in PROFILES.items():
    for repeat in range(1, REPEATS + 1):
        rows_out.append(
            run_one(profile, ctx, out, batch, count, repeat)
        )

outfile = Path("results/qwen3_hard_q4_cuda_nvml_power.csv")

with outfile.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows_out[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows_out)

print("\nMEDIANS")
print("-" * 100)

for profile in PROFILES:
    r = [x for x in rows_out if x["profile"] == profile]

    print(
        f"{profile:6} | "
        f"{statistics.median(x['median_generation_tok_sec'] for x in r):7.2f} tok/s | "
        f"{statistics.median(x['avg_power_w'] for x in r):6.2f} W | "
        f"{statistics.median(x['avg_gpu_util_percent'] for x in r):5.1f}% | "
        f"{statistics.median(x['peak_vram_mb'] for x in r):6.0f} MB | "
        f"{statistics.median(x['peak_temp_c'] for x in r):4.0f} C | "
        f"{statistics.median(x['energy_run_j'] for x in r):7.2f} J/run | "
        f"{statistics.median(x['energy_token_j'] for x in r):.4f} J/token"
    )

print("\nSaved:", outfile)

pynvml.nvmlShutdown()
