import time
import csv
import subprocess
import statistics
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"

REPEATS = 3
WORKLOAD_DURATION = 30
COOLDOWN = 15
IDLE_DURATION = 8
SAMPLE_INTERVAL = 0.5

PROMPT = "Artificial intelligence on edge devices can"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64


def read_gpu():
    cmd = [
        "nvidia-smi",
        "--query-gpu=power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits"
    ]

    out = subprocess.check_output(
        cmd,
        text=True
    ).strip()

    power, util, mem = out.split(",")

    return (
        float(power.strip()),
        float(util.strip()),
        float(mem.strip())
    )


def sample_idle(duration):
    samples = []

    start = time.perf_counter()

    while time.perf_counter() - start < duration:
        try:
            samples.append(read_gpu())
        except Exception:
            pass

        time.sleep(SAMPLE_INTERVAL)

    return samples


device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL)
model.to(device)
model.eval()

inputs = tokenizer(
    [PROMPT] * BATCH_SIZE,
    return_tensors="pt",
    padding=True
)

inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}


print("GPU:", torch.cuda.get_device_name(0))


# Warm-up
with torch.inference_mode():
    model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

torch.cuda.synchronize()


results = []


for repeat in range(1, REPEATS + 1):

    print(f"\n--- Repeat {repeat}/{REPEATS} ---")

    print(f"Cooling down {COOLDOWN}s...")
    time.sleep(COOLDOWN)

    print("Measuring idle GPU power...")
    idle_samples = sample_idle(IDLE_DURATION)

    idle_power = statistics.mean(
        x[0] for x in idle_samples
    )

    print(f"Idle power: {idle_power:.2f} W")

    workload_samples = []
    stop_event = threading.Event()

    def sampler():
        while not stop_event.is_set():

            try:
                workload_samples.append(
                    read_gpu()
                )
            except Exception:
                pass

            time.sleep(SAMPLE_INTERVAL)

    sampler_thread = threading.Thread(
        target=sampler
    )

    sampler_thread.start()

    torch.cuda.synchronize()

    start = time.perf_counter()
    runs = 0

    while time.perf_counter() - start < WORKLOAD_DURATION:

        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()

        runs += 1

    elapsed = time.perf_counter() - start

    stop_event.set()
    sampler_thread.join()

    powers = [x[0] for x in workload_samples]
    utils = [x[1] for x in workload_samples]
    memories = [x[2] for x in workload_samples]

    avg_power = statistics.mean(powers)
    avg_util = statistics.mean(utils)
    avg_memory = statistics.mean(memories)

    dynamic_power = max(
        avg_power - idle_power,
        0
    )

    runs_per_sec = runs / elapsed

    tokens_per_sec = (
        runs *
        BATCH_SIZE *
        MAX_NEW_TOKENS
    ) / elapsed

    energy_per_run = (
        avg_power *
        elapsed /
        runs
    )

    dynamic_energy_per_run = (
        dynamic_power *
        elapsed /
        runs
    )

    print(
        f"runs={runs} | "
        f"{runs_per_sec:.3f} run/s | "
        f"{tokens_per_sec:.1f} tok/s"
    )

    print(
        f"power={avg_power:.2f} W | "
        f"idle={idle_power:.2f} W | "
        f"util={avg_util:.1f}%"
    )

    print(
        f"VRAM={avg_memory:.0f} MB | "
        f"energy/run={energy_per_run:.2f} J | "
        f"dynamic={dynamic_energy_per_run:.2f} J"
    )

    results.append([
        repeat,
        runs,
        elapsed,
        runs_per_sec,
        tokens_per_sec,
        idle_power,
        avg_power,
        dynamic_power,
        avg_util,
        avg_memory,
        energy_per_run,
        dynamic_energy_per_run
    ])


with open(
    "results/msi_cuda_repeated_power.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "repeat",
        "runs",
        "duration_sec",
        "runs_per_sec",
        "tokens_per_sec",
        "idle_power_w",
        "avg_power_w",
        "dynamic_power_w",
        "gpu_util_percent",
        "vram_mb",
        "energy_per_run_j",
        "dynamic_energy_per_run_j"
    ])

    writer.writerows(results)


print("\n======== MEDIAN SUMMARY ========")

print(
    f"CUDA | "
    f"{statistics.median(r[3] for r in results):.3f} run/s | "
    f"{statistics.median(r[4] for r in results):.1f} tok/s | "
    f"{statistics.median(r[6] for r in results):.2f} W | "
    f"{statistics.median(r[8] for r in results):.1f}% util | "
    f"{statistics.median(r[10] for r in results):.2f} J/run | "
    f"dynamic {statistics.median(r[11] for r in results):.2f} J/run"
)

print("\nSaved: results/msi_cuda_repeated_power.csv")
