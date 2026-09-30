import csv
import time
import threading
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pynvml import (
    nvmlInit,
    nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPowerUsage,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
)

MODEL = "distilgpt2"
REPEATS = 3

COOLDOWN = 15
WORKLOAD_DURATION = 60
SAMPLE_INTERVAL = 0.1

SCENARIOS = [
    ("light", 32, 32, 1),
    ("medium", 128, 64, 4),
    ("heavy", 512, 128, 8),
]

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
model.eval()

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)


def make_inputs(ctx, batch):
    text = (
        "Artificial intelligence on edge devices enables "
        "efficient distributed machine learning systems. "
    )

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = (ids * ((ctx // len(ids)) + 2))[:ctx]

    input_ids = torch.tensor(
        [ids] * batch,
        dtype=torch.long,
        device=device
    )

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids)
    }


results = []

for name, ctx, output, batch in SCENARIOS:

    inputs = make_inputs(ctx, batch)

    print(f"\n===== {name.upper()} =====")

    for repeat in range(1, REPEATS + 1):

        print(f"Repeat {repeat}/{REPEATS}")
        print("Cooldown...")
        time.sleep(COOLDOWN)

        # warm-up
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=output,
                min_new_tokens=output,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()

        samples = []
        stop_event = threading.Event()

        def sampler():
            while not stop_event.is_set():
                try:
                    power = nvmlDeviceGetPowerUsage(handle) / 1000.0
                    util = nvmlDeviceGetUtilizationRates(handle).gpu
                    mem = nvmlDeviceGetMemoryInfo(handle).used / 1024**2

                    samples.append((power, util, mem))
                except Exception:
                    pass

                time.sleep(SAMPLE_INTERVAL)

        thread = threading.Thread(target=sampler)
        thread.start()

        torch.cuda.synchronize()

        start = time.perf_counter()
        runs = 0

        while time.perf_counter() - start < WORKLOAD_DURATION:

            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=output,
                    min_new_tokens=output,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            torch.cuda.synchronize()
            runs += 1

        elapsed = time.perf_counter() - start

        stop_event.set()
        thread.join()

        if not samples:
            raise RuntimeError("NVML power samples bulunamadı.")

        avg_power = statistics.mean(x[0] for x in samples)
        avg_util = statistics.mean(x[1] for x in samples)
        avg_mem = statistics.mean(x[2] for x in samples)

        throughput = (
            runs * batch * output
        ) / elapsed

        energy_run = (
            avg_power * elapsed
        ) / runs

        energy_token = (
            energy_run / (batch * output)
        )

        print(
            f"runs={runs} | "
            f"{throughput:.1f} tok/s | "
            f"{avg_power:.2f} W | "
            f"{avg_util:.1f}% util | "
            f"{avg_mem:.0f} MB | "
            f"{energy_run:.2f} J/run | "
            f"{energy_token:.4f} J/token | "
            f"samples={len(samples)}"
        )

        results.append([
            name,
            repeat,
            ctx,
            output,
            batch,
            runs,
            elapsed,
            throughput,
            avg_power,
            avg_util,
            avg_mem,
            energy_run,
            energy_token,
            len(samples)
        ])


with open(
    "results/distilgpt2_easy_cuda_nvml_final.csv",
    "w",
    newline=""
) as f:

    w = csv.writer(f)

    w.writerow([
        "scenario",
        "repeat",
        "context",
        "output",
        "batch",
        "runs",
        "duration_sec",
        "throughput_tok_sec",
        "avg_power_w",
        "gpu_util_percent",
        "vram_mb",
        "energy_run_j",
        "energy_token_j",
        "samples"
    ])

    w.writerows(results)


print("\n===== FINAL MEDIAN SUMMARY =====")

for name, _, _, _ in SCENARIOS:

    rows = [
        r for r in results
        if r[0] == name
    ]

    print(
        f"{name:6s} | "
        f"{statistics.median(r[7] for r in rows):.1f} tok/s | "
        f"{statistics.median(r[8] for r in rows):.2f} W | "
        f"{statistics.median(r[9] for r in rows):.1f}% util | "
        f"{statistics.median(r[10] for r in rows):.0f} MB | "
        f"{statistics.median(r[11] for r in rows):.2f} J/run | "
        f"{statistics.median(r[12] for r in rows):.4f} J/token"
    )

nvmlShutdown()
