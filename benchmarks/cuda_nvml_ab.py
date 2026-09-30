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

CTX = 128
OUT = 64
BATCH = 4

DURATION = 60
COOLDOWN = 15
REPEATS = 3
SAMPLE_INTERVAL = 0.1

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL).to("cuda")
model.eval()

text = (
    "Artificial intelligence on edge devices enables "
    "efficient distributed machine learning systems. "
)

ids = tokenizer(text, add_special_tokens=False)["input_ids"]
ids = (ids * ((CTX // len(ids)) + 2))[:CTX]

input_ids = torch.tensor(
    [ids] * BATCH,
    dtype=torch.long,
    device="cuda"
)

inputs = {
    "input_ids": input_ids,
    "attention_mask": torch.ones_like(input_ids)
}

# GPU warm-up
with torch.inference_mode():
    model.generate(
        **inputs,
        max_new_tokens=OUT,
        min_new_tokens=OUT,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

torch.cuda.synchronize()

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)


def run_workload(use_nvml):

    stop = threading.Event()
    samples = []

    def sampler():
        while not stop.is_set():
            try:
                power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0
                util = nvmlDeviceGetUtilizationRates(handle).gpu
                mem_mb = nvmlDeviceGetMemoryInfo(handle).used / 1024**2

                samples.append((power_w, util, mem_mb))
            except Exception:
                pass

            time.sleep(SAMPLE_INTERVAL)

    thread = None

    if use_nvml:
        thread = threading.Thread(target=sampler)
        thread.start()

    torch.cuda.synchronize()

    start = time.perf_counter()
    runs = 0

    while time.perf_counter() - start < DURATION:

        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=OUT,
                min_new_tokens=OUT,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        runs += 1

    elapsed = time.perf_counter() - start

    if use_nvml:
        stop.set()
        thread.join()

    throughput = runs * BATCH * OUT / elapsed

    return throughput, samples


no_values = []
nvml_values = []

# Alternating order reduces thermal/order bias
orders = [
    [False, True],
    [True, False],
    [False, True],
]

for repeat, order in enumerate(orders, 1):

    print(f"\n===== PAIR {repeat}/3 =====")

    for use_nvml in order:

        print("Cooldown...")
        time.sleep(COOLDOWN)

        label = "NVML" if use_nvml else "NO-MONITOR"

        throughput, samples = run_workload(use_nvml)

        if use_nvml and samples:
            avg_power = statistics.mean(x[0] for x in samples)
            avg_util = statistics.mean(x[1] for x in samples)

            print(
                f"{label}: {throughput:.1f} tok/s | "
                f"{avg_power:.2f} W | "
                f"{avg_util:.1f}% util | "
                f"samples={len(samples)}"
            )
        else:
            print(f"{label}: {throughput:.1f} tok/s")

        if use_nvml:
            nvml_values.append(throughput)
        else:
            no_values.append(throughput)


no_median = statistics.median(no_values)
nvml_median = statistics.median(nvml_values)

difference = (
    (nvml_median / no_median) - 1
) * 100


print("\n===== NVML A/B SUMMARY =====")
print(f"No monitoring : {no_median:.1f} tok/s")
print(f"NVML 100ms    : {nvml_median:.1f} tok/s")
print(f"Difference    : {difference:.1f}%")

nvmlShutdown()
