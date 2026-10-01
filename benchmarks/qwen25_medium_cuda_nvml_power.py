import csv
import time
import threading
import statistics
import torch
import pynvml
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DURATION = 60
REPEATS = 3

PROFILES = {
    "light":  (128, 64, 1),
    "medium": (512, 128, 2),
    "heavy":  (1024, 128, 4),
}

DEVICE = "cuda"

print("=" * 50)
print("QWEN2.5 MEDIUM CUDA NVML POWER")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16
).to(DEVICE)

model.eval()

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_power():
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

print("\nMeasuring idle GPU board power...")
idle_samples = []

for _ in range(50):
    idle_samples.append(gpu_power())
    time.sleep(0.1)

idle_power = statistics.mean(idle_samples)
print(f"Idle GPU power: {idle_power:.2f} W")

seed = (
    "Artificial intelligence on edge devices enables "
    "resource aware distributed inference systems. "
)

def make_inputs(ctx, batch):
    ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
    ids = (ids * ((ctx // len(ids)) + 2))[:ctx]

    x = torch.tensor(
        [ids] * batch,
        dtype=torch.long,
        device=DEVICE
    )

    return {
        "input_ids": x,
        "attention_mask": torch.ones_like(x)
    }

def generate(inputs, out):
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=out,
            min_new_tokens=out,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    torch.cuda.synchronize()
    return result

rows = []

for profile, (ctx, out, batch) in PROFILES.items():

    inputs = make_inputs(ctx, batch)

    print(f"\nWarm-up: {profile}")
    generate(inputs, out)

    for repeat in range(1, REPEATS + 1):

        print(f"\n{profile} repeat {repeat}")

        powers = []
        utils = []
        vrams = []
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                try:
                    powers.append(
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    )

                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utils.append(util.gpu)

                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vrams.append(mem.used / 1024**2)

                except Exception:
                    pass

                time.sleep(0.1)

        t = threading.Thread(target=monitor)
        t.start()

        runs = 0
        tokens = 0

        start = time.perf_counter()

        while time.perf_counter() - start < DURATION:
            result = generate(inputs, out)

            tokens += (result.shape[1] - ctx) * batch
            runs += 1

        elapsed = time.perf_counter() - start

        stop.set()
        t.join()

        tok_s = tokens / elapsed

        avg_power = statistics.mean(powers)
        dynamic_power = max(avg_power - idle_power, 0)

        avg_util = statistics.mean(utils)
        peak_vram = max(vrams)

        energy_run = avg_power * elapsed / runs
        energy_token = avg_power * elapsed / tokens

        dynamic_energy_run = dynamic_power * elapsed / runs
        dynamic_energy_token = dynamic_power * elapsed / tokens

        row = {
            "profile": profile,
            "repeat": repeat,
            "context": ctx,
            "output": out,
            "batch": batch,
            "duration_sec": elapsed,
            "runs": runs,
            "tokens": tokens,
            "throughput_tok_sec": tok_s,
            "idle_power_w": idle_power,
            "avg_power_w": avg_power,
            "dynamic_power_w": dynamic_power,
            "gpu_util_percent": avg_util,
            "peak_vram_mb": peak_vram,
            "energy_run_j": energy_run,
            "energy_token_j": energy_token,
            "dynamic_energy_run_j": dynamic_energy_run,
            "dynamic_energy_token_j": dynamic_energy_token,
            "samples": len(powers),
        }

        rows.append(row)

        print(
            f"{profile:6} R{repeat} | "
            f"{tok_s:6.2f} tok/s | "
            f"{avg_power:6.2f} W | "
            f"{avg_util:5.1f}% | "
            f"{peak_vram:6.0f} MB | "
            f"{energy_run:7.2f} J/run | "
            f"{energy_token:.4f} J/token | "
            f"samples={len(powers)}"
        )

outfile = "results/qwen25_medium_cuda_nvml_power.csv"

with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nMEDIANS")
print("-" * 110)

for profile in PROFILES:
    p = [x for x in rows if x["profile"] == profile]

    print(
        f"{profile:6} | "
        f"{statistics.median(x['throughput_tok_sec'] for x in p):6.2f} tok/s | "
        f"{statistics.median(x['avg_power_w'] for x in p):6.2f} W | "
        f"{statistics.median(x['gpu_util_percent'] for x in p):5.1f}% | "
        f"{statistics.median(x['peak_vram_mb'] for x in p):6.0f} MB | "
        f"{statistics.median(x['energy_run_j'] for x in p):7.2f} J/run | "
        f"{statistics.median(x['energy_token_j'] for x in p):.4f} J/token"
    )

print("\nSaved:", outfile)

pynvml.nvmlShutdown()
