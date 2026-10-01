import csv
import re
import time
import threading
import statistics
import urllib.request
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LHM_URL = "http://localhost:8085/data.json"

CTX = 128
OUT = 16
BATCH = 1
THREADS = [8, 16]
REPEATS = 3

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float32
)
model.eval()

seed = (
    "Artificial intelligence on edge devices enables "
    "resource aware distributed inference systems. "
)

ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
ids = (ids * ((CTX // len(ids)) + 2))[:CTX]

x = torch.tensor([ids], dtype=torch.long)

inputs = {
    "input_ids": x,
    "attention_mask": torch.ones_like(x)
}


def cpu_package_power():
    with urllib.request.urlopen(LHM_URL, timeout=2) as r:
        data = json.load(r)

    def search(obj):
        if isinstance(obj, dict):
            name = str(
                obj.get("Text")
                or obj.get("Title")
                or obj.get("Name")
                or ""
            )

            if "CPU Package" in name:
                value = str(obj.get("Value", ""))
                m = re.search(r"([\d.,]+)\s*W", value)
                if m:
                    return float(m.group(1).replace(",", "."))

            for value in obj.values():
                result = search(value)
                if result is not None:
                    return result

        elif isinstance(obj, list):
            for item in obj:
                result = search(item)
                if result is not None:
                    return result

        return None

    value = search(data)

    if value is None:
        raise RuntimeError("CPU Package power sensor not found")

    return value


def generate(out_tokens):
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=out_tokens,
            min_new_tokens=out_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )


print("\nMeasuring idle CPU package power...")

idle_samples = []

time.sleep(20)

for _ in range(20):
    idle_samples.append(cpu_package_power())
    time.sleep(1.0)

idle_power = statistics.mean(idle_samples)

print(f"Idle CPU package power: {idle_power:.2f} W")

rows = []

for threads in THREADS:

    torch.set_num_threads(threads)

    print(f"\nWarm-up {threads}t...")
    generate(1)

    for repeat in range(1, REPEATS + 1):

        print(f"\n{threads}t repeat {repeat}")

        time.sleep(5)

        powers = []
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                try:
                    powers.append(cpu_package_power())
                except Exception:
                    pass
                time.sleep(1.0)

        t = threading.Thread(target=monitor)
        t.start()

        start = time.perf_counter()
        result = generate(OUT)
        elapsed = time.perf_counter() - start

        stop.set()
        t.join()

        tokens = (result.shape[1] - CTX) * BATCH
        tok_s = tokens / elapsed

        avg_power = statistics.mean(powers)
        dynamic_power = max(avg_power - idle_power, 0)

        energy_run = avg_power * elapsed
        dynamic_energy_run = dynamic_power * elapsed

        energy_token = energy_run / tokens
        dynamic_energy_token = dynamic_energy_run / tokens

        row = {
            "threads": threads,
            "repeat": repeat,
            "context": CTX,
            "output": OUT,
            "batch": BATCH,
            "latency_sec": elapsed,
            "throughput_tok_sec": tok_s,
            "idle_power_w": idle_power,
            "avg_power_w": avg_power,
            "dynamic_power_w": dynamic_power,
            "energy_run_j": energy_run,
            "dynamic_energy_run_j": dynamic_energy_run,
            "energy_token_j": energy_token,
            "dynamic_energy_token_j": dynamic_energy_token,
            "samples": len(powers),
        }

        rows.append(row)

        print(
            f"{threads:2}t R{repeat} | "
            f"{elapsed:7.2f}s | "
            f"{tok_s:.3f} tok/s | "
            f"{avg_power:6.2f} W | "
            f"{energy_run:8.2f} J/run | "
            f"{energy_token:7.2f} J/token | "
            f"samples={len(powers)}"
        )


outfile = "results/qwen25_medium_msi_cpu_power.csv"

with open(outfile, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


print("\nMEDIANS")
print("-" * 100)

for threads in THREADS:
    p = [x for x in rows if x["threads"] == threads]

    print(
        f"{threads:2}t | "
        f"{statistics.median(x['throughput_tok_sec'] for x in p):.3f} tok/s | "
        f"{statistics.median(x['avg_power_w'] for x in p):6.2f} W | "
        f"{statistics.median(x['dynamic_power_w'] for x in p):6.2f} W dyn | "
        f"{statistics.median(x['energy_run_j'] for x in p):8.2f} J/run | "
        f"{statistics.median(x['energy_token_j'] for x in p):7.2f} J/token"
    )

print("\nSaved:", outfile)


