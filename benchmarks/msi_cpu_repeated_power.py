import json
import time
import urllib.request
import statistics
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"

THREAD_COUNTS = [2, 4, 6, 16]
REPEATS = 3

WORKLOAD_DURATION = 30
COOLDOWN = 15
IDLE_DURATION = 8
SAMPLE_INTERVAL = 0.5

PROMPT = "Artificial intelligence on edge devices can"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64

URL = "http://localhost:8085/data.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token


def get_cpu_power():
    with urllib.request.urlopen(URL, timeout=2) as response:
        data = json.load(response)

    def search(node):
        if node.get("Text") == "CPU Package":
            value = str(node.get("Value", ""))

            if "W" in value:
                value = (
                    value.replace("W", "")
                    .replace(",", ".")
                    .strip()
                )
                return float(value)

        for child in node.get("Children", []):
            result = search(child)
            if result is not None:
                return result

        return None

    return search(data)


def sample_power(duration):
    samples = []
    start = time.perf_counter()

    while time.perf_counter() - start < duration:
        try:
            power = get_cpu_power()

            if power is not None:
                samples.append(power)

        except Exception:
            pass

        time.sleep(SAMPLE_INTERVAL)

    return samples


all_results = []


for threads in THREAD_COUNTS:

    print(f"\n==============================")
    print(f"THREADS = {threads}")
    print(f"==============================")

    for repeat in range(1, REPEATS + 1):

        print(f"\n--- Repeat {repeat}/{REPEATS} ---")

        torch.set_num_threads(threads)

        model = AutoModelForCausalLM.from_pretrained(MODEL)
        model.eval()

        inputs = tokenizer(
            [PROMPT] * BATCH_SIZE,
            return_tensors="pt",
            padding=True
        )

        # Warm-up
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        print(f"Cooling down {COOLDOWN}s...")
        time.sleep(COOLDOWN)

        # Gerçek idle ölçümü
        print("Measuring idle power...")
        idle_samples = sample_power(IDLE_DURATION)
        idle_power = statistics.mean(idle_samples)

        print(f"Idle power: {idle_power:.2f} W")

        # Workload + power sampling
        power_samples = []

        start = time.perf_counter()
        last_sample = start
        runs = 0

        while time.perf_counter() - start < WORKLOAD_DURATION:

            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            runs += 1

            now = time.perf_counter()

            if now - last_sample >= SAMPLE_INTERVAL:
                try:
                    power = get_cpu_power()

                    if power is not None:
                        power_samples.append(power)

                except Exception:
                    pass

                last_sample = now

        elapsed = time.perf_counter() - start

        avg_power = statistics.mean(power_samples)

        dynamic_samples = [
            max(p - idle_power, 0)
            for p in power_samples
        ]

        avg_dynamic_power = statistics.mean(dynamic_samples)

        energy_per_run = (
            avg_power * elapsed / runs
        )

        dynamic_energy_per_run = (
            avg_dynamic_power * elapsed / runs
        )

        runs_per_sec = runs / elapsed
        tokens_per_sec = (
            runs * BATCH_SIZE * MAX_NEW_TOKENS
        ) / elapsed

        print(
            f"runs={runs} | "
            f"{runs_per_sec:.3f} run/s | "
            f"{tokens_per_sec:.1f} tok/s"
        )

        print(
            f"power={avg_power:.2f} W | "
            f"idle={idle_power:.2f} W"
        )

        print(
            f"energy/run={energy_per_run:.2f} J | "
            f"dynamic={dynamic_energy_per_run:.2f} J"
        )

        all_results.append([
            threads,
            repeat,
            runs,
            elapsed,
            runs_per_sec,
            tokens_per_sec,
            idle_power,
            avg_power,
            avg_dynamic_power,
            energy_per_run,
            dynamic_energy_per_run
        ])

        del model

        time.sleep(10)


with open(
    "results/msi_cpu_repeated_power.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "threads",
        "repeat",
        "runs",
        "duration_sec",
        "runs_per_sec",
        "tokens_per_sec",
        "idle_power_w",
        "avg_power_w",
        "dynamic_power_w",
        "energy_per_run_j",
        "dynamic_energy_per_run_j"
    ])

    writer.writerows(all_results)


print("\n\n======== MEDIAN SUMMARY ========")

for threads in THREAD_COUNTS:

    rows = [
        r for r in all_results
        if r[0] == threads
    ]

    runs_sec = statistics.median(
        [r[4] for r in rows]
    )

    tok_sec = statistics.median(
        [r[5] for r in rows]
    )

    power = statistics.median(
        [r[7] for r in rows]
    )

    energy = statistics.median(
        [r[9] for r in rows]
    )

    dynamic_energy = statistics.median(
        [r[10] for r in rows]
    )

    print(
        f"{threads:2d}t | "
        f"{runs_sec:.3f} run/s | "
        f"{tok_sec:.1f} tok/s | "
        f"{power:.2f} W | "
        f"{energy:.2f} J/run | "
        f"dynamic {dynamic_energy:.2f} J/run"
    )


print("\nSaved: results/msi_cpu_repeated_power.csv")
