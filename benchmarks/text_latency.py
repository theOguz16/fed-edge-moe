import csv
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"
PROMPT = "Artificial intelligence on edge devices can"
THREAD_COUNTS = [1, 2, 4, 6, 8, 10]
REPEATS = 5
MAX_NEW_TOKENS = 32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def benchmark(device, threads=None):
    if threads is not None:
        torch.set_num_threads(threads)

    print(f"\nLoading model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    # Warm-up: ilk çalıştırma genellikle ekstra maliyetlidir.
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    times = []

    for i in range(REPEATS):
        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()

        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if device == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_latency = sum(times) / len(times)
    tokens_per_second = MAX_NEW_TOKENS / avg_latency

    del model

    if device == "mps":
        torch.mps.empty_cache()

    return avg_latency, tokens_per_second


results = []

print("=== CPU BENCHMARK ===")

for threads in THREAD_COUNTS:
    latency, throughput = benchmark("cpu", threads)

    print(
        f"{threads:2d} threads | "
        f"{latency:.3f} sec | "
        f"{throughput:.2f} tokens/sec"
    )

    results.append([
        "mac",
        MODEL_NAME,
        "cpu",
        threads,
        latency,
        throughput,
    ])


if torch.backends.mps.is_available():
    print("\n=== MPS GPU BENCHMARK ===")

    latency, throughput = benchmark("mps")

    print(
        f"MPS | "
        f"{latency:.3f} sec | "
        f"{throughput:.2f} tokens/sec"
    )

    results.append([
        "mac",
        MODEL_NAME,
        "mps",
        "",
        latency,
        throughput,
    ])


with open("results/mac_text_latency.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "device",
        "model",
        "accelerator",
        "threads",
        "latency_sec",
        "tokens_per_sec",
    ])
    writer.writerows(results)

print("\nSaved: results/mac_text_latency.csv")
