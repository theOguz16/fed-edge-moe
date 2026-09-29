import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"
THREAD_COUNTS = [1, 2, 4, 6, 8, 10]
REPEATS = 5

SCENARIOS = [
    ("long_single", 1, 128),
    ("batch4", 4, 64),
]

PROMPT = "Artificial intelligence on edge devices can"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


def make_inputs(batch_size, device):
    prompts = [PROMPT] * batch_size
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True
    ).to(device)


def run_test(model, device, batch_size, max_new_tokens):
    inputs = make_inputs(batch_size, device)

    # Warm-up
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    times = []

    for _ in range(REPEATS):
        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if device == "mps":
            torch.mps.synchronize()

        times.append(time.perf_counter() - start)

    latency = statistics.median(times)
    total_tokens = batch_size * max_new_tokens
    throughput = total_tokens / latency

    return latency, throughput


print("=== CPU SCALING ===")

for scenario, batch_size, max_tokens in SCENARIOS:
    print(f"\n--- {scenario} | batch={batch_size} | tokens={max_tokens} ---")

    for threads in THREAD_COUNTS:
        torch.set_num_threads(threads)

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model.eval()

        latency, throughput = run_test(
            model,
            "cpu",
            batch_size,
            max_tokens
        )

        print(
            f"{threads:2d} threads | "
            f"{latency:.3f}s | "
            f"{throughput:.1f} tok/s"
        )

        del model


if torch.backends.mps.is_available():
    print("\n=== MPS GPU ===")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    model.to("mps")

    for scenario, batch_size, max_tokens in SCENARIOS:
        latency, throughput = run_test(
            model,
            "mps",
            batch_size,
            max_tokens
        )

        print(
            f"{scenario:12s} | "
            f"{latency:.3f}s | "
            f"{throughput:.1f} tok/s"
        )
