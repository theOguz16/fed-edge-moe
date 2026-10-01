import csv
import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
THREADS = [1, 2, 4, 6, 8, 10]
REPEATS = 3

WORKLOADS = {
    "long_single": (128, 128, 1),
    "batch4":      (128, 64, 4),
}

print("=" * 45)
print("QWEN2.5 MEDIUM MAC CPU SCALING")
print("=" * 45)

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

def make_inputs(ctx, batch):
    ids = tokenizer(
        seed,
        add_special_tokens=False
    )["input_ids"]

    ids = (ids * ((ctx // len(ids)) + 2))[:ctx]

    x = torch.tensor(
        [ids] * batch,
        dtype=torch.long
    )

    return {
        "input_ids": x,
        "attention_mask": torch.ones_like(x)
    }

rows = []

for workload, (ctx, out, batch) in WORKLOADS.items():

    inputs = make_inputs(ctx, batch)

    print(f"\n### {workload}")
    print(f"CTX={ctx} OUT={out} BATCH={batch}")

    for threads in THREADS:

        torch.set_num_threads(threads)

        # Warm-up
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=8,
                min_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        latencies = []
        throughputs = []

        for repeat in range(1, REPEATS + 1):

            start = time.perf_counter()

            with torch.inference_mode():
                result = model.generate(
                    **inputs,
                    max_new_tokens=out,
                    min_new_tokens=out,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            elapsed = time.perf_counter() - start
            generated = (result.shape[1] - ctx) * batch
            tok_s = generated / elapsed

            latencies.append(elapsed)
            throughputs.append(tok_s)

            rows.append({
                "workload": workload,
                "context": ctx,
                "output": out,
                "batch": batch,
                "threads": threads,
                "repeat": repeat,
                "latency_sec": elapsed,
                "throughput_tok_sec": tok_s,
                "dtype": "float32",
                "status": "ok",
            })

            print(
                f"{threads:2}t R{repeat} | "
                f"{elapsed:7.2f}s | "
                f"{tok_s:6.2f} tok/s"
            )

        print(
            f"{threads:2}t MEDIAN | "
            f"{statistics.median(latencies):7.2f}s | "
            f"{statistics.median(throughputs):6.2f} tok/s"
        )

with open(
    "results/qwen25_medium_mac_cpu_scaling.csv",
    "w",
    newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved: results/qwen25_medium_mac_cpu_scaling.csv")
