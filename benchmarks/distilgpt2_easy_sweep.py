import os
import csv
import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"

CONTEXTS = [32, 128, 512]
OUTPUTS = [32, 64, 128]
BATCHES = [1, 2, 4, 8]
REPEATS = 3

os.makedirs("results", exist_ok=True)

if torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_NAME = torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DEVICE_NAME = "Apple MPS"
else:
    raise RuntimeError("CUDA veya MPS bulunamadı.")

print("Device:", DEVICE_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL)
model.to(DEVICE)
model.eval()


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def memory_mb():
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2

    if DEVICE == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024**2
        except Exception:
            return None

    return None


def make_inputs(context_len, batch):
    seed_text = (
        "Artificial intelligence on edge devices enables "
        "efficient distributed machine learning systems. "
    )

    ids = tokenizer(
        seed_text,
        add_special_tokens=False
    )["input_ids"]

    repeated = (
        ids * ((context_len // len(ids)) + 2)
    )[:context_len]

    input_ids = torch.tensor(
        [repeated] * batch,
        dtype=torch.long,
        device=DEVICE
    )

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


results = []

for context in CONTEXTS:
    for output in OUTPUTS:
        for batch in BATCHES:

            print(
                f"\nCTX={context} "
                f"OUT={output} "
                f"BATCH={batch}"
            )

            inputs = make_inputs(context, batch)

            try:
                # Warm-up
                with torch.inference_mode():
                    model.generate(
                        **inputs,
                        max_new_tokens=output,
                        min_new_tokens=output,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                sync()

                latencies = []
                throughputs = []
                memories = []

                for repeat in range(1, REPEATS + 1):

                    if DEVICE == "cuda":
                        torch.cuda.reset_peak_memory_stats()

                    sync()
                    start = time.perf_counter()

                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=output,
                            min_new_tokens=output,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    sync()

                    elapsed = time.perf_counter() - start

                    generated_tokens = (
                        generated.shape[1] - context
                    ) * batch

                    throughput = generated_tokens / elapsed
                    mem = memory_mb()

                    latencies.append(elapsed)
                    throughputs.append(throughput)

                    if mem is not None:
                        memories.append(mem)

                    results.append([
                        DEVICE,
                        DEVICE_NAME,
                        MODEL,
                        context,
                        output,
                        batch,
                        repeat,
                        elapsed,
                        throughput,
                        mem,
                        "ok"
                    ])

                    print(
                        f"  R{repeat}: "
                        f"{elapsed:.3f}s | "
                        f"{throughput:.1f} tok/s | "
                        f"{mem:.0f} MB"
                        if mem is not None
                        else
                        f"  R{repeat}: "
                        f"{elapsed:.3f}s | "
                        f"{throughput:.1f} tok/s"
                    )

                print(
                    "  MEDIAN -> "
                    f"{statistics.median(latencies):.3f}s | "
                    f"{statistics.median(throughputs):.1f} tok/s"
                )

            except RuntimeError as e:

                status = "oom" if "out of memory" in str(e).lower() else "error"

                print(" ", status.upper(), str(e)[:100])

                results.append([
                    DEVICE,
                    DEVICE_NAME,
                    MODEL,
                    context,
                    output,
                    batch,
                    0,
                    None,
                    None,
                    None,
                    status
                ])

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()


outfile = f"results/distilgpt2_easy_{DEVICE}_sweep.csv"

with open(outfile, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "backend",
        "device",
        "model",
        "context_tokens",
        "output_tokens",
        "batch",
        "repeat",
        "latency_sec",
        "throughput_tok_sec",
        "device_memory_mb",
        "status"
    ])

    writer.writerows(results)

print("\n==========================")
print("EASY SWEEP COMPLETE")
print("==========================")
print("Saved:", outfile)
