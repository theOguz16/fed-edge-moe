import os
import csv
import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXTS = [128, 512, 1024]
OUTPUTS = [64, 128]
BATCHES = [1, 2, 4]
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

print("================================")
print("QWEN2.5 MEDIUM TEXT SWEEP")
print("================================")
print("Device:", DEVICE_NAME)
print("Model :", MODEL)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16
).to(DEVICE)

model.eval()


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def clear_memory():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def memory_mb():
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2

    if DEVICE == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024**2
        except Exception:
            return None


def make_inputs(context_len, batch):
    seed = (
        "Artificial intelligence on edge devices enables "
        "resource aware distributed inference systems. "
    )

    ids = tokenizer(
        seed,
        add_special_tokens=False
    )["input_ids"]

    ids = (
        ids * ((context_len // len(ids)) + 2)
    )[:context_len]

    input_ids = torch.tensor(
        [ids] * batch,
        dtype=torch.long,
        device=DEVICE
    )

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids)
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

            clear_memory()

            try:
                inputs = make_inputs(context, batch)

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

                    print(
                        f"  R{repeat}: "
                        f"{elapsed:.2f}s | "
                        f"{throughput:.2f} tok/s | "
                        f"{mem:.0f} MB"
                        if mem is not None
                        else
                        f"  R{repeat}: "
                        f"{elapsed:.2f}s | "
                        f"{throughput:.2f} tok/s"
                    )

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
                    "  MEDIAN -> "
                    f"{statistics.median(latencies):.2f}s | "
                    f"{statistics.median(throughputs):.2f} tok/s"
                )

            except RuntimeError as e:

                message = str(e).lower()

                if "out of memory" in message:
                    status = "oom"
                else:
                    status = "runtime_error"

                print(f"  {status.upper()}")

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

                clear_memory()


outfile = f"results/qwen25_medium_{DEVICE}_sweep.csv"

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


print("\n================================")
print("MEDIUM SWEEP COMPLETE")
print("================================")
print("Saved:", outfile)
