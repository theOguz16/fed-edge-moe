import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-1.7B"
CONTEXT = 128
OUTPUT = 64
BATCH = 1

if torch.cuda.is_available():
    DEVICE = "cuda"
    DEVICE_NAME = torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DEVICE_NAME = "Apple MPS"
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"

print("================================")
print("QWEN3 MEDIUM FEASIBILITY")
print("================================")
print("Device:", DEVICE_NAME)
print("Model :", MODEL)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("\nLoading model...")

load_start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16
).to(DEVICE)

model.eval()

load_time = time.perf_counter() - load_start

print(f"Model load time: {load_time:.2f}s")


# Controlled 128-token synthetic context
text = (
    "Artificial intelligence on edge devices enables "
    "distributed and resource aware machine learning systems. "
)

ids = tokenizer(
    text,
    add_special_tokens=False
)["input_ids"]

ids = (
    ids * ((CONTEXT // len(ids)) + 2)
)[:CONTEXT]

input_ids = torch.tensor(
    [ids] * BATCH,
    dtype=torch.long,
    device=DEVICE
)

attention_mask = torch.ones_like(input_ids)

inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask
}


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


# Warm-up
print("\nWarm-up...")

with torch.inference_mode():
    model.generate(
        **inputs,
        max_new_tokens=OUTPUT,
        min_new_tokens=OUTPUT,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

sync()


if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()


# Measured inference
print("Measured inference...")

sync()
start = time.perf_counter()

with torch.inference_mode():
    result = model.generate(
        **inputs,
        max_new_tokens=OUTPUT,
        min_new_tokens=OUTPUT,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

sync()

elapsed = time.perf_counter() - start

generated_tokens = (
    result.shape[1] - CONTEXT
) * BATCH

throughput = generated_tokens / elapsed


print("\n================================")
print("RESULT")
print("================================")

print(f"Context        : {CONTEXT}")
print(f"Output         : {OUTPUT}")
print(f"Batch          : {BATCH}")
print(f"Latency        : {elapsed:.3f} s")
print(f"Throughput     : {throughput:.2f} tok/s")

if DEVICE == "cuda":
    allocated = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"VRAM allocated : {allocated:.2f} GB")
    print(f"VRAM peak      : {peak:.2f} GB")

elif DEVICE == "mps":
    try:
        memory = torch.mps.current_allocated_memory() / 1024**3
        print(f"MPS allocated  : {memory:.2f} GB")
    except Exception:
        pass

print("Status         : PASS")
