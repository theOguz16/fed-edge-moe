import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3-1.7B"
CTX = 128
OUT = 64
BATCH = 1
DEVICE = "mps"

print("=" * 40)
print("QWEN3 HARD MAC FEASIBILITY")
print("=" * 40)
print("Model :", MODEL)
print("Device:", DEVICE)
print(f"CTX={CTX} OUT={OUT} BATCH={BATCH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("\nLoading model...")

t0 = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16
).to(DEVICE)

model.eval()
torch.mps.synchronize()

print(f"Model load time: {time.perf_counter() - t0:.2f}s")

seed = (
    "Artificial intelligence on edge devices enables "
    "resource aware distributed inference systems. "
)

ids = tokenizer(
    seed,
    add_special_tokens=False
)["input_ids"]

ids = (ids * ((CTX // len(ids)) + 2))[:CTX]

input_ids = torch.tensor(
    [ids] * BATCH,
    dtype=torch.long,
    device=DEVICE
)

inputs = {
    "input_ids": input_ids,
    "attention_mask": torch.ones_like(input_ids)
}

def generate():
    with torch.inference_mode():
        result = model.generate(
            **inputs,
            max_new_tokens=OUT,
            min_new_tokens=OUT,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.mps.synchronize()
    return result

print("\nWarm-up...")
generate()

print("Measured inference...")

start = time.perf_counter()
result = generate()
elapsed = time.perf_counter() - start

tokens = (result.shape[1] - CTX) * BATCH
tok_s = tokens / elapsed

print("\n" + "=" * 40)
print("RESULT")
print("=" * 40)
print(f"Context        : {CTX}")
print(f"Output         : {OUT}")
print(f"Batch          : {BATCH}")
print(f"Latency        : {elapsed:.3f} s")
print(f"Throughput     : {tok_s:.2f} tok/s")

try:
    mem = torch.mps.current_allocated_memory() / 1024**3
    print(f"MPS allocated  : {mem:.2f} GB")
except Exception:
    pass

print("Status         : PASS")
