import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROFILES = {
    "light":  (128, 64, 1),
    "medium": (512, 128, 2),
    "heavy":  (1024, 128, 4),
}

parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=PROFILES, required=True)
parser.add_argument("--duration", type=int, default=60)
args = parser.parse_args()

CTX, OUT, BATCH = PROFILES[args.profile]
DEVICE = "mps"

print("=" * 40)
print("QWEN2.5 MEDIUM MPS POWER")
print("=" * 40)
print(f"Profile : {args.profile.upper()}")
print(f"CTX     : {CTX}")
print(f"OUT     : {OUT}")
print(f"BATCH   : {BATCH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16
).to(DEVICE)

model.eval()

seed = (
    "Artificial intelligence on edge devices enables "
    "resource aware distributed inference systems. "
)

ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
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

import os

READY = "/tmp/qwen_ready"
GO = "/tmp/qwen_go"

if os.path.exists(READY):
    os.remove(READY)
if os.path.exists(GO):
    os.remove(GO)

open(READY, "w").close()
print("READY FOR POWER MEASUREMENT", flush=True)

while not os.path.exists(GO):
    time.sleep(0.05)

runs = 0
total_tokens = 0

start = time.perf_counter()

while (time.perf_counter() - start) < args.duration:
    result = generate()

    tokens = (result.shape[1] - CTX) * BATCH
    total_tokens += tokens
    runs += 1

elapsed = time.perf_counter() - start

print("\nRESULT")
print("=" * 40)
print(f"Profile        : {args.profile}")
print(f"Duration       : {elapsed:.2f} s")
print(f"Runs           : {runs}")
print(f"Generated      : {total_tokens} tokens")
print(f"Throughput     : {total_tokens / elapsed:.2f} tok/s")
print(f"Seconds/run    : {elapsed / runs:.3f}")
