import argparse
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CTX = 128
OUT = 64
BATCH = 4

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, required=True)
parser.add_argument("--duration", type=int, default=60)
args = parser.parse_args()

torch.set_num_threads(args.threads)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

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

x = torch.tensor([ids] * BATCH, dtype=torch.long)

inputs = {
    "input_ids": x,
    "attention_mask": torch.ones_like(x)
}

def generate():
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=OUT,
            min_new_tokens=OUT,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

print(f"Threads: {args.threads}")
print("Warm-up...")
generate()

READY = "/tmp/qwen_cpu_ready"
GO = "/tmp/qwen_cpu_go"

for p in [READY, GO]:
    if os.path.exists(p):
        os.remove(p)

open(READY, "w").close()
print("READY", flush=True)

while not os.path.exists(GO):
    time.sleep(0.05)

runs = 0
tokens = 0
start = time.perf_counter()

while time.perf_counter() - start < args.duration:
    result = generate()
    tokens += (result.shape[1] - CTX) * BATCH
    runs += 1

elapsed = time.perf_counter() - start

print("\nRESULT")
print(f"Threads        : {args.threads}")
print(f"Duration       : {elapsed:.2f} s")
print(f"Runs           : {runs}")
print(f"Generated      : {tokens}")
print(f"Throughput     : {tokens / elapsed:.2f} tok/s")
print(f"Seconds/run    : {elapsed / runs:.3f}")
