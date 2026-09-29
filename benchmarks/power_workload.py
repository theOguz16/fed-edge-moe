import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"
MODE = sys.argv[1]
THREADS = int(sys.argv[2])
DURATION = 40

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

device = "mps" if MODE == "mps" else "cpu"

if device == "cpu":
    torch.set_num_threads(THREADS)

model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()
model.to(device)

inputs = tokenizer(
    ["Artificial intelligence on edge devices can"] * 4,
    return_tensors="pt",
    padding=True
).to(device)

# warm-up
with torch.inference_mode():
    model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

if device == "mps":
    torch.mps.synchronize()

print(f"RUNNING: {MODE}, threads={THREADS}")

start = time.time()
runs = 0

while time.time() - start < DURATION:
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    if device == "mps":
        torch.mps.synchronize()

    runs += 1

print("Runs:", runs)
