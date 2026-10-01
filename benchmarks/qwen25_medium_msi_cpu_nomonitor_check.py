import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CTX = 128
OUT = 16

torch.set_num_threads(16)

tok = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float32
)

model.eval()

seed = (
    "Artificial intelligence on edge devices enables "
    "resource aware distributed inference systems. "
)

ids = tok(seed, add_special_tokens=False)["input_ids"]
ids = (ids * ((CTX // len(ids)) + 2))[:CTX]

x = torch.tensor([ids], dtype=torch.long)

inputs = {
    "input_ids": x,
    "attention_mask": torch.ones_like(x)
}

with torch.inference_mode():
    model.generate(
        **inputs,
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
        pad_token_id=tok.eos_token_id
    )

print("NO-MONITOR 16t")

for r in range(1, 4):
    start = time.perf_counter()

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=OUT,
            min_new_tokens=OUT,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )

    elapsed = time.perf_counter() - start

    print(
        f"R{r}: {elapsed:.2f}s | "
        f"{OUT/elapsed:.3f} tok/s"
    )
