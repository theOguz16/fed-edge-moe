import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DURATION = 60
REPEATS = 3

PROFILES = {
    "light":  (128, 64, 1),
    "medium": (512, 128, 2),
    "heavy":  (1024, 128, 4),
}

tok = AutoTokenizer.from_pretrained(MODEL)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16
).to("cuda")

model.eval()

seed = (
    "Artificial intelligence on edge devices enables "
    "resource aware distributed inference systems. "
)

def inputs(ctx, batch):
    ids = tok(seed, add_special_tokens=False)["input_ids"]
    ids = (ids * ((ctx // len(ids)) + 2))[:ctx]

    x = torch.tensor(
        [ids] * batch,
        dtype=torch.long,
        device="cuda"
    )

    return {
        "input_ids": x,
        "attention_mask": torch.ones_like(x)
    }

def generate(inp, out):
    with torch.inference_mode():
        y = model.generate(
            **inp,
            max_new_tokens=out,
            min_new_tokens=out,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
    torch.cuda.synchronize()
    return y

for name, (ctx, out, batch) in PROFILES.items():

    inp = inputs(ctx, batch)
    generate(inp, out)

    values = []

    print(f"\n{name.upper()}")

    for r in range(1, REPEATS + 1):
        runs = 0
        tokens = 0
        start = time.perf_counter()

        while time.perf_counter() - start < DURATION:
            y = generate(inp, out)
            tokens += (y.shape[1] - ctx) * batch
            runs += 1

        elapsed = time.perf_counter() - start
        ts = tokens / elapsed
        values.append(ts)

        print(f"R{r}: {ts:.2f} tok/s")

    print(f"MEDIAN: {statistics.median(values):.2f} tok/s")
