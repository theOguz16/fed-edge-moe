import time
import subprocess
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

REPEATS = 3
DURATION = 60
CTX = 128
OUT = 64
BATCH = 4

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("distilgpt2").to("cuda")
model.eval()

text = (
    "Artificial intelligence on edge devices enables "
    "efficient distributed machine learning systems. "
)

ids = tokenizer(text, add_special_tokens=False)["input_ids"]
ids = (ids * ((CTX // len(ids)) + 2))[:CTX]

input_ids = torch.tensor([ids] * BATCH, device="cuda")
inputs = {
    "input_ids": input_ids,
    "attention_mask": torch.ones_like(input_ids)
}

def run_test(monitor):
    values = []

    for r in range(REPEATS):
        proc = None

        if monitor:
            proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                    "--loop-ms", "500"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        torch.cuda.synchronize()
        start = time.perf_counter()
        runs = 0

        while time.perf_counter() - start < DURATION:
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=OUT,
                    min_new_tokens=OUT,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            torch.cuda.synchronize()
            runs += 1

        elapsed = time.perf_counter() - start

        if proc:
            proc.terminate()
            proc.wait()

        throughput = runs * BATCH * OUT / elapsed
        values.append(throughput)

        print(
            f"{'MONITOR' if monitor else 'NO-MONITOR'} "
            f"R{r+1}: {throughput:.1f} tok/s"
        )

        time.sleep(10)

    return statistics.median(values)


no_monitor = run_test(False)
monitor = run_test(True)

print("\n===== A/B SUMMARY =====")
print(f"No monitoring : {no_monitor:.1f} tok/s")
print(f"500ms monitor : {monitor:.1f} tok/s")
print(f"Difference    : {(monitor/no_monitor-1)*100:.1f}%")
