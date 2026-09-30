import csv
import re
import subprocess
import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"
REPEATS = 3
DURATION = 30
COOLDOWN = 10

SCENARIOS = [
    ("light", 32, 32, 1),
    ("medium", 128, 64, 4),
    ("heavy", 512, 128, 8),
]

device = "mps"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
model.eval()


def make_inputs(context_len, batch):
    text = (
        "Artificial intelligence on edge devices enables "
        "efficient distributed machine learning systems. "
    )

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = (ids * ((context_len // len(ids)) + 2))[:context_len]

    input_ids = torch.tensor(
        [ids] * batch,
        dtype=torch.long,
        device=device
    )

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids)
    }


def powermetrics(seconds):
    samples = max(2, int(seconds * 2))

    result = subprocess.run(
        [
            "sudo",
            "powermetrics",
            "-i", "500",
            "-n", str(samples),
            "--samplers", "cpu_power,gpu_power"
        ],
        capture_output=True,
        text=True
    )

    text = result.stdout + result.stderr

    values = re.findall(
        r"Combined Power.*?:\s*([\d.]+)\s*mW",
        text
    )

    return [float(x) / 1000.0 for x in values]


results = []

for name, ctx, out_tokens, batch in SCENARIOS:

    print(f"\n===== {name.upper()} =====")

    inputs = make_inputs(ctx, batch)

    for repeat in range(1, REPEATS + 1):

        print(f"Repeat {repeat}/{REPEATS}")

        # warmup
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=out_tokens,
                min_new_tokens=out_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.mps.synchronize()

        time.sleep(COOLDOWN)

        print("Idle power...")
        idle_samples = powermetrics(8)
        idle_power = statistics.mean(idle_samples)

        print("Workload...")

        proc = subprocess.Popen(
            [
                "sudo",
                "powermetrics",
                "-i", "500",
                "-n", str(DURATION * 2),
                "--samplers", "cpu_power,gpu_power"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        start = time.perf_counter()
        runs = 0

        while time.perf_counter() - start < DURATION:
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=out_tokens,
                    min_new_tokens=out_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            torch.mps.synchronize()
            runs += 1

        elapsed = time.perf_counter() - start

        power_text, _ = proc.communicate()

        values = re.findall(
            r"Combined Power.*?:\s*([\d.]+)\s*mW",
            power_text
        )

        powers = [float(x) / 1000.0 for x in values]

        if not powers:
            raise RuntimeError("Combined Power samples bulunamadı.")

        avg_power = statistics.mean(powers)
        dynamic_power = max(avg_power - idle_power, 0)

        energy_run = avg_power * elapsed / runs
        dynamic_energy = dynamic_power * elapsed / runs

        throughput = (
            runs * batch * out_tokens
        ) / elapsed

        print(
            f"{name}: runs={runs} | "
            f"{throughput:.1f} tok/s | "
            f"{avg_power:.2f} W | "
            f"{energy_run:.2f} J/run | "
            f"dynamic={dynamic_energy:.2f} J/run"
        )

        results.append([
            name, ctx, out_tokens, batch, repeat,
            runs, elapsed, throughput,
            idle_power, avg_power,
            energy_run, dynamic_energy
        ])


with open(
    "results/distilgpt2_easy_mps_power.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "scenario",
        "context",
        "output",
        "batch",
        "repeat",
        "runs",
        "duration",
        "throughput_tok_sec",
        "idle_power_w",
        "avg_power_w",
        "energy_run_j",
        "dynamic_energy_run_j"
    ])

    writer.writerows(results)


print("\n===== MEDIAN SUMMARY =====")

for name, _, _, _ in SCENARIOS:

    rows = [r for r in results if r[0] == name]

    print(
        f"{name:6s} | "
        f"{statistics.median(r[7] for r in rows):.1f} tok/s | "
        f"{statistics.median(r[9] for r in rows):.2f} W | "
        f"{statistics.median(r[10] for r in rows):.2f} J/run | "
        f"dynamic {statistics.median(r[11] for r in rows):.2f} J/run"
    )
