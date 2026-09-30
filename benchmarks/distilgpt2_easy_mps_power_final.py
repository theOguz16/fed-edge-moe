import csv
import re
import subprocess
import tempfile
import time
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "distilgpt2"
REPEATS = 3

COOLDOWN = 15
IDLE_DURATION = 10
WORKLOAD_DURATION = 60
INTERVAL_MS = 100

SCENARIOS = [
    ("light", 32, 32, 1),
    ("medium", 128, 64, 4),
    ("heavy", 512, 128, 8),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL).to("mps")
model.eval()


def make_inputs(ctx, batch):
    text = (
        "Artificial intelligence on edge devices enables "
        "efficient distributed machine learning systems. "
    )

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = (ids * ((ctx // len(ids)) + 2))[:ctx]

    input_ids = torch.tensor(
        [ids] * batch,
        dtype=torch.long,
        device="mps"
    )

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids)
    }


def extract_power(text):
    vals = re.findall(
        r"Combined Power.*?:\s*([\d.]+)\s*mW",
        text
    )
    return [float(v) / 1000.0 for v in vals]


def measure_idle(seconds):
    samples = int(seconds * 1000 / INTERVAL_MS)

    result = subprocess.run(
        [
            "sudo", "powermetrics",
            "-i", str(INTERVAL_MS),
            "-n", str(samples),
            "--samplers", "cpu_power,gpu_power"
        ],
        capture_output=True,
        text=True
    )

    return extract_power(result.stdout + result.stderr)


results = []
raw = []

for name, ctx, output, batch in SCENARIOS:

    inputs = make_inputs(ctx, batch)

    print(f"\n===== {name.upper()} =====")

    for repeat in range(1, REPEATS + 1):

        print(f"Repeat {repeat}/{REPEATS}")

        # warm-up
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=output,
                min_new_tokens=output,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.mps.synchronize()

        print("Cooldown...")
        time.sleep(COOLDOWN)

        print("Idle power...")
        idle_samples = measure_idle(IDLE_DURATION)

        if not idle_samples:
            raise RuntimeError("Idle power sample bulunamadı.")

        idle_mean = statistics.mean(idle_samples)

        for i, value in enumerate(idle_samples):
            raw.append([name, repeat, "idle", i, value])

        print(f"Idle mean: {idle_mean:.2f} W")

        samples = int(
            WORKLOAD_DURATION * 1000 / INTERVAL_MS
        )

        print("Workload...")

        with tempfile.NamedTemporaryFile(
            mode="w+",
            delete=True
        ) as tmp:

            proc = subprocess.Popen(
                [
                    "sudo", "powermetrics",
                    "-i", str(INTERVAL_MS),
                    "-n", str(samples),
                    "--samplers", "cpu_power,gpu_power"
                ],
                stdout=tmp,
                stderr=subprocess.STDOUT,
                text=True
            )

            start = time.perf_counter()
            runs = 0

            while time.perf_counter() - start < WORKLOAD_DURATION:

                with torch.inference_mode():
                    model.generate(
                        **inputs,
                        max_new_tokens=output,
                        min_new_tokens=output,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                torch.mps.synchronize()
                runs += 1

            elapsed = time.perf_counter() - start

            proc.wait()

            tmp.seek(0)
            power_text = tmp.read()

        workload_samples = extract_power(power_text)

        if not workload_samples:
            raise RuntimeError("Workload power sample bulunamadı.")

        for i, value in enumerate(workload_samples):
            raw.append([name, repeat, "workload", i, value])

        avg_power = statistics.mean(workload_samples)
        median_power = statistics.median(workload_samples)

        throughput = (
            runs * batch * output
        ) / elapsed

        energy_run = (
            avg_power * elapsed
        ) / runs

        dynamic_power = max(
            avg_power - idle_mean,
            0
        )

        dynamic_energy = (
            dynamic_power * elapsed
        ) / runs

        print(
            f"runs={runs} | "
            f"{throughput:.1f} tok/s | "
            f"mean={avg_power:.2f} W | "
            f"median={median_power:.2f} W | "
            f"{energy_run:.2f} J/run"
        )

        results.append([
            name, ctx, output, batch,
            repeat, runs, elapsed,
            throughput,
            idle_mean,
            avg_power,
            median_power,
            energy_run,
            dynamic_energy,
            len(workload_samples)
        ])


with open(
    "results/distilgpt2_easy_mps_power_final.csv",
    "w",
    newline=""
) as f:

    w = csv.writer(f)

    w.writerow([
        "scenario",
        "context",
        "output",
        "batch",
        "repeat",
        "runs",
        "duration_sec",
        "throughput_tok_sec",
        "idle_mean_w",
        "workload_mean_w",
        "workload_median_w",
        "energy_per_run_j",
        "dynamic_energy_per_run_j",
        "power_samples"
    ])

    w.writerows(results)


with open(
    "results/distilgpt2_easy_mps_power_final_raw.csv",
    "w",
    newline=""
) as f:

    w = csv.writer(f)

    w.writerow([
        "scenario",
        "repeat",
        "phase",
        "sample",
        "combined_power_w"
    ])

    w.writerows(raw)


print("\n===== FINAL MEDIAN SUMMARY =====")

for name, _, _, _ in SCENARIOS:

    rows = [r for r in results if r[0] == name]

    print(
        f"{name:6s} | "
        f"{statistics.median(r[7] for r in rows):.1f} tok/s | "
        f"{statistics.median(r[9] for r in rows):.2f} W | "
        f"{statistics.median(r[11] for r in rows):.2f} J/run"
    )
