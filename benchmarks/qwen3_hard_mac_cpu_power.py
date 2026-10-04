import csv
import re
import signal
import statistics
import subprocess
from pathlib import Path

THREADS = [1, 4, 8]
REPEATS = 3

WORKLOADS = {
    "long_single": (128, 128, 1),
    "batch":       (128, 64, 4),
}

MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"
OUT = Path("results/qwen3_hard_q4_mac_cpu_power.csv")

POWER_RE = re.compile(
    r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW"
)

ROW_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)

subprocess.run(["sudo", "-n", "-v"], check=True)

rows = []

for workload, (pp, tg, npl) in WORKLOADS.items():
    for threads in THREADS:
        for repeat in range(1, REPEATS + 1):

            cmd = [
                "llama-batched-bench",
                "-hf", MODEL,
                "-ngl", "0",
                "-t", str(threads),
                "-tb", str(threads),
                "-npp", str(pp),
                "-ntg", str(tg),
                "-npl", str(npl),
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            captured = []
            pm = None
            pf = None

            for line in proc.stdout:
                captured.append(line)

                if pm is None and "llama_batched_bench:" in line:
                    power_file = Path(
                        f"results/qwen3_cpu_{workload}_{threads}t_r{repeat}_power.txt"
                    )

                    pf = power_file.open("w")

                    pm = subprocess.Popen(
                        [
                            "sudo", "-n", "powermetrics",
                            "--samplers", "cpu_power",
                            "-i", "500",
                        ],
                        stdout=pf,
                        stderr=subprocess.STDOUT,
                    )

            proc.wait()

            if pm is None:
                raise RuntimeError("Benchmark başlangıcı yakalanamadı")

            try:
                pm.send_signal(signal.SIGINT)
                pm.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pm.terminate()
                pm.wait()

            pf.close()

            text = "".join(captured)
            matches = ROW_RE.findall(text)

            if not matches:
                print(text)
                raise RuntimeError("Benchmark row parse edilemedi")

            m = matches[-1]

            s_tg = float(m[7])
            t_total = float(m[8])

            powers = [
                float(x) / 1000.0
                for x in POWER_RE.findall(
                    power_file.read_text(errors="replace")
                )
            ]

            if not powers:
                raise RuntimeError("Power sample bulunamadı")

            avg_power = statistics.mean(powers)
            generated_tokens = tg * npl

            energy_run = avg_power * t_total
            energy_token = energy_run / generated_tokens

            rows.append({
                "workload": workload,
                "threads": threads,
                "repeat": repeat,
                "s_tg_tok_s": s_tg,
                "t_total_sec": t_total,
                "avg_power_w": avg_power,
                "energy_run_j": energy_run,
                "energy_token_j": energy_token,
                "samples": len(powers),
            })

            print(
                f"{workload:12} {threads:2}t R{repeat} | "
                f"{s_tg:7.2f} tok/s | "
                f"{avg_power:6.2f} W | "
                f"{energy_token:.4f} J/token | "
                f"{len(powers)} samples"
            )

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS")
print("-" * 100)

for workload in WORKLOADS:
    for threads in THREADS:
        r = [
            x for x in rows
            if x["workload"] == workload
            and x["threads"] == threads
        ]

        print(
            f"{workload:12} {threads:2}t | "
            f"{statistics.median(x['s_tg_tok_s'] for x in r):7.2f} tok/s | "
            f"{statistics.median(x['avg_power_w'] for x in r):6.2f} W | "
            f"{statistics.median(x['energy_run_j'] for x in r):7.2f} J/run | "
            f"{statistics.median(x['energy_token_j'] for x in r):.4f} J/token"
        )

print("\nSaved:", OUT)
