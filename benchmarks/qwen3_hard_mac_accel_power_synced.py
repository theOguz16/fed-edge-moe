import csv
import re
import signal
import statistics
import subprocess
from pathlib import Path

MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"
REPEATS = 3
INTERVAL_MS = 100

PROFILES = {
    "light":  (128, 64, 1),
    "medium": (512, 128, 2),
    "heavy":  (1024, 128, 4),
}

OUT = Path("results/qwen3_hard_q4_mac_accel_power_synced.csv")

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

for profile, (pp, tg, npl) in PROFILES.items():
    for repeat in range(1, REPEATS + 1):

        cmd = [
            "llama-batched-bench",
            "-hf", MODEL,
            "-ngl", "999",
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

        power_file = Path(
            f"/tmp/qwen3_{profile}_r{repeat}_powermetrics.txt"
        )

        for line in proc.stdout:
            captured.append(line)

            # Model load tamamlandıktan sonra power sampling başlar.
            if pm is None and "llama_batched_bench:" in line:
                pf = power_file.open("w")

                pm = subprocess.Popen(
                    [
                        "sudo", "-n", "powermetrics",
                        "--samplers", "cpu_power",
                        "-i", str(INTERVAL_MS),
                    ],
                    stdout=pf,
                    stderr=subprocess.STDOUT,
                )

        proc.wait()

        if pm is None:
            raise RuntimeError("Benchmark start marker not found")

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
            raise RuntimeError("Benchmark row parse failed")

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
            raise RuntimeError(
                f"No power samples for {profile} R{repeat}"
            )

        mean_power = statistics.mean(powers)

        generated_tokens = tg * npl
        energy_run = mean_power * t_total
        energy_token = energy_run / generated_tokens

        rows.append({
            "profile": profile,
            "pp": pp,
            "tg": tg,
            "batch": npl,
            "repeat": repeat,
            "s_tg_tok_s": s_tg,
            "t_total_sec": t_total,
            "mean_power_w": mean_power,
            "energy_run_j": energy_run,
            "energy_token_j": energy_token,
            "samples": len(powers),
        })

        print(
            f"{profile:6} R{repeat} | "
            f"{s_tg:7.2f} tok/s | "
            f"{mean_power:6.2f} W | "
            f"{energy_run:7.2f} J/run | "
            f"{energy_token:.4f} J/token | "
            f"{len(powers)} samples"
        )

        power_file.unlink(missing_ok=True)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nMEDIANS")
print("-" * 100)

for profile in PROFILES:
    r = [x for x in rows if x["profile"] == profile]

    print(
        f"{profile:6} | "
        f"{statistics.median(x['s_tg_tok_s'] for x in r):7.2f} tok/s | "
        f"{statistics.median(x['mean_power_w'] for x in r):6.2f} W | "
        f"{statistics.median(x['energy_run_j'] for x in r):7.2f} J/run | "
        f"{statistics.median(x['energy_token_j'] for x in r):.4f} J/token | "
        f"samples={','.join(str(x['samples']) for x in r)}"
    )

print("\nSaved:", OUT)
