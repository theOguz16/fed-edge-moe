import csv, os, re, signal, statistics, subprocess, threading, time
from pathlib import Path

MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"

PROFILES = {
    "light":  (128, 64, 1, 64),
    "medium": (512, 128, 2, 22),
    "heavy":  (1024, 128, 4, 9),
}

REPEATS = 3
OUTFILE = Path("results/qwen3_hard_q4_mac_power.csv")

ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
)

POWER_RE = re.compile(
    r"Combined Power.*?:\s*([\d.]+)\s*mW",
    re.I
)

# sudo ticket must already exist
subprocess.run(["sudo", "-n", "-v"], check=True)

stop_sudo = threading.Event()

def sudo_keepalive():
    while not stop_sudo.wait(30):
        subprocess.run(
            ["sudo", "-n", "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

threading.Thread(target=sudo_keepalive, daemon=True).start()

rows_out = []

for profile, (ctx, out, batch, count) in PROFILES.items():
    npp = ",".join([str(ctx)] * count)

    for repeat in range(1, REPEATS + 1):
        print(f"\n{profile.upper()} R{repeat}")

        bench_file = Path(
            f"results/qwen3_hard_mac_{profile}_r{repeat}.txt"
        )
        power_file = Path(
            f"results/qwen3_hard_mac_{profile}_r{repeat}_power.txt"
        )

        cmd = [
            "llama-batched-bench",
            "-hf", MODEL,
            "-c", "8192",
            "-b", "2048",
            "-ub", "512",
            "-ngl", "99",
            "-npp", npp,
            "-ntg", str(out),
            "-npl", str(batch),
        ]

        with power_file.open("w") as pf:
            pm = subprocess.Popen(
                [
                    "sudo", "-n", "powermetrics",
                    "--samplers", "cpu_power",
                    "-i", "100",
                ],
                stdout=pf,
                stderr=subprocess.STDOUT,
            )

            time.sleep(0.3)

            start = time.perf_counter()

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )

            wall = time.perf_counter() - start

            try:
                pm.send_signal(signal.SIGINT)
                pm.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pm.terminate()
                pm.wait()

        bench_file.write_text(result.stdout)

        if result.returncode != 0:
            raise RuntimeError(
                f"{profile} R{repeat} benchmark failed"
            )

        bench_rows = []

        for line in result.stdout.splitlines():
            m = ROW_RE.match(line.strip())
            if m:
                bench_rows.append({
                    "tg": int(m.group(2)),
                    "batch": int(m.group(3)),
                    "s_tg": float(m.group(8)),
                })

        powers = [
            float(x) / 1000.0
            for x in POWER_RE.findall(
                power_file.read_text(errors="replace")
            )
        ]

        if not bench_rows:
            raise RuntimeError("No benchmark rows parsed")

        if not powers:
            raise RuntimeError(
                "No Combined Power samples parsed"
            )

        # trim start/end transition samples
        if len(powers) > 10:
            powers = powers[2:-2]

        median_tg = statistics.median(
            x["s_tg"] for x in bench_rows
        )

        avg_power = statistics.mean(powers)

        generated_tokens = sum(
            x["tg"] * x["batch"] for x in bench_rows
        )

        energy_total = avg_power * wall
        energy_run = energy_total / len(bench_rows)
        energy_token = energy_total / generated_tokens

        row = {
            "profile": profile,
            "repeat": repeat,
            "context": ctx,
            "output": out,
            "batch": batch,
            "runs": len(bench_rows),
            "wall_sec": wall,
            "median_generation_tok_sec": median_tg,
            "first_generation_tok_sec": bench_rows[0]["s_tg"],
            "last_generation_tok_sec": bench_rows[-1]["s_tg"],
            "avg_power_w": avg_power,
            "energy_run_j": energy_run,
            "energy_token_j": energy_token,
            "samples": len(powers),
        }

        rows_out.append(row)

        print(
            f"{median_tg:.2f} tok/s | "
            f"{avg_power:.2f} W | "
            f"{energy_run:.2f} J/run | "
            f"{energy_token:.4f} J/token | "
            f"samples={len(powers)}"
        )

stop_sudo.set()

with OUTFILE.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
    writer.writeheader()
    writer.writerows(rows_out)

print("\nMEDIANS")
print("-" * 85)

for profile in PROFILES:
    r = [x for x in rows_out if x["profile"] == profile]

    print(
        f"{profile:6} | "
        f"{statistics.median(x['median_generation_tok_sec'] for x in r):7.2f} tok/s | "
        f"{statistics.median(x['avg_power_w'] for x in r):6.2f} W | "
        f"{statistics.median(x['energy_run_j'] for x in r):7.2f} J/run | "
        f"{statistics.median(x['energy_token_j'] for x in r):.4f} J/token"
    )

print("\nSaved:", OUTFILE)
