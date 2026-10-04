import csv
import re
import statistics
import subprocess

THREADS = [1, 2, 4, 8, 16]
REPEATS = 3

WORKLOADS = {
    "long_single": (128, 128, 1),
    "batch":       (128, 64, 4),
}

EXE = r"tools\llama-cuda\bin\llama-batched-bench.exe"
MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"

OUT = "results/qwen3_hard_q4_msi_cpu_scaling.csv"

ROW_RE = re.compile(
    r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)

rows = []

for workload, (pp, tg, npl) in WORKLOADS.items():

    print(f"\n{'='*70}")
    print(f"{workload}: PP{pp} TG{tg} B{npl}")
    print("="*70)

    for threads in THREADS:

        tg_vals = []
        total_vals = []

        for repeat in range(1, REPEATS + 1):

            cmd = [
                EXE,
                "-hf", MODEL,
                "-ngl", "0",
                "-t", str(threads),
                "-tb", str(threads),
                "-npp", str(pp),
                "-ntg", str(tg),
                "-npl", str(npl),
            ]

            p = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            matches = ROW_RE.findall(p.stdout)

            if not matches:
                print(p.stdout)
                raise RuntimeError("Benchmark row parse edilemedi")

            m = matches[-1]

            t_pp = float(m[4])
            s_pp = float(m[5])
            t_tg = float(m[6])
            s_tg = float(m[7])
            t_total = float(m[8])
            s_total = float(m[9])

            rows.append({
                "workload": workload,
                "pp": pp,
                "tg": tg,
                "batch": npl,
                "threads": threads,
                "repeat": repeat,
                "t_pp_sec": t_pp,
                "s_pp_tok_s": s_pp,
                "t_tg_sec": t_tg,
                "s_tg_tok_s": s_tg,
                "t_total_sec": t_total,
                "s_total_tok_s": s_total,
            })

            tg_vals.append(s_tg)
            total_vals.append(t_total)

            print(
                f"{threads:2}t R{repeat} | "
                f"S_TG {s_tg:7.2f} tok/s | "
                f"T {t_total:7.3f}s"
            )

        print(
            f"MEDIAN {threads:2}t | "
            f"S_TG {statistics.median(tg_vals):7.2f} tok/s | "
            f"T {statistics.median(total_vals):7.3f}s"
        )

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nSaved:", OUT)
