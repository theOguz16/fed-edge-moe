import argparse
import csv
import re
import statistics
import subprocess
from pathlib import Path

MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"
RESULTS = Path("results")

CPU_WORKLOADS = {
    "long_single": (128, 128, 1),
    "batch": (128, 64, 4),
}

METAL_PROFILES = {
    "light": (128, 64, 1, 64),
    "medium": (512, 128, 2, 22),
    "heavy": (1024, 128, 4, 9),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend", choices=["cpu", "metal"], required=True
)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()

if args.repeats < 1:
    parser.error("--repeats must be >= 1")

RESULTS.mkdir(exist_ok=True)

OUT = RESULTS / f"qwen3_q4_mac_{args.backend}_memory.csv"

configs = []

if args.backend == "cpu":
    for workload, (ctx, out, batch) in CPU_WORKLOADS.items():
        for threads in [1, 4, 8]:
            cmd = [
                "llama-batched-bench",
                "-hf", MODEL,
                "-ngl", "0",
                "-t", str(threads),
                "-tb", str(threads),
                "-npp", str(ctx),
                "-ntg", str(out),
                "-npl", str(batch),
            ]
            configs.append(
                (workload, ctx, out, batch, threads, cmd)
            )
else:
    for profile, (ctx, out, batch, count) in METAL_PROFILES.items():
        cmd = [
            "llama-batched-bench",
            "-hf", MODEL,
            "-c", "8192",
            "-b", "2048",
            "-ub", "512",
            "-ngl", "99",
            "-npp", ",".join([str(ctx)] * count),
            "-ntg", str(out),
            "-npl", str(batch),
        ]
        configs.append(
            (profile, ctx, out, batch, "", cmd)
        )

def extract_bytes(text, label):
    match = re.search(
        rf"^\s*(\d+)\s+{re.escape(label)}\s*$",
        text,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"Missing memory metric: {label}")
    return int(match.group(1))

rows = []

for name, ctx, out, batch, threads, cmd in configs:
    for repeat in range(1, args.repeats + 1):
        print(
            f"{args.backend} | {name} | "
            f"threads={threads or '-'} | "
            f"repeat={repeat}/{args.repeats}",
            flush=True,
        )

        result = subprocess.run(
            ["/usr/bin/time", "-l", *cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )

        if result.returncode:
            print(result.stdout[-2500:])
            raise RuntimeError(
                f"Benchmark failed: {name}, repeat {repeat}"
            )

        rss_bytes = extract_bytes(
            result.stdout, "maximum resident set size"
        )
        footprint_bytes = extract_bytes(
            result.stdout, "peak memory footprint"
        )

        rows.append({
            "backend": args.backend,
            "workload": name if args.backend == "cpu" else "",
            "profile": name if args.backend == "metal" else "",
            "context_tokens": ctx,
            "output_tokens": out,
            "batch": batch,
            "threads": threads,
            "precision": "q4_k_m",
            "repeat": repeat,
            "peak_rss_mib": rss_bytes / 1024**2,
            "peak_footprint_mib": footprint_bytes / 1024**2,
            "measurement": "macos_time_l",
        })

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nMEMORY SUMMARY")
for name, _, _, _, threads, _ in configs:
    selected = [
        r for r in rows
        if (
            (r["workload"] if args.backend == "cpu"
             else r["profile"]) == name
            and r["threads"] == threads
        )
    ]

    rss = [r["peak_rss_mib"] for r in selected]
    footprint = [
        r["peak_footprint_mib"] for r in selected
    ]

    print(
        f"{name:12s} threads={str(threads or '-'):>2s} | "
        f"RSS median={statistics.median(rss):8.1f} MiB | "
        f"RSS max={max(rss):8.1f} MiB | "
        f"footprint max={max(footprint):8.1f} MiB"
    )

print(f"\nSaved: {OUT}")
