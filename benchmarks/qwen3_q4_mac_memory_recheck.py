import csv
import re
import statistics
import subprocess
from pathlib import Path

OUT = Path(
    "results/qwen3_q4_mac_cpu_memory_recheck.csv"
)

CMD = [
    "llama-batched-bench",
    "-hf", "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M",
    "-ngl", "0",
    "-t", "1",
    "-tb", "1",
    "-npp", "128",
    "-ntg", "128",
    "-npl", "1",
]

rows = []

for repeat in range(1, 6):
    print(f"Recheck {repeat}/5...", flush=True)

    result = subprocess.run(
        ["/usr/bin/time", "-l", *CMD],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Repeat {repeat} failed:\n{result.stdout[-1500:]}"
        )

    def extract(label):
        match = re.search(
            rf"^\s*(\d+)\s+{re.escape(label)}\s*$",
            result.stdout,
            re.MULTILINE,
        )
        if not match:
            raise RuntimeError(f"Missing: {label}")
        return int(match.group(1)) / 1024**2

    rss = extract("maximum resident set size")
    footprint = extract("peak memory footprint")

    rows.append({
        "backend": "cpu",
        "workload": "long_single",
        "threads": 1,
        "context_tokens": 128,
        "output_tokens": 128,
        "batch": 1,
        "precision": "q4_k_m",
        "repeat": repeat,
        "peak_rss_mib": rss,
        "peak_footprint_mib": footprint,
    })

    print(
        f"RSS={rss:.1f} MiB | "
        f"footprint={footprint:.1f} MiB"
    )

OUT.parent.mkdir(exist_ok=True)

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

values = [r["peak_rss_mib"] for r in rows]

spread = (
    (max(values) - min(values))
    / statistics.median(values) * 100
)

print("\nRECHECK SUMMARY")
print(f"RSS min    : {min(values):.1f} MiB")
print(f"RSS median : {statistics.median(values):.1f} MiB")
print(f"RSS max    : {max(values):.1f} MiB")
print(f"Spread     : {spread:.1f}%")
print(f"Saved      : {OUT}")
