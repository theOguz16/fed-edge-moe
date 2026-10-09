import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

SRC = Path("results/scheduler_candidate_registry.csv")
OUT = Path("results/scheduler_operating_ranges.csv")

def num(v):
    try:
        if v in ("", None, "nan", "None"):
            return None
        return float(v)
    except Exception:
        return None

with SRC.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows = [
    r for r in rows
    if r.get("scheduler_core") == "1"
]

groups = defaultdict(list)

for r in rows:
    key = (
        r["modality"],
        r["difficulty"],
        r["model"],
    )
    groups[key].append(r)

out_rows = []

for (modality, difficulty, model), group in sorted(groups.items()):
    throughput = [
        num(r["throughput"])
        for r in group
        if num(r["throughput"]) is not None
    ]

    latency = [
        num(r["latency_sec"])
        for r in group
        if num(r["latency_sec"]) is not None
    ]

    energy = [
        num(r["energy_per_item_j"])
        for r in group
        if num(r["energy_per_item_j"]) is not None
    ]

    memory = []

    for r in group:
        vals = [
            num(r.get("memory_peak_mb")),
            num(r.get("device_memory_used_mb")),
            num(r.get("memory_allocated_mb")),
        ]
        vals = [v for v in vals if v is not None]

        if vals:
            memory.append(max(vals))

    row = {
        "modality": modality,
        "difficulty": difficulty,
        "model": model,
        "candidate_count": len(group),

        "throughput_min": min(throughput) if throughput else "",
        "throughput_median": statistics.median(throughput) if throughput else "",
        "throughput_max": max(throughput) if throughput else "",

        "latency_min_s": min(latency) if latency else "",
        "latency_median_s": statistics.median(latency) if latency else "",
        "latency_max_s": max(latency) if latency else "",

        "energy_min_j_item": min(energy) if energy else "",
        "energy_median_j_item": statistics.median(energy) if energy else "",
        "energy_max_j_item": max(energy) if energy else "",

        "memory_min_mb": min(memory) if memory else "",
        "memory_median_mb": statistics.median(memory) if memory else "",
        "memory_max_mb": max(memory) if memory else "",
        "memory_candidate_count": len(memory),
    }

    out_rows.append(row)

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=out_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(out_rows)

print(
    f"{'WORKLOAD':43s} "
    f"{'N':>3s} "
    f"{'THROUGHPUT[min/med/max]':>29s} "
    f"{'ENERGY[min/med/max]':>27s} "
    f"{'MEM N':>5s}"
)
print("-" * 115)

for r in out_rows:
    name = (
        f"{r['modality']}/"
        f"{r['difficulty']}/"
        f"{r['model']}"
    )

    print(
        f"{name:43s} "
        f"{r['candidate_count']:3d} "
        f"{r['throughput_min']:.3f}/"
        f"{r['throughput_median']:.3f}/"
        f"{r['throughput_max']:.3f} "
        f"{r['energy_min_j_item']:.4f}/"
        f"{r['energy_median_j_item']:.4f}/"
        f"{r['energy_max_j_item']:.4f} "
        f"{r['memory_candidate_count']:5d}"
    )

print(f"\nSaved: {OUT}")
