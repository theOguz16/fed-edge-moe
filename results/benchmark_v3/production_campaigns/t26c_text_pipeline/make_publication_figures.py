import csv
import json
import math
import statistics as stats
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("results/benchmark_v3/pipeline_dataset_exports")
exports = sorted(
    p for p in BASE.glob("t26_merged_*")
    if (p / "pipeline_pairs_528.csv").is_file()
)
assert exports, "Missing merged pipeline export"
SOURCE = exports[-1] / "pipeline_pairs_528.csv"
OUT = exports[-1] / "publication_figures"
OUT.mkdir(exist_ok=True)

with SOURCE.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 528
for r in rows:
    r["speed"] = float(
        r["speedup_pipeline_over_sequential"]
    )
    r["energy"] = float(
        r["energy_pipeline_over_sequential"]
    )
    assert math.isfinite(r["speed"]) and r["speed"] > 0
    assert math.isfinite(r["energy"]) and r["energy"] > 0

NAMES = {
    "resnet50": "ResNet50",
    "convnext_base": "ConvNeXt-Base",
    "convnext_large": "ConvNeXt-Large",
    "distilgpt2": "DistilGPT2",
    "qwen2_5_1_5b": "Qwen2.5-1.5B",
    "qwen3_1_7b": "Qwen3-1.7B",
}
models = list(NAMES)

by_model = defaultdict(list)
by_direction = defaultdict(list)
by_qos = defaultdict(list)
by_config = defaultdict(list)

for r in rows:
    by_model[r["model"]].append(r)
    by_direction[(r["model"], r["direction"])].append(r)
    by_qos[(r["model"], r["qos"])].append(r)
    key = tuple(r[k] for k in (
        "model", "direction", "cut", "mapping_bits",
        "microbatch", "qos", "precision",
        "batch_size", "context_tokens"
    ))
    by_config[key].append(r)

assert len(by_config) == 264
assert all(
    len(v) == 2 and {x["repeat"] for x in v} == {"1", "2"}
    for v in by_config.values()
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.dpi": 220,
})

COLORS = {
    "vision": "#2478A8",
    "text": "#D78A37",
}
FOOTNOTE = (
    "Apple M4 | FP32 | estimated CPU+GPU SoC energy "
    "(not wall power)"
)

def save(fig, name):
    fig.savefig(
        OUT / (name + ".png"),
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUT / (name + ".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("FIGURE:", name)

def med(values, field):
    return stats.median(x[field] for x in values)

def style(ax):
    ax.grid(axis="y", alpha=0.17)
    ax.set_axisbelow(True)

# Figure 1: Model-wise throughput
fig, ax = plt.subplots(figsize=(11, 5.4))
speed_vals = [med(by_model[m], "speed") for m in models]
bar_colors = [
    COLORS["vision"] if i < 3 else COLORS["text"]
    for i in range(6)
]
ax.bar(range(6), speed_vals, color=bar_colors, width=0.66)
ax.axhline(1, color="#555", ls="--", lw=1)
ax.set_xticks(range(6), [NAMES[x] for x in models],
              rotation=15, ha="right")
ax.set_ylabel("Median pipeline / sequential throughput")
ax.set_title("Inference pipeline speedup by model")
ax.set_ylim(0, max(speed_vals) * 1.19)
for i, value in enumerate(speed_vals):
    ax.text(i, value + 0.025, f"{value:.3f}×",
            ha="center", fontsize=9)
style(ax)
fig.text(0.11, 0.01, "528 paired cases; model n=48 or 96",
         fontsize=9, color="#666")
save(fig, "01_model_speedup")

# Figure 2: Model-wise estimated energy
fig, ax = plt.subplots(figsize=(11, 5.4))
energy_vals = [med(by_model[m], "energy") for m in models]
ax.bar(range(6), energy_vals, color=bar_colors, width=0.66)
ax.axhline(1, color="#B54646", ls="--", lw=1.6)
ax.set_xticks(range(6), [NAMES[x] for x in models],
              rotation=15, ha="right")
ax.set_ylabel("Median pipeline / sequential energy per item")
ax.set_title("Estimated SoC energy ratio by model")
ax.set_ylim(0, max(energy_vals) * 1.14)
for i, value in enumerate(energy_vals):
    ax.text(i, value + 0.02, f"{value:.3f}×",
            ha="center", fontsize=9)
style(ax)
fig.text(0.11, 0.01,
         "Below 1 = lower estimated energy; vision: J/image, text: J/token",
         fontsize=9, color="#666")
save(fig, "02_model_energy")

# Figure 3: All cases, speedup against energy ratio
fig, ax = plt.subplots(figsize=(9, 6.5))
for family in ("vision", "text"):
    sample = [r for r in rows if r["family"] == family]
    ax.scatter(
        [r["speed"] for r in sample],
        [r["energy"] for r in sample],
        s=24, alpha=0.50,
        color=COLORS[family], label=family.title(),
    )
ax.axvline(1, color="#555", ls="--", lw=1)
ax.axhline(1, color="#555", ls="--", lw=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Pipeline / sequential throughput (log scale)")
ax.set_ylabel("Pipeline / sequential estimated energy (log scale)")
ax.set_title("Performance–energy trade-off: all 528 cases")
ax.legend(frameon=False)
ax.grid(alpha=0.16)
fig.text(0.12, 0.01, FOOTNOTE,
         fontsize=9, color="#666")
save(fig, "03_speedup_vs_energy_scatter")

# Figure 4: Directional comparisons.
# ResNet50 has three distinct mapping/cut directions.
direction_entries = []
for (model, direction), sample in sorted(by_direction.items()):
    label = direction.replace("_", "→")
    label = label.replace("cpu→mps", "CPU→MPS")
    label = label.replace("mps→cpu", "MPS→CPU")
    direction_entries.append((
        f"{NAMES[model]} / {label}",
        med(sample, "speed"),
        med(sample, "energy"),
    ))

fig, ax = plt.subplots(figsize=(11, 8))
yy = np.arange(len(direction_entries))
ax.barh(
    yy, [x[1] for x in direction_entries],
    color="#347B9F", height=0.69
)
ax.axvline(1, color="#B54646", ls="--", lw=1.5)
ax.set_yticks(
    yy, [x[0] for x in direction_entries], fontsize=9
)
ax.invert_yaxis()
ax.set_xlabel("Median pipeline speedup")
ax.set_title("Speedup by model and execution direction")
ax.grid(axis="x", alpha=0.19)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "04_direction_speedup")

# Figure 5: QoS observational heatmap; exclude background.
qos_names = ["utility", "default", "user_initiated"]
matrix = np.array([
    [med(by_qos[(m, q)], "speed") for q in qos_names]
    for m in models
])
fig, ax = plt.subplots(figsize=(9, 5.7))
im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
               vmin=0.95, vmax=1.55)
ax.set_xticks(range(3), ["Utility", "Default", "User Initiated"])
ax.set_yticks(range(6), [NAMES[m] for m in models])
for i in range(6):
    for j in range(3):
        ax.text(j, i, f"{matrix[i, j]:.3f}×",
                ha="center", va="center", fontsize=9)
ax.set_title("Median speedup by QoS (background excluded)")
fig.colorbar(im, ax=ax, label="Throughput ratio")
fig.text(0.10, 0.01,
         "Observational comparison; native PyTorch thread QoS unverified",
         fontsize=9, color="#666")
fig.tight_layout()
save(fig, "05_qos_heatmap")

# Figure 6: Repeat-to-repeat stability
speed_diffs, energy_diffs = [], []
speed_flips = energy_flips = 0
for values in by_config.values():
    a, b = sorted(values, key=lambda r: int(r["repeat"]))
    speed_diffs.append(abs(a["speed"] - b["speed"]))
    energy_diffs.append(abs(a["energy"] - b["energy"]))
    speed_flips += (
        (a["speed"] - 1) * (b["speed"] - 1) < 0
    )
    energy_flips += (
        (a["energy"] - 1) * (b["energy"] - 1) < 0
    )
assert speed_flips == 42 and energy_flips == 113

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].hist(speed_diffs, bins=25, color="#347B9F")
axes[0].axvline(stats.median(speed_diffs),
                ls="--", color="black")
axes[0].set_title("Throughput speedup variability")
axes[0].set_xlabel("Absolute difference between repeats")
axes[0].set_ylabel("Configuration count")
axes[1].hist(energy_diffs, bins=25, color="#D78A37")
axes[1].axvline(stats.median(energy_diffs),
                ls="--", color="black")
axes[1].set_title("Energy ratio variability")
axes[1].set_xlabel("Absolute difference between repeats")
fig.suptitle(
    "Independent-repeat variability: 264 configurations\n"
    "Sign flips: speedup 42; energy 113",
    fontsize=12,
)
fig.tight_layout()
save(fig, "06_repeat_variability")

# Main results: all QoS versus non-background robustness
table = []
for m in models:
    full = by_model[m]
    filtered_configs = [
        v for v in by_config.values()
        if v[0]["model"] == m
        and v[0]["qos"] != "background"
    ]
    robust_speeds = [
        stats.median(r["speed"] for r in config)
        for config in filtered_configs
    ]
    robust_energies = [
        stats.median(r["energy"] for r in config)
        for config in filtered_configs
    ]
    table.append({
        "model": NAMES[m],
        "paired_cases": len(full),
        "all_median_speedup": round(med(full, "speed"), 6),
        "all_median_energy_ratio": round(med(full, "energy"), 6),
        "nonbackground_configurations": len(filtered_configs),
        "nonbackground_median_speedup": round(
            stats.median(robust_speeds), 6
        ),
        "nonbackground_median_energy_ratio": round(
            stats.median(robust_energies), 6
        ),
        "both_repeats_faster": sum(
            all(r["speed"] > 1 for r in config)
            for config in filtered_configs
        ),
        "both_repeats_energy_lower": sum(
            all(r["energy"] < 1 for r in config)
            for config in filtered_configs
        ),
    })

with (OUT / "table_01_main_results.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)

with (OUT / "table_01_main_results.md").open(
    "w", encoding="utf-8"
) as f:
    f.write(
        "| Model | n | Speedup | Energy ratio | "
        "Non-BG speedup | Non-BG energy | "
        "Both faster | Both energy lower |\n"
    )
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in table:
        f.write(
            f"| {r['model']} | {r['paired_cases']} | "
            f"{r['all_median_speedup']:.3f}× | "
            f"{r['all_median_energy_ratio']:.3f}× | "
            f"{r['nonbackground_median_speedup']:.3f}× | "
            f"{r['nonbackground_median_energy_ratio']:.3f}× | "
            f"{r['both_repeats_faster']}/"
            f"{r['nonbackground_configurations']} | "
            f"{r['both_repeats_energy_lower']}/"
            f"{r['nonbackground_configurations']} |\n"
        )

manifest = {
    "source_csv": str(SOURCE),
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "case_count": 528,
    "config_repeat_pairs": 264,
    "speedup_sign_flips": speed_flips,
    "energy_sign_flips": energy_flips,
    "plots": 6,
    "notes": [
        "Estimated CPU+GPU SoC energy, not wall power",
        "Vision throughput in images/s; text prefill in tokens/s",
        "QoS effects are observational, not causal",
        "Repeat comparisons have only two replicates",
        "Older DistilGPT2 measurements retained under legacy protocol",
    ],
}
(OUT / "figure_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("\nPUBLICATION FIGURES: PASS")
print("FIGURES: 6 PNG + 6 SVG")
print("TABLES: 1 CSV + 1 Markdown")
print("CASES:", len(rows))
print("SPEEDUP SIGN FLIPS:", speed_flips)
print("ENERGY SIGN FLIPS:", energy_flips)
print("OUTPUT:", OUT)
print("SQLITE MODIFIED: NO")
print("BENCHMARK RERUN: NO")
