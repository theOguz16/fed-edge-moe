from collections import Counter
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "savefig.dpi": 300})

def load(name):
    with Path("results", name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{extension}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("SAVED:", name, "(PDF + PNG)")

# Figure 1: Step 8 policy ablation.
rows = load("step8_policy_ablation.csv")
policies = ("energy_only", "optimistic_qos_first", "evidence_aware_qos_first")
labels = ("Energy-only", "Optimistic QoS-first", "Evidence-aware QoS-first")
summary = {}
for policy in policies:
    subset = [r for r in rows if r["policy"] == policy]
    assert len(subset) == 14
    summary[policy] = (
        sum(r["decision"] == "SELECTED" for r in subset),
        sum(int(r["known_fail_selected"]) for r in subset),
        sum(int(r["unknown_or_unconfigured_selected"]) for r in subset),
    )
assert [summary[p] for p in policies] == [(14, 6, 4), (10, 0, 3), (7, 0, 0)]
fig, ax = plt.subplots(figsize=(8.2, 4.1))
categories = (("Selected", "#0072B2"), ("Selected with FAIL", "#D55E00"), ("Selected with UNKNOWN", "#CC79A7"))
for j, (name, color) in enumerate(categories):
    locations = [i + (j - 1) * 0.24 for i in range(3)]
    values = [summary[p][j] for p in policies]
    ax.barh(locations, values, height=0.22, color=color, label=name)
    for location, value in zip(locations, values):
        if value:
            ax.text(value + 0.15, location, str(value), va="center", fontsize=9)
ax.set_yticks(range(3), labels)
ax.invert_yaxis()
ax.set_xlim(0, 16)
ax.set_xlabel("Count across 14 synthetic scenarios per policy")
ax.set_title("Offline scheduler policy ablation")
ax.legend(frameon=False, loc="lower right")
fig.text(0.04, 0.01, "Risk flags may overlap; these are not operational failure rates.", fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "step8_policy_ablation")

# Figure 2: Step 9 joint latency-quality decision regions.
rows = load("step9_latency_quality_interaction.csv")
steps = ("below_lower", "at_lower", "between", "at_upper", "above_upper")
lookup = {(r["latency_step"], r["quality_step"]): r for r in rows}
assert len(rows) == len(lookup) == 25
codes = {"": 0, "fp16": 1, "fp32": 2}
grid = [[codes[lookup[(l, q)]["selected_precision"]] for q in steps] for l in steps]
assert Counter(v for line in grid for v in line) == {0: 13, 1: 8, 2: 4}
latencies = [float(lookup[(l, steps[0])]["deadline_s"]) * 1000 for l in steps]
qualities = [float(lookup[(steps[0], q)]["minimum_top5_accuracy_pct"]) for q in steps]
colors = ("#DDDDDD", "#009E73", "#E69F00")
fig, ax = plt.subplots(figsize=(8.0, 5.0))
ax.imshow(grid, cmap=ListedColormap(colors), vmin=-0.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(5), [f"{v:.4f}" for v in qualities], rotation=20)
ax.set_yticks(range(5), [f"{v:.2f}" for v in latencies])
ax.set_xlabel("Minimum Top-5 accuracy requirement (%)")
ax.set_ylabel("Maximum latency requirement (ms)")
ax.set_title("Synthetic latency-quality decision matrix")
for i in range(5):
    for j in range(5):
        ax.text(j, i, ("ABSTAIN", "FP16", "FP32")[grid[i][j]], ha="center", va="center", fontsize=8)
legend = [Patch(facecolor=color, label=label) for color, label in zip(colors, ("ABSTAIN", "FP16", "FP32"))]
ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
fig.tight_layout()
save(fig, "step9_latency_quality_heatmap")

print("PAPER FIGURES INITIAL GENERATION: PASS")
