import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("results")
OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

with (ROOT / "scheduler_pareto_candidates.csv").open(newline="", encoding="utf-8") as f:
    data = list(csv.DictReader(f))

assert len(data) == 57

GROUPS = [
    (
        "ResNet50 medium | Apple M4 / MPS",
        {"service_id": "vision_easy", "request_profile": "medium", "device": "Apple M4", "backend": "mps"},
        2, "combined_soc",
    ),
    (
        "ConvNeXt-Base medium | Intel i7 CPU",
        {"service_id": "vision_medium", "request_profile": "medium", "device": "Intel i7-11800H", "backend": "cpu"},
        3, "cpu_package",
    ),
]

PARETO = {
    "OBSERVED_NONDOMINATED": ("o", "#0072B2", "Observed nondominated"),
    "OBSERVED_DOMINATED": ("X", "#D55E00", "Observed dominated"),
}

groups = []
for name, filters, expected, boundary in GROUPS:
    rows = [r for r in data if all(r[k] == v for k, v in filters.items())]
    assert len(rows) == expected, (name, len(rows))
    assert {r["energy_boundary"] for r in rows} == {boundary}
    for field in ("resolution", "batch", "context_tokens", "output_tokens", "throughput_unit", "throughput_semantics", "latency_semantics"):
        assert len({r[field] for r in rows}) == 1, (name, field)
    assert {r["throughput_unit"] for r in rows} == {"img/s"}
    for r in rows:
        for field in ("latency_sec", "throughput", "energy_per_item_j"):
            value = float(r[field])
            assert math.isfinite(value) and value > 0, (name, field)
        assert r["pareto_status"] in PARETO, (name, r["pareto_status"])
    groups.append((name, rows, boundary))

def point_label(r):
    if r["device"] == "Apple M4":
        return r["precision"].upper()
    return str(int(float(r["threads"]))) + (" thread" if int(float(r["threads"])) == 1 else " threads")

plt.rcParams.update({
    "font.size": 10,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})

def make_plot(field, multiplier, xlabel, stem):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for ax, (name, rows, boundary) in zip(axes, groups):
        for r in rows:
            x = float(r[field]) * multiplier
            y = float(r["energy_per_item_j"])
            marker, color, _ = PARETO[r["pareto_status"]]
            ax.scatter(x, y, marker=marker, s=120, color=color, zorder=3)
            offset = (7, 7)
            align = "left"
            if field == "latency_sec" and r["device"] == "Intel i7-11800H":
                n = int(float(r["threads"]))
                if n == 8:
                    offset, align = (-15, 18), "right"
                elif n == 16:
                    offset, align = (18, -16), "left"
            ax.annotate(point_label(r), (x, y), xytext=offset,
                        textcoords="offset points", fontsize=9, ha=align)
        ax.set_title(name + "\nEnergy boundary: " + boundary, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Energy per image (J)")
        ax.grid(alpha=0.22)
        ax.margins(x=0.28, y=0.32)
        ax.set_axisbelow(True)
    handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor=color, markersize=8, label=label)
        for marker, color, label in PARETO.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.09))
    fig.text(0.5, 0.02, "Measured point estimates; separate energy boundaries. Pareto status is descriptive, not QoS certification.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    for ext in ("png", "pdf"):
        destination = OUT / (stem + "." + ext)
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        assert destination.stat().st_size > 0
    plt.close(fig)
    print("SAVED:", stem, "(PDF + PNG)")

print("=== STEP 7 REPRESENTATIVE GROUP AUDIT ===")
for name, rows, boundary in groups:
    print(name, "|", len(rows), "candidates |", boundary)
    for r in rows:
        print(" ", point_label(r), "|", r["pareto_status"], "|", round(float(r["energy_per_item_j"]), 6), "J/image")

make_plot("latency_sec", 1000, "Latency (ms)", "step7_energy_latency_by_boundary")
make_plot("throughput", 1, "Throughput (images/s)", "step7_energy_throughput_by_boundary")
print("STEP 7 HARDWARE FIGURES: PASS")
