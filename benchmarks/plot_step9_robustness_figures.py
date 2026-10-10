import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("results")
OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    path = ROOT / name
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED: {stem} (PDF + PNG)")


def pick_field(fieldnames, candidates):
    for name in candidates:
        if name in fieldnames:
            return name
    raise KeyError(
        "Could not find any of these fields: "
        + ", ".join(candidates)
    )


def short_device(name):
    mapping = {
        "Apple M4": "Apple M4",
        "Intel i7-11800H": "Intel i7-11800H",
        "RTX 3050 Laptop": "RTX 3050",
    }
    return mapping.get(name, name)


def short_service(service_id):
    mapping = {
        "vision_easy": "ResNet50",
        "vision_medium": "ConvNeXt-Base",
        "vision_hard": "ConvNeXt-Large",
        "text_easy": "DistilGPT2",
        "text_medium": "Qwen2.5-1.5B",
        "text_hard": "Qwen3-1.7B",
    }
    return mapping.get(service_id, service_id)


def normalize_decision(value):
    value = (value or "").strip()
    if not value or value.upper().startswith("ABSTAIN"):
        return "ABSTAIN"
    if value.lower() in {"fp16", "fp32"}:
        return value.lower()
    if "fp16" in value.lower():
        return "fp16"
    if "fp32" in value.lower():
        return "fp32"
    return value


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})

# ------------------------------------------------------------
# Figure 3: Step 9 energy ranking robustness
# ------------------------------------------------------------
energy = load("step9_energy_ranking_robustness.csv")
assert energy, "Empty step9_energy_ranking_robustness.csv"

fields = list(energy[0].keys())

adv_key = pick_field(fields, [
    "median_advantage_pct",
    "median_energy_advantage_pct",
    "advantage_pct",
])

crit_key = pick_field(fields, [
    "critical_symmetric_perturbation_pct",
    "critical_perturbation_pct",
    "critical_pct",
])

rel_key = pick_field(fields, [
    "observed_range_relation",
    "range_relation",
])

low_key = pick_field(fields, [
    "lower_median_candidate",
    "winner_candidate",
    "candidate_a",
])

high_key = pick_field(fields, [
    "higher_median_candidate",
    "loser_candidate",
    "candidate_b",
])

labels = []
advantages = []
criticals = []
relations = []

for row in energy:
    service = short_service(row.get("service_id", ""))
    profile = row.get("request_profile", "")
    device = short_device(row.get("device", ""))

    low = row.get(low_key, "")
    high = row.get(high_key, "")

    detail = ""
    if row["service_id"] == "vision_medium":
        low_t = low.split("threads=")[-1]
        high_t = high.split("threads=")[-1]
        detail = f"{low_t}t vs {high_t}t"
    elif "precision=" in low and "precision=" in high:
        low_p = low.split("precision=")[-1].split(";")[0].upper()
        high_p = high.split("precision=")[-1].split(";")[0].upper()
        detail = f"{low_p} vs {high_p}"

    label = f"{service}\n{profile}"
    if detail:
        label += f"\n{detail}"

    labels.append(label)
    advantages.append(float(row[adv_key]))
    criticals.append(float(row[crit_key]))
    relations.append(row[rel_key])

assert len(labels) == len(advantages) == len(criticals) == 6

x = list(range(len(labels)))
width = 0.36

fig, ax = plt.subplots(figsize=(11.5, 5.8))
bars1 = ax.bar(
    [i - width / 2 for i in x],
    advantages,
    width=width,
    color="#0072B2",
    label="Median energy advantage (%)",
)
bars2 = ax.bar(
    [i + width / 2 for i in x],
    criticals,
    width=width,
    color="#D55E00",
    label="Critical change (%)",
)

for bars in (bars1, bars2):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.6,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

for i, relation in enumerate(relations):
    label = (
        "Disjoint observed\nrepeat ranges"
        if "DISJOINT" in relation
        else "Overlapping observed\nrepeat ranges"
    )
    top = max(advantages[i], criticals[i])
    ax.text(
        i,
        top + 4.0,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

ax.set_xticks(x, labels)
ax.set_ylabel("Percentage (%)")
ax.set_title("Step 9: Energy ranking robustness")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, max(max(advantages), max(criticals)) + 16)
ax.grid(axis="y", alpha=0.25)
fig.text(
    0.01,
    0.01,
    "Critical change = opposing symmetric perturbation at energy tie; reversal occurs beyond it.\n"
    "These are sensitivity indicators, not empirical confidence intervals.",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "step9_energy_ranking_robustness")


# ------------------------------------------------------------
# Figure 4: Step 9 evidence-availability robustness
# ------------------------------------------------------------
rows = load("step9_evidence_availability.csv")
assert rows, "Empty step9_evidence_availability.csv"

fields = list(rows[0].keys())
decision_key = pick_field(fields, [
    "decision",
    "selected_precision",
    "selection",
    "scheduler_decision",
])

decisions = Counter(normalize_decision(r.get(decision_key, "")) for r in rows)
assert sum(decisions.values()) == len(rows)

status_fields = [
    name for name in fields
    if name.endswith("_feasibility")
]
candidate_status = Counter()

for row in rows:
    for field in status_fields:
        value = (row.get(field, "") or "").strip()
        if value in {"FEASIBLE", "UNVERIFIED", "INFEASIBLE"}:
            candidate_status[value] += 1

assert candidate_status, "No feasibility-status fields found"

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

# Left panel: final decision counts
decision_order = ["ABSTAIN", "fp16", "fp32"]
decision_colors = {
    "ABSTAIN": "#BDBDBD",
    "fp16": "#009E73",
    "fp32": "#E69F00",
}
decision_values = [decisions.get(k, 0) for k in decision_order]

bars = axes[0].bar(
    decision_order,
    decision_values,
    color=[decision_colors[k] for k in decision_order],
)
for bar, value in zip(bars, decision_values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        value + 3,
        str(value),
        ha="center",
        va="bottom",
        fontsize=9,
    )

axes[0].set_title("Scheduler decisions")
axes[0].set_ylabel("Count across 256 availability states")
axes[0].grid(axis="y", alpha=0.25)
axes[0].set_ylim(0, max(decision_values) + 35)

# Right panel: candidate feasibility counts
status_order = [k for k in ("FEASIBLE", "UNVERIFIED", "INFEASIBLE") if k in candidate_status]
status_colors = {
    "FEASIBLE": "#009E73",
    "UNVERIFIED": "#999999",
    "INFEASIBLE": "#CC79A7",
}
status_values = [candidate_status[k] for k in status_order]

bars = axes[1].bar(
    status_order,
    status_values,
    color=[status_colors[k] for k in status_order],
)
for bar, value in zip(bars, status_values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        value + 8,
        str(value),
        ha="center",
        va="bottom",
        fontsize=9,
    )

axes[1].set_title("Candidate feasibility outcomes")
axes[1].set_ylabel("Count across 512 candidate evaluations")
axes[1].grid(axis="y", alpha=0.25)
axes[1].set_ylim(0, max(status_values) + 60)

fig.suptitle("Step 9: Evidence-availability robustness", fontsize=13)
fig.text(
    0.01,
    0.01,
    "The 225 abstentions come from exhaustive enumeration of artificial missing-evidence combinations;\n"
    "they are not an operational failure rate.",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
save(fig, "step9_evidence_availability")

print("STEP 9 ROBUSTNESS FIGURES: PASS")
