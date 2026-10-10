import csv
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path("results/scheduler_pareto_group_summary.csv")
OUT = Path("figures/paper")
OUT.mkdir(parents=True, exist_ok=True)

with SOURCE.open(newline="", encoding="utf-8") as f:
    groups = list(csv.DictReader(f))

assert len(groups) == 37
for row in groups:
    for field in ("candidate_count", "observed_nondominated", "observed_dominated", "not_evaluable"):
        row[field] = int(row[field])
    assert row["candidate_count"] == row["observed_nondominated"] + row["observed_dominated"] + row["not_evaluable"]

single = [r for r in groups if r["candidate_count"] == 1]
multi = [r for r in groups if r["candidate_count"] > 1]
assert len(single) == 18 and len(multi) == 19
assert sum(r["candidate_count"] for r in groups) == 57
assert sum(r["not_evaluable"] for r in groups) == 0

trivial = sum(r["observed_nondominated"] for r in single)
non = sum(r["observed_nondominated"] for r in multi)
dom = sum(r["observed_dominated"] for r in multi)
assert (trivial, non, dom) == (18, 23, 16)

coverage = Counter(r["repeat_evidence_coverage"] for r in multi)
assert coverage == {"COMPLETE_N3_DESCRIPTIVE": 4, "NO_REPEAT_PAIRWISE_EVIDENCE": 15}

plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "savefig.dpi": 300})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))

ax = axes[0]
labels = ["Singleton: nondominated by default", "Multi-candidate: nondominated", "Multi-candidate: dominated"]
values = [trivial, non, dom]
colors = ["#999999", "#0072B2", "#D55E00"]
bars = ax.barh(labels, values, color=colors, height=0.58)
ax.invert_yaxis()
ax.set_xlim(0, 27)
ax.set_xlabel("Canonical candidates (count)")
ax.set_title("Group-local Pareto classification")
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)
for bar, value in zip(bars, values):
    ax.text(value + 0.4, bar.get_y() + bar.get_height() / 2, str(value), va="center")

ax = axes[1]
labels = ["Complete pairwise repeats (n=3)", "No pairwise repeat evidence"]
values = [coverage["COMPLETE_N3_DESCRIPTIVE"], coverage["NO_REPEAT_PAIRWISE_EVIDENCE"]]
bars = ax.barh(labels, values, color=["#009E73", "#999999"], height=0.58)
ax.invert_yaxis()
ax.set_xlim(0, 19)
ax.set_xlabel("Multi-candidate groups (count)")
ax.set_title("Energy repeat-evidence coverage")
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)
for bar, value in zip(bars, values):
    ax.text(value + 0.4, bar.get_y() + bar.get_height() / 2, str(value), va="center")

fig.suptitle("Step 7: Pareto comparison coverage", fontsize=13)
fig.text(0.5, 0.025, "Singletons have no within-group competitor. Pareto status is descriptive, not QoS certification.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.10, 1, 0.94))

for ext in ("png", "pdf"):
    path = OUT / ("step7_pareto_coverage_summary." + ext)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    assert path.is_file() and path.stat().st_size > 0
    print("SAVED:", path)
plt.close(fig)

print("Groups: 37 | Singleton: 18 | Multi-candidate: 19")
print("Candidates: 57 | Nondominated: 41 | Dominated: 16")
print("Complete repeat evidence: 4/19 multi-candidate groups")
print("STEP 7 PARETO SUMMARY FIGURE: PASS")
