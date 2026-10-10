import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("results")
REPORT = Path("reports/step7_scheduler_experimental_summary.md")

SOURCES = {
    "registry": "scheduler_candidate_registry.csv",
    "applicability": "service_candidate_quality_applicability.csv",
    "feasibility": "service_feasibility_baseline.csv",
    "pareto": "scheduler_pareto_candidates.csv",
    "groups": "scheduler_pareto_group_summary.csv",
    "uncertainty": "energy_uncertainty_summary.csv",
    "pairwise": "energy_pairwise_repeat_evidence.csv",
    "latency": "qos_latency_sensitivity_resnet50_m4_medium.csv",
    "cpu_latency": "qos_latency_sensitivity_convnext_cpu_medium_partial.csv",
    "throughput": "qos_throughput_sensitivity_resnet50_m4_medium.csv",
    "quality": "qos_quality_sensitivity_resnet50_m4_medium.csv",
}

def load(filename):
    with (ROOT / filename).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

data = {name: load(filename) for name, filename in SOURCES.items()}

registry = data["registry"]
app = data["applicability"]
feas = data["feasibility"]
pareto = data["pareto"]
groups = data["groups"]
uncertainty = data["uncertainty"]
pairwise = data["pairwise"]

assert len(registry) == 385
assert sum(r["scheduler_core"] == "1" for r in registry) == 109
assert len(app) == len(feas) == len(pareto) == 57
assert len(groups) == 37
assert len(uncertainty) == 9
assert len(pairwise) == 6

quality_counts = Counter(
    r["quality_evidence_status"] for r in app
)
pareto_counts = Counter(
    r["pareto_status"] for r in pareto
)
feas_counts = Counter(
    r["feasibility_status"] for r in feas
)

assert quality_counts == {
    "SUPPORTED": 39,
    "CONDITIONAL": 6,
    "UNVERIFIED": 12,
}
assert pareto_counts == {
    "OBSERVED_NONDOMINATED": 38,
    "OBSERVED_DOMINATED": 16,
    "NOT_EVALUABLE_MISSING_METRICS": 3,
}
assert feas_counts == {"UNVERIFIED": 57}

multi = [r for r in groups if int(r["candidate_count"]) > 1]
repeat_supported = [
    r for r in multi
    if r["repeat_evidence_coverage"] == "COMPLETE_N3_DESCRIPTIVE"
]
assert len(multi) == 19
assert len(repeat_supported) == 4

sensitivity_counts = {
    "Latency — ResNet50": (data["latency"], 3),
    "Latency — Intel CPU (partial)": (data["cpu_latency"], 4),
    "Throughput — ResNet50": (data["throughput"], 3),
    "Quality — ResNet50": (data["quality"], 4),
}

for name, (rows, expected_scenarios) in sensitivity_counts.items():
    assert len({r["scenario"] for r in rows}) == expected_scenarios

assert sum(len(rows) for rows, _ in sensitivity_counts.values()) == 32

def fmt(value, digits=4):
    return f"{float(value):.{digits}f}"

lines = [
    "# Step 7 — Scheduler Experimental Results",
    "",
    "**Status:** Measured characterization, conservative feasibility, "
    "energy selection, uncertainty, sensitivity and Pareto analysis.",
    "",
    "> Scope: Reproducible results from existing CSV artifacts. "
    "No new benchmark measurements are performed.",
    "",
    "## 1. Experimental coverage",
    "",
    "| Measurement | Count |",
    "|---|---:|",
    f"| All characterized configurations | {len(registry)} |",
    f"| Scheduler core configurations | {sum(r['scheduler_core'] == '1' for r in registry)} |",
    f"| Canonical service candidates | {len(app)} |",
    f"| Service/profile combinations | {len({(r['service_id'], r['request_profile']) for r in app})} |",
    f"| Candidates with latency evidence | "
    f"{sum(bool(r.get('latency_sec')) for r in app)} |",
    f"| Candidates with memory evidence | "
    f"{sum(r.get('has_memory') == '1' for r in app)} |",
    "",
    "## 2. Quality applicability and feasibility",
    "",
    "| Evidence applicability | Candidates |",
    "|---|---:|",
]

for label in ("SUPPORTED", "CONDITIONAL", "UNVERIFIED"):
    lines.append(f"| {label} | {quality_counts[label]} |")

lines.extend([
    "",
    f"All {len(feas)} baseline candidates are **UNVERIFIED**, "
    "not proven FEASIBLE or INFEASIBLE, because real service QoS "
    "thresholds are not configured.",
    "",
    "SUPPORTED quality evidence means a matching benchmark reference "
    "exists; it does not independently establish a deployment SLA.",
    "",
    "Quality limitations include missing Qwen3 Q4_K_M task-quality "
    "evidence, uncontrolled DistilGPT2 precision, and conditional "
    "cross-device references.",
    "",
    "## 3. Pareto analysis",
    "",
    "| Pareto classification | Candidates |",
    "|---|---:|",
    f"| Observed nondominated | "
    f"{pareto_counts['OBSERVED_NONDOMINATED']} |",
    f"| Observed dominated | "
    f"{pareto_counts['OBSERVED_DOMINATED']} |",
    f"| Missing objective metrics | "
    f"{pareto_counts['NOT_EVALUABLE_MISSING_METRICS']} |",
    "",
    f"Comparisons cover **{len(groups)} groups**, "
    f"including **{len(multi)} multi-candidate groups**. "
    f"Only **{len(repeat_supported)} of these {len(multi)} groups** "
    "have complete three-repeat pairwise energy evidence.",
    "",
    "Pareto comparisons use observed energy, latency and throughput "
    "point estimates within compatible service, workload shape, "
    "device, backend, energy boundary and measurement semantics.",
    "",
    "## 4. Repeat-based energy evidence",
    "",
    f"Energy uncertainty analysis covers **{len(uncertainty)} "
    "configurations**, each with three raw repetitions "
    f"(**{sum(int(r['repeat_count']) for r in uncertainty)} "
    "total runs**), across four comparison groups.",
    "",
    "| Workload | Lower-median alternative | "
    "Median advantage | Observed range relation |",
    "|---|---|---:|---|",
])

for row in pairwise:
    name = (
        f"{row['service_id']} / {row['request_profile']} / "
        f"{row['device']} "
        f"(vs {row['higher_median_candidate']})"
    )
    lines.append(
        f"| {name} | {row['lower_median_candidate']} | "
        f"{fmt(row['median_advantage_pct'], 2)}% | "
        f"{row['range_relation']} |"
    )

lines.extend([
    "",
    "OBSERVED_DISJOINT means only that the recorded three-repeat "
    "ranges do not overlap. It is not a confidence interval "
    "or a statistical significance claim.",
    "",
    "## 5. Synthetic QoS sensitivity",
    "",
    "| Analysis | Scenarios | Candidate evaluations |",
    "|---|---:|---:|",
])

for name, (rows, expected) in sensitivity_counts.items():
    lines.append(f"| {name} | {expected} | {len(rows)} |")

lines.extend([
    "| **Total** | **14** | **32** |",
    "",
    "ResNet50 medium / Apple M4 demonstrates the expected "
    "changes in feasibility as latency and throughput thresholds "
    "are tightened. A synthetic ImageNetV2 Top-5 threshold "
    "can select FP32 as the sole eligible candidate even though "
    "FP16 has lower measured energy.",
    "",
    "Intel CPU sensitivity is **partial evidence only**: "
    "missing memory measurements and conditional quality "
    "evidence prevent full feasibility certification.",
    "",
    "## 6. Measurement boundaries and limitations",
    "",
    "- Apple M4 energy: `combined_soc`; RTX 3050 energy: "
    "`gpu_board`; Intel CPU energy: `cpu_package`. "
    "These boundaries must not be treated as interchangeable.",
    "- Memory observations include different measurement "
    "semantics (allocator, process, NVML). An observed "
    "allocated-memory PASS does not prove total device capacity.",
    "- ImageNetV2 Top-1/Top-5 accuracy, WikiText-2 perplexity "
    "and SelfCheck evidence follow distinct evaluation protocols. "
    "Benchmark-reference outcomes are not deployment-quality guarantees.",
    "- Three energy repetitions provide descriptive ranges, "
    "not statistical proof of superiority.",
    "- Synthetic QoS thresholds test decision logic; "
    "they are not validated UAV operating requirements.",
    "- No baseline candidate is certified FEASIBLE and "
    "no cross-device global energy winner is reported.",
    "",
    "## 7. Source artifacts",
    "",
])

for filename in SOURCES.values():
    lines.append(f"- `results/{filename}`")

lines.extend([
    "",
    "Regeneration entrypoint: "
    "`python benchmarks/rebuild_scheduler_artifacts.py`.",
    "",
])

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines), encoding="utf-8")

print("=== SCIENTIFIC SUMMARY ===")
print("Canonical candidates   :", len(app))
print("Pareto groups          :", len(groups))
print("Multi-choice groups    :", len(multi))
print("Energy repeats         :", sum(
    int(r["repeat_count"]) for r in uncertainty
))
print("Sensitivity scenarios  : 14")
print("Candidate evaluations  : 32")
print("Baseline FEASIBLE      : 0")
print("SCIENTIFIC SUMMARY AUDIT: PASS")
print("Saved:", REPORT)
