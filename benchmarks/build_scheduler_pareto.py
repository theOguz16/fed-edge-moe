import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path("results/service_feasibility_baseline.csv")
OUTPUT = Path("results/scheduler_pareto_candidates.csv")

with SOURCE.open(newline="", encoding="utf-8") as f:
    candidates = list(csv.DictReader(f))

assert len(candidates) == 57

GROUP_KEYS = (
    "service_id",
    "request_profile",
    "model",
    "device",
    "backend",
    "energy_boundary",
    "throughput_unit",
    "throughput_semantics",
    "latency_semantics",
    "resolution",
    "batch",
    "context_tokens",
    "output_tokens",
)

IDENTITY_KEYS = (
    "model",
    "precision",
    "quantization",
    "threads",
    "resolution",
    "batch",
    "context_tokens",
    "output_tokens",
)

METRICS = (
    "energy_per_item_j",
    "latency_sec",
    "throughput",
)

def number(value):
    try:
        x = float(value)
    except (ValueError, TypeError):
        return None
    return x if math.isfinite(x) and x > 0 else None

def identity(row):
    return ";".join(
        f"{key}={row.get(key, '')}"
        for key in IDENTITY_KEYS
    )

def dominates(a, b):
    # Energy lower, latency lower, throughput higher.
    no_worse = (
        a[0] <= b[0]
        and a[1] <= b[1]
        and a[2] >= b[2]
    )
    strictly_better = (
        a[0] < b[0]
        or a[1] < b[1]
        or a[2] > b[2]
    )
    return no_worse and strictly_better

groups = defaultdict(list)

for row in candidates:
    key = tuple(row[k] for k in GROUP_KEYS)
    groups[key].append(row)

assert len(groups) == 37, len(groups)

output = []
multi_groups = 0

for key, rows in sorted(groups.items()):
    if len(rows) > 1:
        multi_groups += 1

    ids = [identity(row) for row in rows]
    assert len(ids) == len(set(ids)), (
        "Duplicate candidate identity", key
    )

    observed = {}

    for row in rows:
        values = tuple(number(row.get(k)) for k in METRICS)
        if all(v is not None for v in values):
            observed[identity(row)] = values

    for row in rows:
        candidate_id = identity(row)
        result = dict(row)

        result["pareto_group_candidates"] = len(rows)
        result["pareto_group_complete_metrics"] = len(observed)

        missing = [
            name for name in METRICS
            if number(row.get(name)) is None
        ]

        result["pareto_missing_metrics"] = ";".join(missing)

        if missing:
            result["pareto_status"] = (
                "NOT_EVALUABLE_MISSING_METRICS"
            )
            dominators = []
        else:
            values = observed[candidate_id]

            dominators = [
                other_id
                for other_id, other_values in observed.items()
                if other_id != candidate_id
                and dominates(other_values, values)
            ]

            result["pareto_status"] = (
                "OBSERVED_DOMINATED"
                if dominators else "OBSERVED_NONDOMINATED"
            )

        result["pareto_dominator_count"] = len(dominators)
        result["pareto_dominators"] = " | ".join(dominators)

        result["pareto_evidence_scope"] = (
            "measured_point_estimates_only"
        )
        result["pareto_qos_interpretation"] = (
            "pareto_status_does_not_prove_feasibility"
        )

        output.append(result)

assert len(output) == 57

def find(service, profile, device, **filters):
    found = [
        r for r in output
        if r["service_id"] == service
        and r["request_profile"] == profile
        and r["device"] == device
        and all(
            str(r[k]) == str(v)
            for k, v in filters.items()
        )
    ]
    assert len(found) == 1, (service, profile, filters)
    return found[0]

# ResNet50 medium M4: FP16 improves all three objectives.
fp16 = find(
    "vision_easy", "medium", "Apple M4",
    precision="fp16"
)
fp32 = find(
    "vision_easy", "medium", "Apple M4",
    precision="fp32"
)
assert fp16["pareto_status"] == "OBSERVED_NONDOMINATED"
assert fp32["pareto_status"] == "OBSERVED_DOMINATED"

# Intel: one thread is dominated; 8 and 16 trade latency
# against energy/throughput. None is QoS-proven feasible.
cpu = [
    r for r in output
    if r["service_id"] == "vision_medium"
    and r["request_profile"] == "medium"
    and r["device"] == "Intel i7-11800H"
]
assert len(cpu) == 3

cpu_status = {
    int(float(r["threads"])): r["pareto_status"]
    for r in cpu
}

assert cpu_status == {
    1: "OBSERVED_DOMINATED",
    8: "OBSERVED_NONDOMINATED",
    16: "OBSERVED_NONDOMINATED",
}, cpu_status

assert all(
    r["feasibility_status"] == "UNVERIFIED"
    for r in cpu
)

fields = [
    *candidates[0].keys(),
    "pareto_group_candidates",
    "pareto_group_complete_metrics",
    "pareto_missing_metrics",
    "pareto_status",
    "pareto_dominator_count",
    "pareto_dominators",
    "pareto_evidence_scope",
    "pareto_qos_interpretation",
]

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=fields, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(output)

counts = Counter(r["pareto_status"] for r in output)

print("=== SCHEDULER PARETO ANALYSIS ===")
print("Candidates          :", len(output))
print("Comparison groups   :", len(groups))
print("Multi-choice groups :", multi_groups)
print("Pareto statuses     :", dict(counts))
print("Baseline FEASIBLE   :", sum(
    r["feasibility_status"] == "FEASIBLE"
    for r in output
))
print("ResNet50 M4 medium  : FP16 dominates FP32 — PASS")
print("Intel CPU medium    : 1 dominated; 8/16 nondominated — PASS")
print("Saved:", OUTPUT)
print(
    "NOTE: Descriptive point estimates; Pareto status "
    "does not imply QoS feasibility or statistical superiority."
)
