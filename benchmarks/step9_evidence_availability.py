import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path

RESULTS = Path("results")

def read_csv(name):
    with (RESULTS / name).open(
        newline="", encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))

candidates = read_csv(
    "service_candidate_quality_applicability.csv"
)
evidence = read_csv("scheduler_quality_evidence.csv")

assert len(candidates) == 57

pair = {
    r["precision"]: r
    for r in candidates
    if r["service_id"] == "vision_easy"
    and r["request_profile"] == "medium"
    and r["device"] == "Apple M4"
    and r["backend"] == "mps"
}
assert set(pair) == {"fp16", "fp32"}

matched_refs = [
    r for r in evidence
    if r["service_id"] == "vision_easy"
    and r["device"] == "Apple M4"
    and r["backend"] == "mps"
    and r["metric"] == "top5_accuracy_pct"
    and r["dataset"] == "imagenetv2_matched_frequency"
    and r["precision"] in pair
]

assert len(matched_refs) == 2
refs = {r["precision"]: r for r in matched_refs}
assert set(refs) == set(pair)
assert len({r["preprocessing"] for r in matched_refs}) == 1

energy = {
    p: float(r["energy_per_item_j"])
    for p, r in pair.items()
}

assert energy["fp16"] < energy["fp32"]
assert all(
    r["energy_boundary"] == "combined_soc"
    for r in pair.values()
)

requirements = json.loads(
    Path("configs/service_requirements.json")
    .read_text(encoding="utf-8")
)

contracts = json.loads(
    Path("configs/memory_metric_contracts.json")
    .read_text(encoding="utf-8")
)["contracts"]

memory_key = "vision_easy|Apple M4|mps"

assert contracts[memory_key]["metric_field"] == (
    "memory_allocated_mb"
)
assert contracts[memory_key]["unit"] == "MiB"

# Synthetic memory budget, changed only in memory.
contracts[memory_key]["budget_mib"] = 256

request = dict(
    requirements["services"]["vision_easy"]["profiles"]["medium"]
)

request.update({
    "min_throughput": 100,
    "max_latency_s": 0.05,
    "min_quality": min(
        float(r["value"]) for r in matched_refs
    ) - 0.1,
    "max_quality": None,
    "quality_metric": "top5_accuracy_pct",
    "quality_scope": "benchmark_reference",
    "quality_dataset": "imagenetv2_matched_frequency",
    "quality_preprocessing": matched_refs[0]["preprocessing"],
})

# Load the actual QoS function definitions.
source = Path("benchmarks/evaluate_service_feasibility.py")
tree = ast.parse(source.read_text(encoding="utf-8"))

names = {
    "measured", "threshold",
    "throughput_gate", "latency_gate",
    "memory_contract", "memory_gate",
    "quality_gate", "overall_status",
}

functions = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in names
]

assert {node.name for node in functions} == names

env = {
    "math": math,
    "memory_contracts": contracts,
    "quality_evidence": evidence,
}

exec(
    compile(
        ast.Module(body=functions, type_ignores=[]),
        str(source),
        "exec",
    ),
    env,
)

precisions = ("fp16", "fp32")
metrics = ("throughput", "latency", "memory", "quality")

bits = [
    (precision, metric)
    for precision in precisions
    for metric in metrics
]

assert len(bits) == 8

source_fields = {
    "throughput": "throughput",
    "latency": "latency_sec",
    "memory": "memory_allocated_mb",
}

results = []
states = {}
decision_counts = Counter()
feasibility_counts = Counter()
gate_counts = Counter()

for mask in range(256):
    rows = {
        precision: dict(pair[precision])
        for precision in precisions
    }

    missing_quality_refs = []

    for bit, (precision, metric) in enumerate(bits):
        available = bool(mask & (1 << bit))

        if available:
            continue

        if metric == "quality":
            missing_quality_refs.append(refs[precision])
        else:
            rows[precision][source_fields[metric]] = ""

    # Remove only the selected candidate's matching
    # quality-reference record from the in-memory list.
    env["quality_evidence"] = [
        record for record in evidence
        if all(
            record is not missing
            for missing in missing_quality_refs
        )
    ]

    record = {
        "availability_mask": mask,
        "available_evidence_count": mask.bit_count(),
    }

    eligible = []

    for precision in precisions:
        row = rows[precision]

        gates = {
            "throughput": env["throughput_gate"](
                row, request
            ),
            "latency": env["latency_gate"](
                row, request
            ),
            "memory": env["memory_gate"](row),
            "quality": env["quality_gate"](
                row, request
            ),
        }

        for metric in metrics:
            bit = bits.index((precision, metric))
            available = bool(mask & (1 << bit))
            expected = "PASS" if available else "UNKNOWN"

            assert gates[metric] == expected, (
                mask, precision, metric, gates
            )

            gate_counts[gates[metric]] += 1
            record[f"{precision}_{metric}_gate"] = gates[metric]

        status = env["overall_status"](gates)
        expected_status = (
            "FEASIBLE"
            if all(v == "PASS" for v in gates.values())
            else "UNVERIFIED"
        )

        assert status == expected_status

        record[f"{precision}_feasibility"] = status
        feasibility_counts[status] += 1

        if status == "FEASIBLE":
            eligible.append(precision)

    selected = (
        min(eligible, key=lambda p: energy[p])
        if eligible else ""
    )

    decision_counts[selected or "ABSTAIN"] += 1

    record.update({
        "feasible_count": len(eligible),
        "selected_precision": selected,
        "selected_energy_j_per_image": (
            energy[selected] if selected else ""
        ),
        "energy_boundary": "combined_soc",
        "analysis_scope": (
            "synthetic_offline_evidence_availability_replay"
        ),
    })

    results.append(record)
    states[mask] = (frozenset(eligible), selected)

assert len(results) == 256

assert decision_counts == {
    "fp16": 16,
    "fp32": 15,
    "ABSTAIN": 225,
}, decision_counts

assert feasibility_counts == {
    "FEASIBLE": 32,
    "UNVERIFIED": 480,
}, feasibility_counts

assert gate_counts == {
    "PASS": 1024,
    "UNKNOWN": 1024,
}, gate_counts

# Hypercube monotonicity:
# Adding evidence must never remove a feasible candidate.
edges = 0
transitions = Counter()

for mask, (eligible, selected) in states.items():
    for bit in range(8):
        if mask & (1 << bit):
            continue

        richer_mask = mask | (1 << bit)
        richer_eligible, richer_selected = states[richer_mask]

        assert eligible.issubset(richer_eligible)

        # Once a valid selection exists, additional PASS
        # evidence cannot force abstention.
        if selected:
            assert richer_selected

            # More available evidence may reveal a
            # lower-energy feasible candidate.
            assert energy[richer_selected] <= energy[selected]

        transitions[
            (selected or "ABSTAIN",
             richer_selected or "ABSTAIN")
        ] += 1

        edges += 1

assert edges == 1024

assert transitions[("ABSTAIN", "fp16")] == 60
assert transitions[("ABSTAIN", "fp32")] == 60
assert transitions[("fp32", "fp16")] == 4

assert sum(
    count
    for (before, after), count in transitions.items()
    if before == after
) == 900

OUTPUT = RESULTS / "step9_evidence_availability.csv"

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(results)

print("=== STEP 9 EVIDENCE AVAILABILITY ===")
print("Evidence fields        :", len(bits))
print("Availability scenarios :", len(results))
print("Candidate evaluations  :", len(results) * 2)
print("Gate evaluations       :", sum(gate_counts.values()))
print("Decisions              :", dict(decision_counts))
print("Candidate statuses     :", dict(feasibility_counts))
print()
print("=== EVIDENCE MONOTONICITY ===")
print("Single-field additions :", edges)
print("ABSTAIN -> fp16        :", transitions[("ABSTAIN", "fp16")])
print("ABSTAIN -> fp32        :", transitions[("ABSTAIN", "fp32")])
print("fp32 -> fp16           :", transitions[("fp32", "fp16")])
print("Unchanged decisions    :", 900)
print("Monotonicity           : PASS")
print()
print("Source measurements changed: NO")
print("Original QoS config changed: NO")
print("STEP 9 EVIDENCE AVAILABILITY: PASS")
print("Saved:", OUTPUT)
