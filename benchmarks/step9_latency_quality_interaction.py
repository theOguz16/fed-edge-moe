import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path("results")

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

candidates = read_csv(
    ROOT / "service_candidate_quality_applicability.csv"
)
evidence = read_csv(ROOT / "scheduler_quality_evidence.csv")

pair = {
    r["precision"]: r
    for r in candidates
    if r["service_id"] == "vision_easy"
    and r["request_profile"] == "medium"
    and r["device"] == "Apple M4"
    and r["backend"] == "mps"
}
assert len(pair) == 2
assert set(pair) == {"fp16", "fp32"}

refs = [
    r for r in evidence
    if r["service_id"] == "vision_easy"
    and r["device"] == "Apple M4"
    and r["backend"] == "mps"
    and r["metric"] == "top5_accuracy_pct"
    and r["dataset"] == "imagenetv2_matched_frequency"
    and r["precision"] in pair
]

assert len(refs) == 2
reference = {r["precision"]: r for r in refs}
assert set(reference) == set(pair)

preprocessing = {r["preprocessing"] for r in refs}
assert len(preprocessing) == 1

latency = {
    p: float(r["latency_sec"])
    for p, r in pair.items()
}
quality = {
    p: float(reference[p]["value"])
    for p in pair
}
energy = {
    p: float(r["energy_per_item_j"])
    for p, r in pair.items()
}

assert latency["fp16"] < latency["fp32"]
assert quality["fp16"] < quality["fp32"]
assert energy["fp16"] < energy["fp32"]
assert all(
    r["energy_boundary"] == "combined_soc"
    for r in pair.values()
)

req_config = json.loads(
    Path("configs/service_requirements.json")
    .read_text(encoding="utf-8")
)
mem_config = json.loads(
    Path("configs/memory_metric_contracts.json")
    .read_text(encoding="utf-8")
)
contracts = mem_config["contracts"]

memory_key = "vision_easy|Apple M4|mps"
assert contracts[memory_key]["metric_field"] == "memory_allocated_mb"
assert contracts[memory_key]["unit"] == "MiB"

# In-memory synthetic budget; the real config file is unchanged.
contracts[memory_key]["budget_mib"] = 256

source = Path("benchmarks/evaluate_service_feasibility.py")
tree = ast.parse(source.read_text(encoding="utf-8"))

names = {
    "measured", "threshold", "throughput_gate",
    "latency_gate", "memory_contract", "memory_gate",
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

def five_thresholds(low, high):
    assert 0 < low < high
    delta = (high - low) / 4
    return [
        low - delta,
        low,
        (low + high) / 2,
        high,
        high + delta,
    ]

latency_limits = five_thresholds(
    latency["fp16"], latency["fp32"]
)
quality_limits = five_thresholds(
    quality["fp16"], quality["fp32"]
)

labels = (
    "below_lower",
    "at_lower",
    "between",
    "at_upper",
    "above_upper",
)

base = dict(
    req_config["services"]["vision_easy"]["profiles"]["medium"]
)

base.update({
    "min_throughput": 100,
    "max_quality": None,
    "quality_metric": "top5_accuracy_pct",
    "quality_scope": "benchmark_reference",
    "quality_dataset": "imagenetv2_matched_frequency",
    "quality_preprocessing": next(iter(preprocessing)),
})

output = []
matrix = []

for li, deadline in enumerate(latency_limits):
    line = []
    for qi, minimum_quality in enumerate(quality_limits):
        request = dict(base)
        request["max_latency_s"] = deadline
        request["min_quality"] = minimum_quality

        states = {}
        gates_by_precision = {}
        feasible = []

        for precision, row in pair.items():
            gates = {
                "throughput": env["throughput_gate"](row, request),
                "latency": env["latency_gate"](row, request),
                "memory": env["memory_gate"](row),
                "quality": env["quality_gate"](row, request),
            }

            assert gates["throughput"] == "PASS"
            assert gates["memory"] == "PASS"

            # Independent numeric oracle for the two varied gates.
            assert gates["latency"] == (
                "PASS" if latency[precision] <= deadline
                else "FAIL"
            )
            assert gates["quality"] == (
                "PASS" if quality[precision] >= minimum_quality
                else "FAIL"
            )

            state = env["overall_status"](gates)
            expected = (
                "FEASIBLE"
                if (
                    latency[precision] <= deadline
                    and quality[precision] >= minimum_quality
                )
                else "INFEASIBLE"
            )
            assert state == expected

            states[precision] = state
            gates_by_precision[precision] = gates

            if state == "FEASIBLE":
                feasible.append(precision)

        selected = (
            min(feasible, key=lambda p: energy[p])
            if feasible else ""
        )

        line.append(selected or "ABSTAIN")

        output.append({
            "latency_step": labels[li],
            "quality_step": labels[qi],
            "deadline_s": deadline,
            "minimum_top5_accuracy_pct": minimum_quality,
            "fp16_latency_gate":
                gates_by_precision["fp16"]["latency"],
            "fp16_quality_gate":
                gates_by_precision["fp16"]["quality"],
            "fp16_feasibility": states["fp16"],
            "fp32_latency_gate":
                gates_by_precision["fp32"]["latency"],
            "fp32_quality_gate":
                gates_by_precision["fp32"]["quality"],
            "fp32_feasibility": states["fp32"],
            "feasible_count": len(feasible),
            "selected_precision": selected,
            "selected_energy_j_per_image":
                energy[selected] if selected else "",
            "energy_boundary": "combined_soc",
            "analysis_scope":
                "synthetic_offline_latency_quality_interaction",
        })

    matrix.append(line)

assert len(output) == 25
assert all(len(line) == 5 for line in matrix)

expected_matrix = [
    ["ABSTAIN"] * 5,
    ["fp16", "fp16", "ABSTAIN", "ABSTAIN", "ABSTAIN"],
    ["fp16", "fp16", "ABSTAIN", "ABSTAIN", "ABSTAIN"],
    ["fp16", "fp16", "fp32", "fp32", "ABSTAIN"],
    ["fp16", "fp16", "fp32", "fp32", "ABSTAIN"],
]
assert matrix == expected_matrix, matrix

# Latency relaxation cannot reduce the feasible set.
for qi in range(5):
    counts = [
        output[li * 5 + qi]["feasible_count"]
        for li in range(5)
    ]
    assert counts == sorted(counts)

# Increasing minimum quality cannot enlarge the feasible set.
for li in range(5):
    counts = [
        output[li * 5 + qi]["feasible_count"]
        for qi in range(5)
    ]
    assert counts == sorted(counts, reverse=True)

counts = Counter(
    row["selected_precision"] or "ABSTAIN"
    for row in output
)
assert counts == {
    "ABSTAIN": 13,
    "fp16": 8,
    "fp32": 4,
}, counts

out = ROOT / "step9_latency_quality_interaction.csv"

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(output[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(output)

print("=== LATENCY x QUALITY DECISION MATRIX ===")
print(
    "Rows: latency threshold increasing; "
    "columns: minimum Top-5 quality increasing"
)

for label, line in zip(labels, matrix):
    print(f"{label:14s} | " + " | ".join(
        f"{item:7s}" for item in line
    ))

print("\n=== STEP 9 JOINT QOS SWEEP ===")
print("Synthetic scenarios    :", len(output))
print("Candidate evaluations  :", len(output) * 2)
print("Selections             :", dict(counts))
print("Latency monotonicity   : PASS")
print("Quality monotonicity   : PASS")
print("Real QoS config changed: NO")
print("STEP 9 JOINT QOS SWEEP: PASS")
print("Saved:", out)
