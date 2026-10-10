import ast
import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path("results")
SOURCE = Path("benchmarks/evaluate_service_feasibility.py")

def read_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

candidates = read_csv(
    "service_candidate_quality_applicability.csv"
)
evidence = read_csv("scheduler_quality_evidence.csv")

assert len(candidates) == 57

pair = {
    r["precision"]: r for r in candidates
    if r["service_id"] == "vision_easy"
    and r["request_profile"] == "medium"
    and r["device"] == "Apple M4"
    and r["backend"] == "mps"
}
assert set(pair) == {"fp16", "fp32"}

references = {
    r["precision"]: r for r in evidence
    if r["service_id"] == "vision_easy"
    and r["device"] == "Apple M4"
    and r["backend"] == "mps"
    and r["metric"] == "top5_accuracy_pct"
    and r["dataset"] == "imagenetv2_matched_frequency"
    and r["precision"] in pair
}
assert set(references) == set(pair)

preprocessing = {
    r["preprocessing"] for r in references.values()
}
assert len(preprocessing) == 1

observed = {
    "latency": {
        p: float(r["latency_sec"]) for p, r in pair.items()
    },
    "throughput": {
        p: float(r["throughput"]) for p, r in pair.items()
    },
    "memory": {
        p: float(r["memory_allocated_mb"])
        for p, r in pair.items()
    },
    "quality": {
        p: float(r["value"]) for p, r in references.items()
    },
}
energy = {
    p: float(r["energy_per_item_j"])
    for p, r in pair.items()
}

assert energy["fp16"] < energy["fp32"]
assert observed["quality"]["fp16"] < observed["quality"]["fp32"]
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
assert contracts[memory_key]["metric_field"] == "memory_allocated_mb"
assert contracts[memory_key]["unit"] == "MiB"

# Use actual QoS function definitions, without running
# their top-level CSV output pipeline.
names = {
    "measured", "threshold", "throughput_gate",
    "latency_gate", "memory_contract", "memory_gate",
    "quality_gate", "overall_status",
}
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
nodes = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in names
]
assert {node.name for node in nodes} == names

environment = {
    "math": math,
    "memory_contracts": contracts,
    "quality_evidence": evidence,
}
module = ast.Module(body=nodes, type_ignores=[])
exec(compile(module, str(SOURCE), "exec"), environment)

def thresholds(values):
    low, high = sorted(values)
    assert 0 < low < high
    delta = (high - low) / 4
    points = [
        low - delta,
        low,
        low + delta,
        (low + high) / 2,
        high - delta,
        high,
        high + delta,
    ]
    assert len(set(points)) == 7
    return points

axes = {
    "latency": ("max_latency_s", "upper"),
    "throughput": ("min_throughput", "lower"),
    "memory": ("budget_mib", "upper"),
    "quality": ("min_quality", "lower"),
}

output = []

for axis, (setting, direction) in axes.items():
    limits = thresholds(list(observed[axis].values()))
    feasible_counts = []

    for step, limit in enumerate(limits):
        request = copy.deepcopy(
            requirements["services"]["vision_easy"]["profiles"]["medium"]
        )

        request.update({
            "min_throughput":
                min(observed["throughput"].values()) - 1,
            "max_latency_s":
                max(observed["latency"].values()) + 0.01,
            "min_quality":
                min(observed["quality"].values()) - 0.1,
            "max_quality": None,
            "quality_metric": "top5_accuracy_pct",
            "quality_scope": "benchmark_reference",
            "quality_dataset": "imagenetv2_matched_frequency",
            "quality_preprocessing": next(iter(preprocessing)),
        })

        contracts[memory_key]["budget_mib"] = (
            max(observed["memory"].values()) + 10
        )

        if axis == "memory":
            contracts[memory_key]["budget_mib"] = limit
        else:
            request[setting] = limit

        statuses = {}
        eligible = []

        for precision, row in pair.items():
            gates = {
                "throughput": environment["throughput_gate"](
                    row, request
                ),
                "latency": environment["latency_gate"](
                    row, request
                ),
                "memory": environment["memory_gate"](row),
                "quality": environment["quality_gate"](
                    row, request
                ),
            }

            value = observed[axis][precision]
            expected_pass = (
                value <= limit
                if direction == "upper"
                else value >= limit
            )

            assert gates[axis] == (
                "PASS" if expected_pass else "FAIL"
            ), (axis, precision, limit, gates)

            assert all(
                gate == "PASS"
                for name, gate in gates.items()
                if name != axis
            ), (axis, precision, gates)

            status = environment["overall_status"](gates)

            assert status == (
                "FEASIBLE" if expected_pass else "INFEASIBLE"
            )
            statuses[precision] = status

            if status == "FEASIBLE":
                eligible.append(precision)

        selected = (
            min(eligible, key=lambda p: energy[p])
            if eligible else ""
        )

        feasible_counts.append(len(eligible))

        output.append({
            "axis": axis,
            "step": step,
            "threshold": limit,
            "threshold_unit": {
                "latency": "s",
                "throughput": "img/s",
                "memory": "MiB",
                "quality": "Top5_accuracy_pct",
            }[axis],
            "fp16_feasibility": statuses["fp16"],
            "fp32_feasibility": statuses["fp32"],
            "feasible_count": len(eligible),
            "selected_precision": selected,
            "selected_energy_j_per_image":
                energy[selected] if selected else "",
            "energy_boundary": "combined_soc",
            "analysis_scope": "synthetic_offline_threshold_sweep",
        })

    if direction == "upper":
        assert feasible_counts == sorted(feasible_counts)
        assert feasible_counts[0] == 0
        assert feasible_counts[-1] == 2
    else:
        assert feasible_counts == sorted(
            feasible_counts, reverse=True
        )
        assert feasible_counts[0] == 2
        assert feasible_counts[-1] == 0

    print(
        f"{axis:11s} | 7 thresholds | "
        f"feasible counts: {feasible_counts}"
    )

assert len(output) == 28

quality_choices = [
    r["selected_precision"]
    for r in output if r["axis"] == "quality"
]
assert quality_choices[0] == "fp16"
assert quality_choices[3] == "fp32"
assert quality_choices[-1] == ""

out = ROOT / "step9_qos_threshold_sweep.csv"

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(output[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(output)

print("\n=== STEP 9 QOS THRESHOLD SWEEP ===")
print("Axes:", len(axes))
print("Evaluations:", len(output))
print(
    "Decisions:",
    dict(Counter(
        r["selected_precision"] or "ABSTAIN"
        for r in output
    )),
)
print("Original measurements: UNMODIFIED")
print("Real deployment QoS: UNMODIFIED")
print("STEP 9 THRESHOLD SWEEP: PASS")
print("Saved:", out)
