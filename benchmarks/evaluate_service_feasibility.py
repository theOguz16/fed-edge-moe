import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path("results")
INPUT = ROOT / "service_candidate_quality_applicability.csv"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--requirements",
    default="configs/service_requirements.json",
)
parser.add_argument(
    "--memory-contracts",
    default="configs/memory_metric_contracts.json",
)
parser.add_argument(
    "--output",
    default="results/service_feasibility_baseline.csv",
)
args = parser.parse_args()
CONFIG = Path(args.requirements)
OUTPUT = Path(args.output)

config = json.loads(CONFIG.read_text(encoding="utf-8"))
services = config["services"]
memory_contracts = json.loads(
    Path(args.memory_contracts).read_text(encoding="utf-8")
)["contracts"]

with INPUT.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    input_fields = reader.fieldnames or []
    candidates = list(reader)

with (
    ROOT / "scheduler_quality_evidence.csv"
).open(newline="", encoding="utf-8") as f:
    quality_evidence = list(csv.DictReader(f))

GATES = ("throughput", "latency", "memory", "quality")

def measured(value):
    if value is None or str(value).strip() == "":
        return None
    x = float(value)
    return x if math.isfinite(x) else None

def threshold(value):
    if value is None:
        return None
    x = measured(value)
    if x is None:
        raise ValueError(f"Invalid configured threshold: {value}")
    return x

def throughput_gate(row, request):
    minimum = threshold(request.get("min_throughput"))

    if minimum is None:
        return "NOT_CONFIGURED"

    expected_unit = request.get("throughput_unit")
    expected_semantics = request.get("throughput_semantics")

    if (
        not expected_unit
        or not expected_semantics
        or expected_unit != row["throughput_unit"]
        or expected_semantics != row["throughput_semantics"]
    ):
        return "UNKNOWN"

    value = measured(row.get("throughput"))
    if value is None:
        return "UNKNOWN"

    return "PASS" if value >= minimum else "FAIL"

def latency_gate(row, request):
    maximum = threshold(request.get("max_latency_s"))

    if maximum is None:
        return "NOT_CONFIGURED"

    expected_semantics = request.get("latency_semantics")
    observed_semantics = row.get("latency_semantics")

    # Numeric latency alone is not enough:
    # batch completion and per-request latency can differ.
    if (
        not expected_semantics
        or not observed_semantics
        or expected_semantics != observed_semantics
    ):
        return "UNKNOWN"

    value = measured(row.get("latency_sec"))
    if value is None:
        return "UNKNOWN"

    return "PASS" if value <= maximum else "FAIL"

def memory_contract(row):
    key = "|".join((
        row["service_id"],
        row["device"],
        row["backend"],
    ))
    if key not in memory_contracts:
        raise ValueError(f"Missing memory contract: {key}")
    return memory_contracts[key]

def memory_gate(row):
    contract = memory_contract(row)

    if contract["unit"] != "MiB":
        raise ValueError("Memory contract unit must be MiB")

    budget = threshold(contract.get("budget_mib"))

    if budget is None:
        return "NOT_CONFIGURED"

    if budget <= 0:
        raise ValueError("Memory budget must be positive")

    observed = measured(row.get(contract["metric_field"]))

    if observed is None or observed < 0:
        return "UNKNOWN"

    return "PASS" if observed <= budget else "FAIL"

def quality_gate(row, request):
    minimum = threshold(request.get("min_quality"))
    maximum = threshold(request.get("max_quality"))

    if minimum is None and maximum is None:
        return "NOT_CONFIGURED"

    if minimum is not None and maximum is not None:
        raise ValueError("Only one quality bound is allowed")

    metric = request.get("quality_metric")

    directions = {
        "top1_accuracy_pct": "higher",
        "top5_accuracy_pct": "higher",
        "perplexity": "lower",
    }

    if metric not in directions:
        return "UNKNOWN"

    direction = directions[metric]

    if direction == "higher" and minimum is None:
        raise ValueError(f"{metric} requires min_quality")

    if direction == "lower" and maximum is None:
        raise ValueError(f"{metric} requires max_quality")

    # Benchmark evidence must be explicitly requested.
    # It does not certify deployment quality.
    if request.get("quality_scope") != "benchmark_reference":
        return "UNKNOWN"

    dataset = request.get("quality_dataset")

    if not dataset:
        return "UNKNOWN"

    # Cross-device evidence and uncontrolled precision
    # cannot prove a quality constraint.
    if row["quality_evidence_status"] != "SUPPORTED":
        return "UNKNOWN"

    matches = [
        evidence for evidence in quality_evidence
        if (
            evidence["service_id"] == row["service_id"]
            and evidence["device"] == row["device"]
            and evidence["backend"] == row["backend"]
            and evidence["precision"] == row["precision"]
            and evidence["quantization"] == row["quantization"]
            and evidence["dataset"] == dataset
            and evidence["metric"] == metric
            and evidence["direction"] == direction
        )
    ]

    if len(matches) != 1:
        return "UNKNOWN"

    evidence = matches[0]

    if row["modality"] == "vision":
        observed_resolution = measured(row.get("resolution"))
        evaluated_resolution = measured(
            evidence.get("evaluation_input_resolution")
        )

        if (
            observed_resolution is None
            or evaluated_resolution is None
            or observed_resolution != evaluated_resolution
        ):
            return "UNKNOWN"

        expected_preprocessing = request.get(
            "quality_preprocessing"
        )

        if (
            not expected_preprocessing
            or expected_preprocessing
            != evidence["preprocessing"]
        ):
            return "UNKNOWN"

    elif row["modality"] == "text":
        expected_seq = request.get("quality_eval_seq_len")
        expected_tokens = request.get("quality_eval_tokens")

        if (
            expected_seq is None
            or expected_tokens is None
            or measured(evidence.get("evaluation_seq_len"))
            != expected_seq
            or measured(evidence.get("evaluation_tokens"))
            != expected_tokens
        ):
            return "UNKNOWN"

    else:
        return "UNKNOWN"

    observed = measured(evidence["value"])

    if observed is None:
        return "UNKNOWN"

    if direction == "higher":
        return "PASS" if observed >= minimum else "FAIL"

    return "PASS" if observed <= maximum else "FAIL"

def overall_status(results):
    statuses = set(results.values())

    if "FAIL" in statuses:
        return "INFEASIBLE"

    if all(status == "PASS" for status in results.values()):
        return "FEASIBLE"

    return "UNVERIFIED"

output = []
seen_profiles = set()

for row in candidates:
    service_id = row["service_id"]
    profile = row["request_profile"]

    request = services[service_id]["profiles"][profile]
    seen_profiles.add((service_id, profile))

    gates = {
        "throughput": throughput_gate(row, request),
        "latency": latency_gate(row, request),
        "memory": memory_gate(row),
        "quality": quality_gate(row, request),
    }

    record = dict(row)

    for gate, status in gates.items():
        record[f"{gate}_gate"] = status

    contract = memory_contract(row)
    observed_field = contract["metric_field"]

    record["memory_metric_used"] = observed_field
    record["memory_observed_mib"] = row.get(observed_field, "")
    record["memory_budget_mib"] = (
        "" if contract["budget_mib"] is None
        else contract["budget_mib"]
    )
    record["feasibility_status"] = overall_status(gates)
    output.append(record)

assert len(output) == 57
assert len(seen_profiles) == 18

fields = [
    *input_fields,
    *(f"{gate}_gate" for gate in GATES),
    "memory_metric_used",
    "memory_observed_mib",
    "memory_budget_mib",
    "feasibility_status",
]

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(output)

counts = Counter(
    row["feasibility_status"] for row in output
)

print("=== FEASIBILITY BASELINE: PASS ===")
print("Service profiles :", len(seen_profiles))
print("Candidates       :", len(output))
print("FEASIBLE         :", counts["FEASIBLE"])
print("INFEASIBLE       :", counts["INFEASIBLE"])
print("UNVERIFIED       :", counts["UNVERIFIED"])

print("\nGATE STATUS COUNTS")
for gate in GATES:
    statuses = Counter(
        row[f"{gate}_gate"] for row in output
    )
    print(f"{gate:11s}: {dict(statuses)}")

print("\nMEASUREMENT COVERAGE")
print(
    "Latency available:",
    sum(measured(r.get("latency_sec")) is not None
        for r in output),
)
print(
    "Memory evidence  :",
    sum(
        str(r["has_any_memory_observation"]).lower()
        in ("true", "1", "yes")
        for r in output
    ),
)

assert sum(counts.values()) == 57

baseline_unconfigured = (
    all(
        item["budget_mib"] is None
        for item in memory_contracts.values()
    )
    and all(
        request.get("min_throughput") is None
        and request.get("max_latency_s") is None
        and request.get("min_quality") is None
        and request.get("max_quality") is None
        for service in services.values()
        for request in service["profiles"].values()
    )
)

if baseline_unconfigured:
    assert counts == {"UNVERIFIED": 57}, counts

print("\nSaved:", OUTPUT)
