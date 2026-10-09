import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path("results")
INPUT = ROOT / "service_candidate_quality_applicability.csv"
CONFIG = Path("configs/service_requirements.json")
OUTPUT = ROOT / "service_feasibility_baseline.csv"

config = json.loads(CONFIG.read_text(encoding="utf-8"))
services = config["services"]

with INPUT.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    input_fields = reader.fieldnames or []
    candidates = list(reader)

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

def memory_gate(row):
    budgets = config.get("device_memory_budgets", {})

    if row["device"] not in budgets:
        return "NOT_CONFIGURED"

    # Pending explicit memory metric/boundary contract.
    # A memory observation is not proof of budget compliance.
    return "UNKNOWN"

def quality_gate(row, request):
    minimum = threshold(request.get("min_quality"))

    if minimum is None:
        return "NOT_CONFIGURED"

    # Pending task-specific metric, protocol, and threshold
    # evaluation. Evidence presence is not a quality PASS.
    return "UNKNOWN"

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

    record["feasibility_status"] = overall_status(gates)
    output.append(record)

assert len(output) == 57
assert len(seen_profiles) == 18

fields = [
    *input_fields,
    *(f"{gate}_gate" for gate in GATES),
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

assert counts == {"UNVERIFIED": 57}, (
    f"Unexpected baseline outcomes: {counts}"
)

print("\nSaved:", OUTPUT)
