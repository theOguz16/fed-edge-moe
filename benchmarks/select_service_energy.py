import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="results/service_feasibility_baseline.csv",
)
parser.add_argument(
    "--output",
    default="results/service_energy_selection_baseline.csv",
)
args = parser.parse_args()

with Path(args.input).open(
    newline="", encoding="utf-8"
) as f:
    candidates = list(csv.DictReader(f))

assert len(candidates) == 57

KEYS = (
    "service_id",
    "request_profile",
    "device",
    "backend",
    "energy_boundary",
    "throughput_unit",
    "throughput_semantics",
)

SHAPE_KEYS = (
    "resolution",
    "context_tokens",
    "output_tokens",
    "batch",
)

GATES = (
    "throughput_gate",
    "latency_gate",
    "memory_gate",
    "quality_gate",
)

VALID_BOUNDARIES = {
    "combined_soc",
    "cpu_package",
    "gpu_board",
}

def positive_number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None

    if not math.isfinite(number) or number <= 0:
        return None

    return number

def candidate_identity(row):
    columns = (
        "model",
        "precision",
        "quantization",
        "threads",
        "resolution",
        "batch",
        "context_tokens",
        "output_tokens",
    )

    return ";".join(
        f"{column}={row.get(column, '')}"
        for column in columns
    )

groups = defaultdict(list)

for row in candidates:
    key = tuple(row[name] for name in KEYS)
    groups[key].append(row)

results = []

for key, rows in sorted(groups.items()):
    shapes = {
        tuple(row.get(name, "") for name in SHAPE_KEYS)
        for row in rows
    }

    if len(shapes) != 1:
        raise RuntimeError(
            f"Mixed workload shapes in comparison group: {key}"
        )

    feasible = [
        row for row in rows
        if row["feasibility_status"] == "FEASIBLE"
    ]

    for row in feasible:
        if any(row[gate] != "PASS" for gate in GATES):
            raise RuntimeError(
                "FEASIBLE candidate contains non-PASS gate"
            )

    measured = [
        (positive_number(row["energy_per_item_j"]), row)
        for row in feasible
    ]

    invalid_energy = sum(
        value is None for value, _ in measured
    )

    selected = None
    selected_energy = None

    if not feasible:
        status = "NO_FEASIBLE"

    elif key[4] not in VALID_BOUNDARIES:
        status = "ABSTAIN_UNKNOWN_ENERGY_BOUNDARY"

    elif invalid_energy:
        status = "ABSTAIN_MISSING_ENERGY"

    else:
        measured.sort(
            key=lambda item: (
                item[0],
                candidate_identity(item[1]),
            )
        )

        best_energy, best_row = measured[0]

        ties = [
            item for item in measured
            if math.isclose(
                item[0],
                best_energy,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        ]

        if len(ties) > 1:
            status = "ABSTAIN_TIED_POINT_ESTIMATE"
        else:
            selected = candidate_identity(best_row)
            selected_energy = best_energy

            if len(feasible) == 1:
                status = "SOLE_FEASIBLE"
            else:
                status = "PROVISIONAL_MIN_POINT_ESTIMATE"

    result = dict(zip(KEYS, key))

    result.update({
        "total_candidates": len(rows),
        "feasible_candidates": len(feasible),
        "feasible_missing_energy": invalid_energy,
        "selection_status": status,
        "selected_candidate": selected or "",
        "selected_energy_j_per_item": (
            "" if selected_energy is None
            else selected_energy
        ),
        "selection_scope": (
            "same_service_profile_device_backend_"
            "energy_boundary_only"
        ),
        "energy_uncertainty_status": (
            "not_statistically_evaluated"
        ),
    })

    results.append(result)

fields = [
    *KEYS,
    "total_candidates",
    "feasible_candidates",
    "feasible_missing_energy",
    "selection_status",
    "selected_candidate",
    "selected_energy_j_per_item",
    "selection_scope",
    "energy_uncertainty_status",
]

out = Path(args.output)

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(results)

statuses = Counter(
    row["selection_status"] for row in results
)

selected_count = sum(
    bool(row["selected_candidate"])
    for row in results
)

print("=== CONSERVATIVE ENERGY SELECTOR ===")
print("Input candidates :", len(candidates))
print("Comparison groups:", len(results))
print("FEASIBLE inputs  :", sum(
    row["feasibility_status"] == "FEASIBLE"
    for row in candidates
))
print("Selections       :", selected_count)
print("Statuses         :", dict(statuses))
print("Saved            :", out)
print()
print(
    "NOTE: Point-estimate ordering does not establish "
    "statistically reliable energy superiority."
)

if all(
    row["feasibility_status"] == "UNVERIFIED"
    for row in candidates
):
    assert selected_count == 0
    assert statuses == {"NO_FEASIBLE": len(results)}
    print("BASELINE ABSTENTION: PASS")
