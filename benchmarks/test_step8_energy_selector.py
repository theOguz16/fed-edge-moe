import copy
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

INPUT = Path("results/service_feasibility_baseline.csv")
SELECTOR = Path("benchmarks/select_service_energy.py")

GATES = (
    "throughput_gate",
    "latency_gate",
    "memory_gate",
    "quality_gate",
)

KEYS = (
    "service_id",
    "request_profile",
    "device",
    "backend",
    "energy_boundary",
    "throughput_unit",
    "throughput_semantics",
)

with INPUT.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    FIELDS = reader.fieldnames
    BASE = list(reader)

assert FIELDS is not None
assert len(BASE) == 57
assert all(r["feasibility_status"] == "UNVERIFIED" for r in BASE)

def find_pair(rows):
    pair = {
        r["precision"]: r
        for r in rows
        if r["service_id"] == "vision_easy"
        and r["request_profile"] == "medium"
        and r["device"] == "Apple M4"
        and r["backend"] == "mps"
    }
    assert set(pair) == {"fp16", "fp32"}
    assert len({
        tuple(r[k] for k in KEYS)
        for r in pair.values()
    }) == 1
    return pair

def mark_feasible(row):
    row["feasibility_status"] = "FEASIBLE"
    for gate in GATES:
        row[gate] = "PASS"

def prepare(pair, name):
    if name == "baseline":
        return

    if name == "sole_fp32":
        mark_feasible(pair["fp32"])
        pair["fp16"]["energy_per_item_j"] = "0.0001"
        return

    for row in pair.values():
        mark_feasible(row)

    if name == "fp32_cheaper":
        pair["fp16"]["energy_per_item_j"] = "0.20"
        pair["fp32"]["energy_per_item_j"] = "0.10"

    elif name == "fp16_cheaper":
        pair["fp16"]["energy_per_item_j"] = "0.05"
        pair["fp32"]["energy_per_item_j"] = "0.10"

    elif name == "exact_tie":
        pair["fp16"]["energy_per_item_j"] = "0.10"
        pair["fp32"]["energy_per_item_j"] = "0.10"

    elif name == "near_tie":
        pair["fp16"]["energy_per_item_j"] = "0.10"
        pair["fp32"]["energy_per_item_j"] = "0.10000000000005"

    elif name == "missing_energy":
        pair["fp32"]["energy_per_item_j"] = ""

    elif name == "nan_energy":
        pair["fp32"]["energy_per_item_j"] = "nan"

    elif name == "zero_energy":
        pair["fp32"]["energy_per_item_j"] = "0"

    elif name == "unknown_boundary":
        for row in pair.values():
            row["energy_boundary"] = "unknown_boundary"

    elif name == "mixed_shape":
        pair["fp32"]["batch"] = "999"

    elif name == "contradictory_feasible":
        pair["fp16"]["quality_gate"] = "UNKNOWN"

    else:
        raise ValueError(name)

TESTS = [
    ("baseline", "NO_FEASIBLE", None, None),
    ("sole_fp32", "SOLE_FEASIBLE", "fp32", None),
    (
        "fp32_cheaper",
        "PROVISIONAL_MIN_POINT_ESTIMATE",
        "fp32",
        None,
    ),
    (
        "fp16_cheaper",
        "PROVISIONAL_MIN_POINT_ESTIMATE",
        "fp16",
        None,
    ),
    ("exact_tie", "ABSTAIN_TIED_POINT_ESTIMATE", None, None),
    ("near_tie", "ABSTAIN_TIED_POINT_ESTIMATE", None, None),
    ("missing_energy", "ABSTAIN_MISSING_ENERGY", None, None),
    ("nan_energy", "ABSTAIN_MISSING_ENERGY", None, None),
    ("zero_energy", "ABSTAIN_MISSING_ENERGY", None, None),
    (
        "unknown_boundary",
        "ABSTAIN_UNKNOWN_ENERGY_BOUNDARY",
        None,
        None,
    ),
    (
        "mixed_shape",
        None,
        None,
        "Mixed workload shapes in comparison group",
    ),
    (
        "contradictory_feasible",
        None,
        None,
        "FEASIBLE candidate contains non-PASS gate",
    ),
]

for name, expected_status, expected_precision, expected_error in TESTS:
    rows = copy.deepcopy(BASE)
    pair = find_pair(rows)
    prepare(pair, name)

    target_key = tuple(pair["fp16"][k] for k in KEYS)

    with tempfile.TemporaryDirectory() as td:
        source = Path(td) / "input.csv"
        destination = Path(td) / "output.csv"

        with source.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        process = subprocess.run(
            [
                sys.executable,
                str(SELECTOR),
                "--input", str(source),
                "--output", str(destination),
            ],
            capture_output=True,
            text=True,
        )

        if expected_error:
            assert process.returncode != 0, name
            assert expected_error in (
                process.stdout + process.stderr
            ), (name, process.stderr)

        else:
            assert process.returncode == 0, (
                name, process.stdout, process.stderr
            )

            with destination.open(
                newline="", encoding="utf-8"
            ) as f:
                output = list(csv.DictReader(f))

            assert len(output) == 37, name

            matching = [
                r for r in output
                if tuple(r[k] for k in KEYS) == target_key
            ]
            assert len(matching) == 1, name
            result = matching[0]

            assert result["selection_status"] == expected_status, (
                name, result
            )

            selected = result["selected_candidate"]

            if expected_precision is None:
                assert selected == "", (name, selected)
            else:
                assert (
                    f"precision={expected_precision}" in selected
                ), (name, selected)

    print(f"{name:25s} PASS")

print()
print("Direct subprocess tests :", len(TESTS))
print("Original CSV files      : UNMODIFIED")
print("STEP 8 DIRECT SELECTOR REGRESSION: PASS")
