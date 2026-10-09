import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("results")

def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

base_requirements = json.loads(
    Path("configs/service_requirements.json").read_text(
        encoding="utf-8"
    )
)
base_memory = json.loads(
    Path("configs/memory_metric_contracts.json").read_text(
        encoding="utf-8"
    )
)

candidates = load_csv(
    ROOT / "service_candidate_quality_applicability.csv"
)
evidence = load_csv(
    ROOT / "scheduler_quality_evidence.csv"
)

target = [
    row for row in candidates
    if row["service_id"] == "vision_easy"
    and row["request_profile"] == "medium"
    and row["device"] == "Apple M4"
    and row["backend"] == "mps"
]

assert len(target) == 2
by_precision = {row["precision"]: row for row in target}
assert set(by_precision) == {"fp16", "fp32"}

assert all(
    float(row["resolution"]) == 224
    and row["quality_evidence_status"] == "SUPPORTED"
    and row["energy_boundary"] == "combined_soc"
    for row in target
)

fp16 = by_precision["fp16"]
fp32 = by_precision["fp32"]

assert (
    float(fp16["energy_per_item_j"])
    < float(fp32["energy_per_item_j"])
)
assert (
    float(fp16["latency_sec"])
    < float(fp32["latency_sec"])
)

quality = [
    row for row in evidence
    if row["service_id"] == "vision_easy"
    and row["device"] == "Apple M4"
    and row["backend"] == "mps"
    and row["metric"] == "top1_accuracy_pct"
    and row["precision"] in ("fp16", "fp32")
]

assert len(quality) == 2
assert len({r["dataset"] for r in quality}) == 1
assert len({r["preprocessing"] for r in quality}) == 1

# Synthetic thresholds derived from existing observations.
# These are not real application SLAs.
minimum_throughput = min(
    float(r["throughput"]) for r in target
) * 0.95

relaxed_deadline = max(
    float(r["latency_sec"]) for r in target
) * 1.05

tight_deadline = (
    float(fp16["latency_sec"])
    + float(fp32["latency_sec"])
) / 2

minimum_quality = min(
    float(r["value"]) for r in quality
) - 0.5

memory_budget = max(
    float(r["memory_allocated_mb"]) for r in target
) + 128

cases = [
    (
        "BOTH_FEASIBLE_ENERGY_CHOICE",
        relaxed_deadline,
        "benchmark_reference",
        {"fp16": "FEASIBLE", "fp32": "FEASIBLE"},
        "PROVISIONAL_MIN_POINT_ESTIMATE",
    ),
    (
        "TIGHT_DEADLINE_EXCLUDES_FP32",
        tight_deadline,
        "benchmark_reference",
        {"fp16": "FEASIBLE", "fp32": "INFEASIBLE"},
        "SOLE_FEASIBLE",
    ),
    (
        "DEPLOYMENT_QUALITY_ABSTAINS",
        relaxed_deadline,
        "deployment_quality",
        {"fp16": "UNVERIFIED", "fp32": "UNVERIFIED"},
        "NO_FEASIBLE",
    ),
]

with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)

    for i, (
        label,
        deadline,
        quality_scope,
        expected_statuses,
        expected_selection,
    ) in enumerate(cases):

        requirements = copy.deepcopy(base_requirements)
        memory = copy.deepcopy(base_memory)

        request = requirements["services"]["vision_easy"][
            "profiles"
        ]["medium"]

        request.update({
            "min_throughput": minimum_throughput,
            "max_latency_s": deadline,
            "min_quality": minimum_quality,
            "quality_metric": "top1_accuracy_pct",
            "quality_scope": quality_scope,
            "quality_dataset": quality[0]["dataset"],
            "quality_preprocessing": quality[0]["preprocessing"],
        })

        memory["contracts"][
            "vision_easy|Apple M4|mps"
        ]["budget_mib"] = memory_budget

        requirements_file = temp / f"requirements_{i}.json"
        memory_file = temp / f"memory_{i}.json"
        feasibility_file = temp / f"feasibility_{i}.csv"
        selection_file = temp / f"selection_{i}.csv"

        requirements_file.write_text(
            json.dumps(requirements), encoding="utf-8"
        )
        memory_file.write_text(
            json.dumps(memory), encoding="utf-8"
        )

        subprocess.run(
            [
                sys.executable,
                "benchmarks/evaluate_service_feasibility.py",
                "--requirements", str(requirements_file),
                "--memory-contracts", str(memory_file),
                "--output", str(feasibility_file),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        subprocess.run(
            [
                sys.executable,
                "benchmarks/select_service_energy.py",
                "--input", str(feasibility_file),
                "--output", str(selection_file),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        evaluated = [
            r for r in load_csv(feasibility_file)
            if r["service_id"] == "vision_easy"
            and r["request_profile"] == "medium"
            and r["device"] == "Apple M4"
            and r["backend"] == "mps"
        ]

        assert len(evaluated) == 2
        actual_statuses = {
            r["precision"]: r["feasibility_status"]
            for r in evaluated
        }
        assert actual_statuses == expected_statuses, (
            label, actual_statuses
        )

        for row in evaluated:
            if row["feasibility_status"] == "FEASIBLE":
                assert all(
                    row[f"{gate}_gate"] == "PASS"
                    for gate in (
                        "throughput",
                        "latency",
                        "memory",
                        "quality",
                    )
                ), label

        selections = [
            r for r in load_csv(selection_file)
            if r["service_id"] == "vision_easy"
            and r["request_profile"] == "medium"
            and r["device"] == "Apple M4"
            and r["backend"] == "mps"
        ]

        assert len(selections) == 1
        selected = selections[0]

        assert (
            selected["energy_boundary"] == "combined_soc"
        )
        assert (
            selected["selection_status"] == expected_selection
        ), (label, selected["selection_status"])

        if expected_selection == "NO_FEASIBLE":
            assert not selected["selected_candidate"]
        else:
            assert (
                "precision=fp16"
                in selected["selected_candidate"]
            )

        print(f"{label}: PASS")
        print("  feasibility:", actual_statuses)
        print("  selection  :", expected_selection)

print()
print("ENERGY SELECTOR REGRESSION TESTS: PASS")
print("Scenarios completed:", len(cases))
print("Real service requirements changed: NO")
