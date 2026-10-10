import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

RESULTS = Path("results")
OUTPUT = RESULTS / "qos_latency_sensitivity_resnet50_m4_medium.csv"

def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

requirements_base = json.loads(
    Path("configs/service_requirements.json").read_text(
        encoding="utf-8"
    )
)
memory_base = json.loads(
    Path("configs/memory_metric_contracts.json").read_text(
        encoding="utf-8"
    )
)

candidates = read_csv(
    RESULTS / "service_candidate_quality_applicability.csv"
)
evidence = read_csv(
    RESULTS / "scheduler_quality_evidence.csv"
)

def is_target(row):
    return (
        row["service_id"] == "vision_easy"
        and row["request_profile"] == "medium"
        and row["device"] == "Apple M4"
        and row["backend"] == "mps"
    )

target = [r for r in candidates if is_target(r)]
assert len(target) == 2

by_precision = {r["precision"]: r for r in target}
assert set(by_precision) == {"fp16", "fp32"}

for row in target:
    assert float(row["resolution"]) == 224
    assert float(row["batch"]) == 4
    assert row["energy_boundary"] == "combined_soc"
    assert row["quality_evidence_status"] == "SUPPORTED"

fp16 = by_precision["fp16"]
fp32 = by_precision["fp32"]

fast_latency = float(fp16["latency_sec"])
slow_latency = float(fp32["latency_sec"])

assert fast_latency < slow_latency
assert (
    float(fp16["energy_per_item_j"])
    < float(fp32["energy_per_item_j"])
)

quality = [
    r for r in evidence
    if r["service_id"] == "vision_easy"
    and r["device"] == "Apple M4"
    and r["backend"] == "mps"
    and r["metric"] == "top1_accuracy_pct"
    and r["precision"] in ("fp16", "fp32")
]

assert len(quality) == 2
assert len({r["dataset"] for r in quality}) == 1
assert len({r["preprocessing"] for r in quality}) == 1

min_throughput = (
    min(float(r["throughput"]) for r in target) * 0.95
)
min_quality = min(
    float(r["value"]) for r in quality
) - 0.5
memory_budget = max(
    float(r["memory_allocated_mb"]) for r in target
) + 128

scenarios = [
    (
        "below_both",
        fast_latency * 0.95,
        {"fp16": "INFEASIBLE", "fp32": "INFEASIBLE"},
        "NO_FEASIBLE",
        "NOT_APPLICABLE_NO_SELECTION",
    ),
    (
        "between_candidates",
        (fast_latency + slow_latency) / 2,
        {"fp16": "FEASIBLE", "fp32": "INFEASIBLE"},
        "SOLE_FEASIBLE",
        "NOT_APPLICABLE_SOLE_FEASIBLE",
    ),
    (
        "above_both",
        slow_latency * 1.05,
        {"fp16": "FEASIBLE", "fp32": "FEASIBLE"},
        "PROVISIONAL_MIN_POINT_ESTIMATE",
        "OBSERVED_DISJOINT_VS_ALL",
    ),
]

report = []

with tempfile.TemporaryDirectory() as dirname:
    temp = Path(dirname)

    for index, (
        name, deadline, expected_feasibility,
        expected_selection, expected_evidence
    ) in enumerate(scenarios):

        requirements = copy.deepcopy(requirements_base)
        memory = copy.deepcopy(memory_base)

        request = requirements["services"]["vision_easy"][
            "profiles"
        ]["medium"]

        request.update({
            "min_throughput": min_throughput,
            "max_latency_s": deadline,
            "min_quality": min_quality,
            "max_quality": None,
            "quality_metric": "top1_accuracy_pct",
            "quality_scope": "benchmark_reference",
            "quality_dataset": quality[0]["dataset"],
            "quality_preprocessing": quality[0]["preprocessing"],
        })

        contract = memory["contracts"][
            "vision_easy|Apple M4|mps"
        ]
        assert contract["metric_field"] == "memory_allocated_mb"
        contract["budget_mib"] = memory_budget

        req_file = temp / f"requirements_{index}.json"
        mem_file = temp / f"memory_{index}.json"
        feasibility_file = temp / f"feasibility_{index}.csv"
        selection_file = temp / f"selection_{index}.csv"
        attached_file = temp / f"evidence_{index}.csv"

        req_file.write_text(
            json.dumps(requirements), encoding="utf-8"
        )
        mem_file.write_text(
            json.dumps(memory), encoding="utf-8"
        )

        commands = [
            [
                sys.executable,
                "benchmarks/evaluate_service_feasibility.py",
                "--requirements", str(req_file),
                "--memory-contracts", str(mem_file),
                "--output", str(feasibility_file),
            ],
            [
                sys.executable,
                "benchmarks/select_service_energy.py",
                "--input", str(feasibility_file),
                "--output", str(selection_file),
            ],
            [
                sys.executable,
                "benchmarks/attach_energy_repeat_evidence.py",
                "--feasibility", str(feasibility_file),
                "--selection", str(selection_file),
                "--output", str(attached_file),
            ],
        ]

        for command in commands:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
            )

        evaluated = [
            r for r in read_csv(feasibility_file)
            if is_target(r)
        ]
        assert len(evaluated) == 2

        actual_feasibility = {
            r["precision"]: r["feasibility_status"]
            for r in evaluated
        }
        assert actual_feasibility == expected_feasibility, (
            name, actual_feasibility
        )

        selected = [
            r for r in read_csv(attached_file)
            if is_target(r)
        ]
        assert len(selected) == 1
        selected = selected[0]

        assert selected["selection_status"] == expected_selection
        assert (
            selected["repeat_evidence_status"]
            == expected_evidence
        )

        if expected_selection == "NO_FEASIBLE":
            assert not selected["selected_candidate"]
        else:
            assert (
                "precision=fp16"
                in selected["selected_candidate"]
            )

        for row in sorted(
            evaluated, key=lambda r: r["precision"]
        ):
            expected_latency_gate = (
                "PASS"
                if float(row["latency_sec"]) <= deadline
                else "FAIL"
            )
            assert row["latency_gate"] == expected_latency_gate

            for gate in ("throughput", "memory", "quality"):
                assert row[f"{gate}_gate"] == "PASS", (
                    name, row["precision"], gate
                )

            report.append({
                "scenario": name,
                "deadline_s": deadline,
                "deadline_ms": deadline * 1000,
                "service_id": row["service_id"],
                "request_profile": row["request_profile"],
                "device": row["device"],
                "backend": row["backend"],
                "precision": row["precision"],
                "observed_latency_s": row["latency_sec"],
                "observed_throughput": row["throughput"],
                "energy_j_per_image": row["energy_per_item_j"],
                "throughput_gate": row["throughput_gate"],
                "latency_gate": row["latency_gate"],
                "memory_gate": row["memory_gate"],
                "quality_gate": row["quality_gate"],
                "feasibility_status": row["feasibility_status"],
                "selection_status": selected["selection_status"],
                "selected_candidate": selected["selected_candidate"],
                "repeat_evidence_status": (
                    selected["repeat_evidence_status"]
                ),
                "qos_scope": "synthetic_benchmark_reference",
                "energy_boundary": row["energy_boundary"],
            })

        print(
            f"{name:20s} "
            f"deadline={deadline * 1000:7.3f} ms "
            f"fp16={actual_feasibility['fp16']:10s} "
            f"fp32={actual_feasibility['fp32']:10s} "
            f"selection={expected_selection}"
        )

assert len(report) == 6

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(report[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(report)

print()
print("LATENCY QOS SENSITIVITY: PASS")
print("Scenarios:", len(scenarios))
print("Candidate evaluations:", len(report))
print("Real QoS requirements modified: NO")
print("Saved:", OUTPUT)
