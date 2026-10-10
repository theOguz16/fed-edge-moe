import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("results")
OUTPUT = ROOT / "qos_latency_sensitivity_convnext_cpu_medium_partial.csv"

def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

base_req = json.loads(
    Path("configs/service_requirements.json").read_text()
)
base_mem = json.loads(
    Path("configs/memory_metric_contracts.json").read_text()
)

candidates = read_csv(
    ROOT / "service_candidate_quality_applicability.csv"
)
quality = read_csv(ROOT / "scheduler_quality_evidence.csv")
pairwise = read_csv(ROOT / "energy_pairwise_repeat_evidence.csv")

def target(row):
    return (
        row["service_id"] == "vision_medium"
        and row["request_profile"] == "medium"
        and row["device"] == "Intel i7-11800H"
        and row["backend"] == "cpu"
    )

cpu = [r for r in candidates if target(r)]
assert len(cpu) == 3

by_threads = {int(float(r["threads"])): r for r in cpu}
assert set(by_threads) == {1, 8, 16}

for row in cpu:
    assert row["precision"] == "fp32"
    assert float(row["resolution"]) == 224
    assert float(row["batch"]) == 4
    assert row["energy_boundary"] == "cpu_package"
    assert row["quality_evidence_status"] == "CONDITIONAL"
    assert not row.get("windows_peak_working_set_mib_max")

lat = {
    thread: float(row["latency_sec"])
    for thread, row in by_threads.items()
}

assert lat[8] < lat[16] < lat[1]

references = [
    r for r in quality
    if r["service_id"] == "vision_medium"
    and r["metric"] == "top1_accuracy_pct"
    and r["precision"] == "fp32"
    and r["device"] == "RTX 3050 Laptop"
]

assert len(references) == 1
reference = references[0]

min_throughput = (
    min(float(r["throughput"]) for r in cpu) * 0.95
)

# Synthetic thresholds, not deployment SLAs.
scenarios = [
    ("below_all", lat[8] * 0.95, set()),
    (
        "eight_only",
        (lat[8] + lat[16]) / 2,
        {8},
    ),
    (
        "eight_and_sixteen",
        (lat[16] + lat[1]) / 2,
        {8, 16},
    ),
    ("all_three", lat[1] * 1.05, {1, 8, 16}),
]

def repeat_id(row):
    return (
        f"precision={row['precision']};"
        f"threads={int(float(row['threads']))}"
    )

report = []

with tempfile.TemporaryDirectory() as dirname:
    temp = Path(dirname)

    for i, (name, deadline, expected_threads) in enumerate(scenarios):
        requirements = copy.deepcopy(base_req)
        memory = copy.deepcopy(base_mem)

        request = requirements["services"]["vision_medium"][
            "profiles"
        ]["medium"]

        request.update({
            "min_throughput": min_throughput,
            "max_latency_s": deadline,
            "min_quality": float(reference["value"]) - 0.5,
            "max_quality": None,
            "quality_metric": "top1_accuracy_pct",
            "quality_scope": "benchmark_reference",
            "quality_dataset": reference["dataset"],
            "quality_preprocessing": reference["preprocessing"],
        })

        # Intentionally synthetic. The CPU memory observation
        # is missing, so this must produce UNKNOWN, not PASS.
        memory["contracts"][
            "vision_medium|Intel i7-11800H|cpu"
        ]["budget_mib"] = 4096

        req_file = temp / f"req_{i}.json"
        mem_file = temp / f"mem_{i}.json"
        feas_file = temp / f"feas_{i}.csv"
        selection_file = temp / f"selection_{i}.csv"

        req_file.write_text(json.dumps(requirements))
        mem_file.write_text(json.dumps(memory))

        subprocess.run([
            sys.executable,
            "benchmarks/evaluate_service_feasibility.py",
            "--requirements", str(req_file),
            "--memory-contracts", str(mem_file),
            "--output", str(feas_file),
        ], check=True, stdout=subprocess.DEVNULL)

        subprocess.run([
            sys.executable,
            "benchmarks/select_service_energy.py",
            "--input", str(feas_file),
            "--output", str(selection_file),
        ], check=True, stdout=subprocess.DEVNULL)

        evaluated = [r for r in read_csv(feas_file) if target(r)]
        assert len(evaluated) == 3

        partial = [
            r for r in evaluated
            if r["throughput_gate"] == "PASS"
            and r["latency_gate"] == "PASS"
        ]

        actual_threads = {
            int(float(r["threads"])) for r in partial
        }
        assert actual_threads == expected_threads, (
            name, actual_threads
        )

        for row in evaluated:
            assert row["throughput_gate"] == "PASS"
            assert row["memory_gate"] == "UNKNOWN"
            assert row["quality_gate"] == "UNKNOWN"

            thread = int(float(row["threads"]))
            if thread in expected_threads:
                assert row["feasibility_status"] == "UNVERIFIED"
            else:
                assert row["feasibility_status"] == "INFEASIBLE"

        selection = [r for r in read_csv(selection_file) if target(r)]
        assert len(selection) == 1
        assert selection[0]["selection_status"] == "NO_FEASIBLE"
        assert not selection[0]["selected_candidate"]

        leader = min(
            partial,
            key=lambda r: float(r["energy_per_item_j"]),
            default=None,
        )

        relations = []

        if leader is not None:
            for other in partial:
                if other is leader:
                    continue

                matches = [
                    p for p in pairwise
                    if p["service_id"] == "vision_medium"
                    and p["request_profile"] == "medium"
                    and p["device"] == "Intel i7-11800H"
                    and {
                        p["lower_median_candidate"],
                        p["higher_median_candidate"],
                    } == {repeat_id(leader), repeat_id(other)}
                ]

                assert len(matches) == 1
                assert (
                    matches[0]["lower_median_candidate"]
                    == repeat_id(leader)
                )
                relations.append(matches[0]["range_relation"])

        if not leader:
            relation = "NOT_APPLICABLE_NO_PARTIAL_CANDIDATE"
        elif not relations:
            relation = "NOT_APPLICABLE_SOLE_PARTIAL_CANDIDATE"
        elif "OBSERVED_OVERLAP" in relations:
            relation = "OBSERVED_OVERLAP_WITH_SOME"
        else:
            assert all(
                r == "OBSERVED_DISJOINT" for r in relations
            )
            relation = "OBSERVED_DISJOINT_VS_ALL"

        for row in sorted(
            evaluated, key=lambda r: int(float(r["threads"]))
        ):
            thread = int(float(row["threads"]))

            report.append({
                "scenario": name,
                "deadline_ms": deadline * 1000,
                "threads": thread,
                "observed_latency_ms": (
                    float(row["latency_sec"]) * 1000
                ),
                "throughput_img_s": row["throughput"],
                "energy_j_per_image": row["energy_per_item_j"],
                "throughput_gate": row["throughput_gate"],
                "latency_gate": row["latency_gate"],
                "memory_gate": row["memory_gate"],
                "quality_gate": row["quality_gate"],
                "feasibility_status": row["feasibility_status"],
                "scheduler_selection_status": (
                    selection[0]["selection_status"]
                ),
                "partial_latency_eligible": (
                    thread in actual_threads
                ),
                "partial_energy_leader_not_selected": (
                    "" if leader is None
                    else int(float(leader["threads"]))
                ),
                "partial_repeat_relation": relation,
                "energy_boundary": "cpu_package",
                "analysis_scope": "synthetic_partial_evidence_only",
            })

        print(
            f"{name:20s} "
            f"deadline={deadline * 1000:8.2f} ms "
            f"latency_pass={sorted(actual_threads)} "
            f"partial_energy_leader="
            f"{int(float(leader['threads'])) if leader else '-'} "
            f"repeat={relation} "
            f"scheduler=NO_FEASIBLE"
        )

assert len(report) == 12

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(report[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(report)

print()
print("INTEL CPU PARTIAL QOS SENSITIVITY: PASS")
print("Scenarios:", len(scenarios))
print("Candidate evaluations:", len(report))
print("Actual scheduler selections: 0")
print("Real QoS requirements modified: NO")
print("Saved:", OUTPUT)
