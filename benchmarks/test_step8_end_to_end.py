import copy
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REQ = Path("configs/service_requirements.json")
MEM = Path("configs/memory_metric_contracts.json")
CAND = Path("results/service_candidate_quality_applicability.csv")
QUAL = Path("results/scheduler_quality_evidence.csv")
BASE = Path("results/service_feasibility_baseline.csv")

PROTECTED = (REQ, MEM, CAND, QUAL, BASE)

def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

before = {str(p): fingerprint(p) for p in PROTECTED}

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

requirements = json.loads(REQ.read_text(encoding="utf-8"))
contracts = json.loads(MEM.read_text(encoding="utf-8"))
candidates = read_csv(CAND)
quality = read_csv(QUAL)

assert len(candidates) == 57

def is_target(row):
    return (
        row["service_id"] == "vision_easy"
        and row["request_profile"] == "medium"
        and row["device"] == "Apple M4"
        and row["backend"] == "mps"
    )

pair = {r["precision"]: r for r in candidates if is_target(r)}
assert len(pair) == 2
assert set(pair) == {"fp16", "fp32"}

references = [
    r for r in quality
    if r["service_id"] == "vision_easy"
    and r["device"] == "Apple M4"
    and r["backend"] == "mps"
    and r["metric"] == "top5_accuracy_pct"
    and r["dataset"] == "imagenetv2_matched_frequency"
    and r["precision"] in pair
]
assert len(references) == 2

ref_by_precision = {r["precision"]: r for r in references}
assert set(ref_by_precision) == set(pair)

preprocessing = {r["preprocessing"] for r in references}
assert len(preprocessing) == 1

quality_fp16 = float(ref_by_precision["fp16"]["value"])
quality_fp32 = float(ref_by_precision["fp32"]["value"])

assert quality_fp16 < quality_fp32
assert quality_fp32 - quality_fp16 < 0.1

latency = {p: float(r["latency_sec"]) for p, r in pair.items()}
throughput = {p: float(r["throughput"]) for p, r in pair.items()}
memory = {p: float(r["memory_allocated_mb"]) for p, r in pair.items()}
energy = {p: float(r["energy_per_item_j"]) for p, r in pair.items()}

assert latency["fp16"] < latency["fp32"]
assert throughput["fp16"] > throughput["fp32"]
assert memory["fp16"] < memory["fp32"]
assert energy["fp16"] < energy["fp32"]

quality_mid = (quality_fp16 + quality_fp32) / 2
latency_mid = (latency["fp16"] + latency["fp32"]) / 2
throughput_mid = (throughput["fp16"] + throughput["fp32"]) / 2
memory_mid = (memory["fp16"] + memory["fp32"]) / 2

# name, requirement overrides, memory budget,
# expected FP16/FP32 feasibility, selected precision,
# expected selection status
CASES = [
    (
        "both_feasible",
        {},
        256,
        ("FEASIBLE", "FEASIBLE"),
        "fp16",
        "PROVISIONAL_MIN_POINT_ESTIMATE",
    ),
    (
        "quality_fp32_only",
        {"min_quality": quality_mid},
        256,
        ("INFEASIBLE", "FEASIBLE"),
        "fp32",
        "SOLE_FEASIBLE",
    ),
    (
        "latency_fp16_only",
        {"max_latency_s": latency_mid},
        256,
        ("FEASIBLE", "INFEASIBLE"),
        "fp16",
        "SOLE_FEASIBLE",
    ),
    (
        "throughput_fp16_only",
        {"min_throughput": throughput_mid},
        256,
        ("FEASIBLE", "INFEASIBLE"),
        "fp16",
        "SOLE_FEASIBLE",
    ),
    (
        "memory_fp16_only",
        {},
        memory_mid,
        ("FEASIBLE", "INFEASIBLE"),
        "fp16",
        "SOLE_FEASIBLE",
    ),
    (
        "quality_unknown",
        {"quality_scope": "deployment"},
        256,
        ("UNVERIFIED", "UNVERIFIED"),
        None,
        "NO_FEASIBLE",
    ),
    (
        "latency_both_fail",
        {"max_latency_s": latency["fp16"] / 2},
        256,
        ("INFEASIBLE", "INFEASIBLE"),
        None,
        "NO_FEASIBLE",
    ),
]

def run_script(script, arguments):
    process = subprocess.run(
        [sys.executable, script, *arguments],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, (
        script, process.stdout, process.stderr
    )

memory_key = "vision_easy|Apple M4|mps"
assert memory_key in contracts["contracts"]
assert (
    contracts["contracts"][memory_key]["metric_field"]
    == "memory_allocated_mb"
)
assert contracts["contracts"][memory_key]["unit"] == "MiB"

print("=== STEP 8 END-TO-END SCHEDULER TESTS ===")

with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)

    for (
        name, overrides, budget, expected_states,
        selected_precision, expected_selection_status
    ) in CASES:

        req = copy.deepcopy(requirements)
        mem = copy.deepcopy(contracts)

        request = req["services"]["vision_easy"]["profiles"]["medium"]

        request.update({
            "min_throughput": 100,
            "max_latency_s": 0.05,
            "min_quality": min(quality_fp16, quality_fp32) - 0.05,
            "max_quality": None,
            "quality_metric": "top5_accuracy_pct",
            "quality_scope": "benchmark_reference",
            "quality_dataset": "imagenetv2_matched_frequency",
            "quality_preprocessing": next(iter(preprocessing)),
        })
        request.update(overrides)

        mem["contracts"][memory_key]["budget_mib"] = budget

        req_path = temp / f"{name}_requirements.json"
        mem_path = temp / f"{name}_memory.json"
        feasible_path = temp / f"{name}_feasibility.csv"
        selection_path = temp / f"{name}_selection.csv"

        req_path.write_text(
            json.dumps(req, indent=2), encoding="utf-8"
        )
        mem_path.write_text(
            json.dumps(mem, indent=2), encoding="utf-8"
        )

        run_script(
            "benchmarks/evaluate_service_feasibility.py",
            [
                "--requirements", str(req_path),
                "--memory-contracts", str(mem_path),
                "--output", str(feasible_path),
            ],
        )

        feasibility = read_csv(feasible_path)
        assert len(feasibility) == 57

        target = {
            r["precision"]: r
            for r in feasibility if is_target(r)
        }
        assert set(target) == {"fp16", "fp32"}

        observed_states = (
            target["fp16"]["feasibility_status"],
            target["fp32"]["feasibility_status"],
        )

        assert observed_states == expected_states, (
            name, observed_states, expected_states
        )

        # QoS requirements apply across devices for a service/profile.
        # Memory contracts apply across profiles for a service/device/backend.
        # Only candidates outside both changed scopes must remain UNVERIFIED.
        def affected_by_test(row):
            same_request = (
                row["service_id"],
                row["request_profile"],
            ) == ("vision_easy", "medium")

            same_memory_contract = (
                name == "memory_fp16_only"
                and (
                    row["service_id"],
                    row["device"],
                    row["backend"],
                ) == ("vision_easy", "Apple M4", "mps")
            )

            return same_request or same_memory_contract

        assert all(
            r["feasibility_status"] == "UNVERIFIED"
            for r in feasibility
            if not affected_by_test(r)
        ), name

        run_script(
            "benchmarks/select_service_energy.py",
            [
                "--input", str(feasible_path),
                "--output", str(selection_path),
            ],
        )

        selections = read_csv(selection_path)
        assert len(selections) == 37, name

        matched = [
            r for r in selections
            if r["service_id"] == "vision_easy"
            and r["request_profile"] == "medium"
            and r["device"] == "Apple M4"
            and r["backend"] == "mps"
        ]
        assert len(matched) == 1, name
        result = matched[0]

        assert (
            result["selection_status"]
            == expected_selection_status
        ), (name, result)

        # Do not require other devices in vision_easy/medium
        # to remain unchanged: they share the synthetic QoS
        # request. Unrelated service/profiles must not select.
        assert all(
            not r["selected_candidate"]
            for r in selections
            if (
                r["service_id"],
                r["request_profile"],
            ) != ("vision_easy", "medium")
        ), name

        selected = result["selected_candidate"]

        if selected_precision is None:
            assert selected == "", (name, selected)
        else:
            assert (
                f"precision={selected_precision}" in selected
            ), (name, selected)

            observed_energy = float(
                result["selected_energy_j_per_item"]
            )
            assert abs(
                observed_energy - energy[selected_precision]
            ) < 1e-10, name

        print(
            f"{name:24s} PASS "
            f"| FP16={observed_states[0]:11s} "
            f"| FP32={observed_states[1]:11s} "
            f"| selected={selected_precision or 'NONE'}"
        )

for path in PROTECTED:
    assert fingerprint(path) == before[str(path)], path

print()
print("End-to-end scenarios   :", len(CASES))
print("Canonical rows checked : 57 per scenario")
print("Comparison groups      : 37 per scenario")
print("Original files modified: NO")
print("STEP 8 END-TO-END VALIDATION: PASS")
