import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("results")

def load(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

base_req = json.loads(
    Path("configs/service_requirements.json").read_text(
        encoding="utf-8"
    )
)
base_mem = json.loads(
    Path("configs/memory_metric_contracts.json").read_text(
        encoding="utf-8"
    )
)

candidates = load(
    ROOT / "service_candidate_quality_applicability.csv"
)
evidence = load(ROOT / "scheduler_quality_evidence.csv")

def target(row):
    return (
        row["service_id"] == "vision_easy"
        and row["request_profile"] == "medium"
        and row["device"] == "Apple M4"
        and row["backend"] == "mps"
    )

selected = [r for r in candidates if target(r)]
assert len(selected) == 2

P = {r["precision"]: r for r in selected}
assert set(P) == {"fp16", "fp32"}

for row in selected:
    assert float(row["resolution"]) == 224
    assert float(row["batch"]) == 4
    assert row["energy_boundary"] == "combined_soc"
    assert row["quality_evidence_status"] == "SUPPORTED"

Q = {}

for metric in ("top1_accuracy_pct", "top5_accuracy_pct"):
    records = [
        r for r in evidence
        if r["service_id"] == "vision_easy"
        and r["device"] == "Apple M4"
        and r["backend"] == "mps"
        and r["metric"] == metric
        and r["precision"] in P
    ]

    assert len(records) == 2
    Q[metric] = {r["precision"]: r for r in records}
    assert set(Q[metric]) == {"fp16", "fp32"}
    assert len({r["dataset"] for r in records}) == 1
    assert len({r["preprocessing"] for r in records}) == 1

tp16 = float(P["fp16"]["throughput"])
tp32 = float(P["fp32"]["throughput"])
assert tp16 > tp32

top1 = {
    p: float(Q["top1_accuracy_pct"][p]["value"])
    for p in P
}
top5 = {
    p: float(Q["top5_accuracy_pct"][p]["value"])
    for p in P
}

assert top1["fp16"] > top1["fp32"]
assert top5["fp32"] > top5["fp16"]

low_tp = tp32 * 0.95
middle_tp = (tp16 + tp32) / 2
high_tp = tp16 * 1.05

low_top1 = min(top1.values()) - 0.5
middle_top1 = sum(top1.values()) / 2
high_top1 = max(top1.values()) + 0.1

middle_top5 = sum(top5.values()) / 2

relaxed_latency = max(
    float(r["latency_sec"]) for r in selected
) * 1.05

memory_budget = max(
    float(r["memory_allocated_mb"]) for r in selected
) + 128

# Each tuple:
# dimension, scenario, throughput threshold,
# quality metric, quality threshold, expected PASS precisions.
scenarios = [
    (
        "throughput", "below_both",
        low_tp, "top1_accuracy_pct", low_top1,
        {"fp16", "fp32"},
    ),
    (
        "throughput", "between_candidates",
        middle_tp, "top1_accuracy_pct", low_top1,
        {"fp16"},
    ),
    (
        "throughput", "above_both",
        high_tp, "top1_accuracy_pct", low_top1,
        set(),
    ),
    (
        "quality", "top1_below_both",
        low_tp, "top1_accuracy_pct", low_top1,
        {"fp16", "fp32"},
    ),
    (
        "quality", "top1_between_candidates",
        low_tp, "top1_accuracy_pct", middle_top1,
        {"fp16"},
    ),
    (
        "quality", "top1_above_both",
        low_tp, "top1_accuracy_pct", high_top1,
        set(),
    ),
    (
        "quality", "top5_between_candidates",
        low_tp, "top5_accuracy_pct", middle_top5,
        {"fp32"},
    ),
]

reports = {"throughput": [], "quality": []}

with tempfile.TemporaryDirectory() as directory:
    temp = Path(directory)

    for i, (
        dimension, scenario, min_tp,
        metric, min_q, expected_pass
    ) in enumerate(scenarios):

        req = copy.deepcopy(base_req)
        mem = copy.deepcopy(base_mem)

        reference = Q[metric]["fp16"]

        contract = req["services"]["vision_easy"][
            "profiles"
        ]["medium"]

        contract.update({
            "min_throughput": min_tp,
            "max_latency_s": relaxed_latency,
            "min_quality": min_q,
            "max_quality": None,
            "quality_metric": metric,
            "quality_scope": "benchmark_reference",
            "quality_dataset": reference["dataset"],
            "quality_preprocessing": reference["preprocessing"],
        })

        mem_contract = mem["contracts"][
            "vision_easy|Apple M4|mps"
        ]
        assert mem_contract["metric_field"] == "memory_allocated_mb"
        mem_contract["budget_mib"] = memory_budget

        req_path = temp / f"requirements_{i}.json"
        mem_path = temp / f"memory_{i}.json"
        feas_path = temp / f"feasibility_{i}.csv"
        selection_path = temp / f"selection_{i}.csv"
        attached_path = temp / f"attached_{i}.csv"

        req_path.write_text(json.dumps(req), encoding="utf-8")
        mem_path.write_text(json.dumps(mem), encoding="utf-8")

        commands = [
            [
                sys.executable,
                "benchmarks/evaluate_service_feasibility.py",
                "--requirements", str(req_path),
                "--memory-contracts", str(mem_path),
                "--output", str(feas_path),
            ],
            [
                sys.executable,
                "benchmarks/select_service_energy.py",
                "--input", str(feas_path),
                "--output", str(selection_path),
            ],
            [
                sys.executable,
                "benchmarks/attach_energy_repeat_evidence.py",
                "--feasibility", str(feas_path),
                "--selection", str(selection_path),
                "--output", str(attached_path),
            ],
        ]

        for command in commands:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
            )

        evaluated = [r for r in load(feas_path) if target(r)]
        chosen = [r for r in load(attached_path) if target(r)]

        assert len(evaluated) == 2
        assert len(chosen) == 1
        decision = chosen[0]

        expected_status = {
            0: "NO_FEASIBLE",
            1: "SOLE_FEASIBLE",
            2: "PROVISIONAL_MIN_POINT_ESTIMATE",
        }[len(expected_pass)]

        expected_evidence = {
            0: "NOT_APPLICABLE_NO_SELECTION",
            1: "NOT_APPLICABLE_SOLE_FEASIBLE",
            2: "OBSERVED_DISJOINT_VS_ALL",
        }[len(expected_pass)]

        assert decision["selection_status"] == expected_status
        assert decision["repeat_evidence_status"] == expected_evidence

        expected_winner = (
            "fp16" if len(expected_pass) == 2
            else next(iter(expected_pass)) if expected_pass
            else None
        )

        if expected_winner is None:
            assert not decision["selected_candidate"]
        else:
            assert (
                f"precision={expected_winner}"
                in decision["selected_candidate"]
            )

        actual_pass = set()

        for row in evaluated:
            precision = row["precision"]
            expected_feasible = precision in expected_pass

            assert row["latency_gate"] == "PASS"
            assert row["memory_gate"] == "PASS"

            if dimension == "throughput":
                assert row["quality_gate"] == "PASS"
                checked_gate = "throughput_gate"
            else:
                assert row["throughput_gate"] == "PASS"
                checked_gate = "quality_gate"

            expected_gate = "PASS" if expected_feasible else "FAIL"
            assert row[checked_gate] == expected_gate, (
                scenario, precision, checked_gate,
                row[checked_gate]
            )

            expected_overall = (
                "FEASIBLE" if expected_feasible
                else "INFEASIBLE"
            )
            assert row["feasibility_status"] == expected_overall

            if expected_feasible:
                actual_pass.add(precision)

            reports[dimension].append({
                "scenario": scenario,
                "service_id": row["service_id"],
                "request_profile": row["request_profile"],
                "device": row["device"],
                "precision": precision,
                "min_throughput_img_s": min_tp,
                "observed_throughput_img_s": row["throughput"],
                "quality_metric": metric,
                "min_quality_pct": min_q,
                "observed_quality_pct": Q[metric][precision]["value"],
                "throughput_gate": row["throughput_gate"],
                "quality_gate": row["quality_gate"],
                "latency_gate": row["latency_gate"],
                "memory_gate": row["memory_gate"],
                "feasibility_status": row["feasibility_status"],
                "selection_status": decision["selection_status"],
                "selected_candidate": decision["selected_candidate"],
                "repeat_evidence_status": (
                    decision["repeat_evidence_status"]
                ),
                "energy_boundary": row["energy_boundary"],
                "analysis_scope": "synthetic_benchmark_reference",
            })

        assert actual_pass == expected_pass

        print(
            f"{dimension:10s} {scenario:24s} "
            f"PASS={sorted(actual_pass)} "
            f"selected={expected_winner or '-'} "
            f"status={expected_status}"
        )

outputs = {
    "throughput": (
        ROOT / "qos_throughput_sensitivity_resnet50_m4_medium.csv"
    ),
    "quality": (
        ROOT / "qos_quality_sensitivity_resnet50_m4_medium.csv"
    ),
}

assert len(reports["throughput"]) == 6
assert len(reports["quality"]) == 8

for dimension, output in outputs.items():
    rows = reports[dimension]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output)

print()
print("THROUGHPUT AND QUALITY QOS SENSITIVITY: PASS")
print("Scenarios:", len(scenarios))
print("Candidate evaluations:", sum(map(len, reports.values())))
print("Real service requirements modified: NO")
print(
    "NOTE: Top-1/Top-5 thresholds are synthetic;"
    " small accuracy differences are not significance claims."
)
