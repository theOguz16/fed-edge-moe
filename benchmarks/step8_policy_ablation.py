import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("results")
GATES = (
    "throughput_gate",
    "latency_gate",
    "memory_gate",
    "quality_gate",
)
VALID = {"PASS", "FAIL", "UNKNOWN", "NOT_CONFIGURED"}

SOURCES = (
    ("resnet_latency",
     "qos_latency_sensitivity_resnet50_m4_medium.csv", 6),
    ("resnet_throughput",
     "qos_throughput_sensitivity_resnet50_m4_medium.csv", 6),
    ("resnet_quality",
     "qos_quality_sensitivity_resnet50_m4_medium.csv", 8),
    ("intel_latency",
     "qos_latency_sensitivity_convnext_cpu_medium_partial.csv", 12),
)

POLICIES = (
    "energy_only",
    "optimistic_qos_first",
    "evidence_aware_qos_first",
)

def load(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

canonical = load("service_candidate_matching.csv")
assert len(canonical) == 57

resnet = [
    r for r in canonical
    if r["service_id"] == "vision_easy"
    and r["request_profile"] == "medium"
    and r["device"] == "Apple M4"
    and r["backend"] == "mps"
]
assert len(resnet) == 2
assert {r["precision"] for r in resnet} == {"fp16", "fp32"}

resnet_energy = {
    r["precision"]: float(r["energy_per_item_j"])
    for r in resnet
}
assert all(v > 0 and math.isfinite(v)
           for v in resnet_energy.values())

groups = defaultdict(list)

for scope, filename, expected_count in SOURCES:
    rows = load(filename)
    assert len(rows) == expected_count, filename

    for r in rows:
        gates = {k: r[k] for k in GATES}
        assert set(gates.values()) <= VALID

        if scope.startswith("resnet_"):
            identity = "precision=" + r["precision"]
            energy = resnet_energy[r["precision"]]
            assert r["energy_boundary"] == "combined_soc"

            if "energy_j_per_image" in r:
                assert math.isclose(
                    energy, float(r["energy_j_per_image"]),
                    rel_tol=1e-10,
                )
            official_status = r["selection_status"]
            official_candidate = r["selected_candidate"]

        else:
            assert scope == "intel_latency"
            identity = "threads=" + str(
                int(float(r["threads"]))
            )
            energy = float(r["energy_j_per_image"])
            assert r["energy_boundary"] == "cpu_package"
            official_status = r["scheduler_selection_status"]
            official_candidate = ""

        assert math.isfinite(energy) and energy > 0

        groups[(scope, r["scenario"])].append({
            "id": identity,
            "energy": energy,
            "boundary": r["energy_boundary"],
            "gates": gates,
            "official_status": official_status,
            "official_candidate": official_candidate,
        })

assert len(groups) == 14

def eligible(item, policy):
    values = item["gates"].values()

    if policy == "energy_only":
        return True
    if policy == "optimistic_qos_first":
        return "FAIL" not in values
    return all(value == "PASS" for value in values)

output = []

for (scope, scenario), candidates in sorted(groups.items()):
    assert len(candidates) == (
        3 if scope == "intel_latency" else 2
    )
    assert len({x["id"] for x in candidates}) == len(candidates)
    assert len({x["boundary"] for x in candidates}) == 1
    assert len({
        x["official_status"] for x in candidates
    }) == 1

    official = candidates[0]["official_status"]
    original_candidate = candidates[0]["official_candidate"]

    for policy in POLICIES:
        pool = [x for x in candidates if eligible(x, policy)]
        pool.sort(key=lambda x: (x["energy"], x["id"]))

        chosen = None
        if pool:
            best = pool[0]
            tied = [
                x for x in pool
                if math.isclose(
                    x["energy"], best["energy"],
                    rel_tol=1e-12, abs_tol=0.0,
                )
            ]
            if len(tied) == 1:
                chosen = best

        if chosen is None:
            decision = (
                "ABSTAIN_TIED_ENERGY"
                if pool else "ABSTAIN_NO_ELIGIBLE"
            )
        else:
            decision = "SELECTED"

        if policy == "evidence_aware_qos_first":
            if official == "NO_FEASIBLE":
                assert chosen is None, (scope, scenario)
            else:
                assert chosen is not None
                assert chosen["id"] in original_candidate, (
                    scope, scenario, chosen["id"]
                )

        selected_gates = (
            chosen["gates"] if chosen else {}
        )

        output.append({
            "scope": scope,
            "scenario": scenario,
            "policy": policy,
            "candidate_count": len(candidates),
            "eligible_count": len(pool),
            "decision": decision,
            "selected_candidate": (
                chosen["id"] if chosen else ""
            ),
            "selected_energy_j_per_item": (
                chosen["energy"] if chosen else ""
            ),
            "known_fail_selected": (
                int("FAIL" in selected_gates.values())
                if chosen else 0
            ),
            "unknown_or_unconfigured_selected": (
                int(any(
                    v in {"UNKNOWN", "NOT_CONFIGURED"}
                    for v in selected_gates.values()
                )) if chosen else 0
            ),
            "energy_boundary": candidates[0]["boundary"],
            "original_scheduler_status": official,
            "analysis_scope": "synthetic_offline_policy_replay",
        })

assert len(output) == 42

selection_counts = Counter(
    r["policy"] for r in output
    if r["decision"] == "SELECTED"
)
fail_counts = Counter(
    r["policy"] for r in output
    if r["known_fail_selected"]
)
unknown_counts = Counter(
    r["policy"] for r in output
    if r["unknown_or_unconfigured_selected"]
)

assert selection_counts == {
    "energy_only": 14,
    "optimistic_qos_first": 10,
    "evidence_aware_qos_first": 7,
}, selection_counts

assert fail_counts == {"energy_only": 6}, fail_counts
assert unknown_counts == {
    "energy_only": 4,
    "optimistic_qos_first": 3,
}, unknown_counts

by_case = {
    (r["scope"], r["scenario"], r["policy"]): r
    for r in output
}
assert by_case[
    ("resnet_quality", "top5_between_candidates",
     "energy_only")
]["selected_candidate"] == "precision=fp16"

assert by_case[
    ("resnet_quality", "top5_between_candidates",
     "evidence_aware_qos_first")
]["selected_candidate"] == "precision=fp32"

path = ROOT / "step8_policy_ablation.csv"
with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(output[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(output)

print("=== STEP 8 POLICY ABLATION ===")
print("Synthetic scenarios:", len(groups))
print("Policy decisions    :", len(output))
print()
for policy in POLICIES:
    print(
        f"{policy:27s} "
        f"selected={selection_counts[policy]:2d}/14 "
        f"known_FAIL={fail_counts[policy]} "
        f"unknown={unknown_counts[policy]}"
    )
print("STEP 8 POLICY ABLATION: PASS")
print("Saved:", path)
