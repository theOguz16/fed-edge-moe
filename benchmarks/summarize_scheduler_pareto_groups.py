import csv
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path("results")

def read_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

candidates = read_csv("scheduler_pareto_candidates.csv")
pairwise = read_csv("energy_pairwise_repeat_evidence.csv")

GROUP_KEYS = (
    "service_id", "request_profile", "model",
    "device", "backend", "energy_boundary",
    "throughput_unit", "throughput_semantics",
    "latency_semantics", "resolution", "batch",
    "context_tokens", "output_tokens",
)

IDENTITY_KEYS = (
    "model", "precision", "quantization", "threads",
    "resolution", "batch", "context_tokens", "output_tokens",
)

def identity(row):
    return ";".join(
        f"{key}={row.get(key, '')}"
        for key in IDENTITY_KEYS
    )

def repeat_identity(row):
    threads = (row.get("threads") or "").strip()
    threads = str(int(float(threads))) if threads else "-"
    return f"precision={row['precision']};threads={threads}"

def normalized_shape(value):
    if value is None or str(value).strip() == "":
        return ""
    number = float(value)
    assert math.isfinite(number)
    return number

def evidence_key(row):
    return (
        row["service_id"],
        row["request_profile"],
        row["device"],
        row["backend"],
        row["energy_boundary"],
        normalized_shape(row["resolution"]),
        normalized_shape(row["batch"]),
    )

groups = defaultdict(list)
for row in candidates:
    groups[tuple(row[k] for k in GROUP_KEYS)].append(row)

evidence_groups = defaultdict(list)
for row in pairwise:
    evidence_groups[evidence_key(row)].append(row)

assert len(candidates) == 57
assert len(groups) == 37
assert len(pairwise) == 6

summary = []
used_evidence = 0

for group_key, rows in sorted(groups.items()):
    counts = Counter(r["pareto_status"] for r in rows)

    n = len(rows)
    nondominated = counts["OBSERVED_NONDOMINATED"]
    dominated = counts["OBSERVED_DOMINATED"]
    unknown = counts["NOT_EVALUABLE_MISSING_METRICS"]

    assert n == nondominated + dominated + unknown
    assert all(
        int(r["pareto_group_candidates"]) == n
        for r in rows
    )

    assert all(
        r["feasibility_status"] == "UNVERIFIED"
        for r in rows
    )

    matches = evidence_groups.get(evidence_key(rows[0]), [])
    required_pairs = math.comb(n, 2)

    if matches:
        ids = [repeat_identity(r) for r in rows]
        assert len(ids) == len(set(ids)), group_key

        expected_pairs = {
            frozenset((a, b))
            for a, b in combinations(ids, 2)
        }

        actual_pairs = {
            frozenset((
                p["lower_median_candidate"],
                p["higher_median_candidate"],
            ))
            for p in matches
        }

        assert len(actual_pairs) == len(matches)
        assert actual_pairs <= expected_pairs
        assert all(
            int(p["repeats_per_candidate"]) == 3
            for p in matches
        )

    used_evidence += len(matches)

    if n == 1:
        repeat_status = "NOT_APPLICABLE_SINGLETON"
    elif len(matches) == required_pairs:
        repeat_status = "COMPLETE_N3_DESCRIPTIVE"
    elif matches:
        repeat_status = "PARTIAL_N3_DESCRIPTIVE"
    else:
        repeat_status = "NO_REPEAT_PAIRWISE_EVIDENCE"

    record = dict(zip(GROUP_KEYS, group_key))
    record.update({
        "candidate_count": n,
        "observed_nondominated": nondominated,
        "observed_dominated": dominated,
        "not_evaluable": unknown,
        "pareto_metric_coverage": (
            "INCOMPLETE_METRICS"
            if unknown else "COMPLETE_METRICS"
        ),
        "nondominated_candidate_ids": " | ".join(
            identity(r) for r in rows
            if r["pareto_status"] == "OBSERVED_NONDOMINATED"
        ),
        "pairwise_energy_pairs_required": required_pairs,
        "pairwise_energy_pairs_available": len(matches),
        "observed_disjoint_energy_pairs": sum(
            p["range_relation"] == "OBSERVED_DISJOINT"
            for p in matches
        ),
        "observed_overlap_energy_pairs": sum(
            p["range_relation"] == "OBSERVED_OVERLAP"
            for p in matches
        ),
        "repeat_evidence_coverage": repeat_status,
        "qos_feasible_candidates": 0,
        "analysis_scope": (
            "point_estimate_pareto_and_descriptive_energy_repeats"
        ),
        "interpretation": (
            "pareto_not_qos_feasibility_or_statistical_proof"
        ),
    })
    summary.append(record)

assert used_evidence == len(pairwise)
assert len(summary) == 37
assert sum(r["candidate_count"] for r in summary) == 57
assert sum(r["candidate_count"] > 1 for r in summary) == 19
assert sum(r["observed_nondominated"] for r in summary) == 41
assert sum(r["observed_dominated"] for r in summary) == 16
assert sum(r["not_evaluable"] for r in summary) == 0

coverage = Counter(
    r["repeat_evidence_coverage"] for r in summary
)

assert coverage == {
    "NOT_APPLICABLE_SINGLETON": 18,
    "COMPLETE_N3_DESCRIPTIVE": 4,
    "NO_REPEAT_PAIRWISE_EVIDENCE": 15,
}, coverage

output = ROOT / "scheduler_pareto_group_summary.csv"
with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(summary)

print("=== PARETO GROUP SUMMARY ===")
print("Candidates                :", len(candidates))
print("Comparison groups         :", len(summary))
print("Multi-choice groups       :", sum(
    r["candidate_count"] > 1 for r in summary
))
print("Complete repeat groups    :", coverage["COMPLETE_N3_DESCRIPTIVE"])
print("Multi-choice without repeats:",
      coverage["NO_REPEAT_PAIRWISE_EVIDENCE"])
print("Pairwise energy records   :", used_evidence)
print("Pareto nondominated       :", sum(r["observed_nondominated"] for r in summary))
print("Pareto dominated          :", sum(r["observed_dominated"] for r in summary))
print("Missing metrics           :", sum(r["not_evaluable"] for r in summary))
print("QoS FEASIBLE              : 0")

print("\n=== MULTI-CHOICE GROUPS ===")
for row in summary:
    if row["candidate_count"] <= 1:
        continue
    print(
        f"{row['service_id']:14s} "
        f"{row['request_profile']:6s} "
        f"{row['device']:16s} "
        f"N={row['candidate_count']} "
        f"ND={row['observed_nondominated']} "
        f"D={row['observed_dominated']} "
        f"repeat={row['repeat_evidence_coverage']}"
    )

print("\nPARETO GROUP AUDIT: PASS")
print("Saved:", output)
