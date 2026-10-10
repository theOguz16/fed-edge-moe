import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path("results")

source = ROOT / "energy_uncertainty_summary.csv"

with source.open(newline="", encoding="utf-8") as f:
    evidence = list(csv.DictReader(f))

assert len(evidence) == 9
assert all(int(r["repeat_count"]) == 3 for r in evidence)
assert all(r["energy_unit"] == "J/image" for r in evidence)

GROUP_FIELDS = (
    "service_id",
    "request_profile",
    "device",
    "backend",
    "energy_boundary",
    "resolution",
    "batch",
)

groups = defaultdict(list)

for row in evidence:
    groups[tuple(row[k] for k in GROUP_FIELDS)].append(row)

assert len(groups) == 4, len(groups)

def identity(row):
    return (
        f"precision={row['precision']};"
        f"threads={row['threads'] or '-'}"
    )

def number(row, field):
    return float(row[field])

comparisons = []

for key, candidates in sorted(groups.items()):
    for a, b in combinations(
        sorted(candidates, key=identity), 2
    ):
        a_median = number(a, "median_j_per_image")
        b_median = number(b, "median_j_per_image")

        assert a_median != b_median, (
            "Tied medians need a separate decision rule",
            key,
        )

        winner, loser = (
            (a, b) if a_median < b_median else (b, a)
        )

        winner_median = number(
            winner, "median_j_per_image"
        )
        loser_median = number(
            loser, "median_j_per_image"
        )

        winner_max = number(
            winner, "maximum_j_per_image"
        )
        loser_min = number(
            loser, "minimum_j_per_image"
        )

        status = (
            "OBSERVED_DISJOINT"
            if winner_max < loser_min
            else "OBSERVED_OVERLAP"
        )

        record = dict(zip(GROUP_FIELDS, key))

        record.update({
            "lower_median_candidate": identity(winner),
            "higher_median_candidate": identity(loser),
            "lower_median_j_per_image": winner_median,
            "higher_median_j_per_image": loser_median,
            "median_advantage_pct": (
                100 * (loser_median - winner_median)
                / loser_median
            ),
            "lower_median_observed_max": winner_max,
            "higher_median_observed_min": loser_min,
            "range_relation": status,
            "repeats_per_candidate": 3,
            "interpretation": (
                "descriptive_only_no_significance_claim"
            ),
        })

        comparisons.append(record)

assert len(comparisons) == 6, len(comparisons)

def find(service, profile, device, winner, loser):
    found = [
        r for r in comparisons
        if r["service_id"] == service
        and r["request_profile"] == profile
        and r["device"] == device
        and {
            r["lower_median_candidate"],
            r["higher_median_candidate"],
        } == {winner, loser}
    ]
    assert len(found) == 1
    return found[0]

for profile, winner, expected_relation in (
    ("light", "fp32", "OBSERVED_OVERLAP"),
    ("medium", "fp16", "OBSERVED_DISJOINT"),
    ("heavy", "fp16", "OBSERVED_DISJOINT"),
):
    record = find(
        "vision_easy",
        profile,
        "Apple M4",
        "precision=fp16;threads=-",
        "precision=fp32;threads=-",
    )
    assert record["lower_median_candidate"] == (
        f"precision={winner};threads=-"
    )
    assert record["range_relation"] == expected_relation

intel = find(
    "vision_medium",
    "medium",
    "Intel i7-11800H",
    "precision=fp32;threads=8",
    "precision=fp32;threads=16",
)

assert intel["lower_median_candidate"] == (
    "precision=fp32;threads=16"
)
assert intel["range_relation"] == "OBSERVED_OVERLAP"

output = ROOT / "energy_pairwise_repeat_evidence.csv"

fields = [
    *GROUP_FIELDS,
    "lower_median_candidate",
    "higher_median_candidate",
    "lower_median_j_per_image",
    "higher_median_j_per_image",
    "median_advantage_pct",
    "lower_median_observed_max",
    "higher_median_observed_min",
    "range_relation",
    "repeats_per_candidate",
    "interpretation",
]

with output.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(comparisons)

print("=== ENERGY PAIRWISE EVIDENCE ===")
print("Configurations    :", len(evidence))
print("Comparison groups :", len(groups))
print("Pairwise comparisons:", len(comparisons))

for row in comparisons:
    print(
        f"{row['service_id']}/{row['request_profile']} "
        f"| {row['device']} "
        f"| {row['lower_median_candidate']} "
        f"| advantage={row['median_advantage_pct']:.2f}% "
        f"| {row['range_relation']}"
    )

print("EXPECTED RELATIONS: PASS")
print("Saved:", output)
print("NOTE: Descriptive n=3 only; no statistical guarantees.")
