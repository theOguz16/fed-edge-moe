import csv
import math
from collections import Counter
from pathlib import Path

SOURCE = Path("results/energy_pairwise_repeat_evidence.csv")
OUTPUT = Path("results/step9_energy_ranking_robustness.csv")

with SOURCE.open(newline="", encoding="utf-8") as f:
    pairs = list(csv.DictReader(f))

assert len(pairs) == 6

results = []
group_ids = set()

for row in pairs:
    low = float(row["lower_median_j_per_image"])
    high = float(row["higher_median_j_per_image"])
    low_max = float(row["lower_median_observed_max"])
    high_min = float(row["higher_median_observed_min"])
    advantage = float(row["median_advantage_pct"])

    assert all(
        math.isfinite(v) and v > 0
        for v in (low, high, low_max, high_min)
    )
    assert low < high
    assert low <= low_max
    assert high_min <= high
    assert int(row["repeats_per_candidate"]) == 3

    calculated_advantage = 100 * (high - low) / high
    assert math.isclose(
        calculated_advantage,
        advantage,
        rel_tol=1e-10,
    )

    gap = high_min - low_max
    relation = (
        "OBSERVED_DISJOINT"
        if gap > 0
        else "OBSERVED_OVERLAP"
    )
    assert relation == row["range_relation"]

    group = tuple(row[k] for k in (
        "service_id",
        "request_profile",
        "device",
        "backend",
        "energy_boundary",
    ))
    group_ids.add(group)

    critical = (high - low) / (high + low)

    # Hypothetical independent, adversarial +/- perturbations.
    # These are not inferred measurement confidence intervals.
    def guaranteed_under(fraction):
        return low * (1 + fraction) < high * (1 - fraction)

    result = {
        "service_id": row["service_id"],
        "request_profile": row["request_profile"],
        "device": row["device"],
        "backend": row["backend"],
        "energy_boundary": row["energy_boundary"],
        "lower_median_candidate": row["lower_median_candidate"],
        "higher_median_candidate": row["higher_median_candidate"],
        "lower_median_j_per_image": low,
        "higher_median_j_per_image": high,
        "median_advantage_pct": advantage,
        "observed_range_relation": relation,
        "observed_boundary_gap_j_per_image": gap,
        "repeats_per_candidate": 3,
        "critical_symmetric_perturbation_pct": critical * 100,
        "ranking_guaranteed_pm1pct": guaranteed_under(0.01),
        "ranking_guaranteed_pm3pct": guaranteed_under(0.03),
        "ranking_guaranteed_pm5pct": guaranteed_under(0.05),
        "analysis_scope": (
            "descriptive_observed_ranges_and_"
            "hypothetical_point_estimate_perturbation"
        ),
    }
    results.append(result)

assert len(group_ids) == 4
assert Counter(
    r["observed_range_relation"] for r in results
) == {
    "OBSERVED_DISJOINT": 4,
    "OBSERVED_OVERLAP": 2,
}

assert sum(
    r["ranking_guaranteed_pm1pct"] for r in results
) == 6

assert sum(
    r["ranking_guaranteed_pm3pct"] for r in results
) == 4

assert sum(
    r["ranking_guaranteed_pm5pct"] for r in results
) == 4

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(results)

print("=== STEP 9 ENERGY RANKING ROBUSTNESS ===")
print("Pairwise comparisons:", len(results))
print("Compatible groups    :", len(group_ids))
print()

for r in results:
    print(
        f'{r["service_id"]:14s} '
        f'{r["request_profile"]:6s} '
        f'{r["device"]:16s} | '
        f'advantage={r["median_advantage_pct"]:6.2f}% | '
        f'critical={r["critical_symmetric_perturbation_pct"]:6.2f}% | '
        f'{r["observed_range_relation"]}'
    )

print()
print("Ranking retained under hypothetical bounds:")
for percent in (1, 3, 5):
    key = f"ranking_guaranteed_pm{percent}pct"
    count = sum(r[key] for r in results)
    print(f"  +/-{percent}% : {count}/6 pairs")

print()
print("NOTE: Hypothetical perturbations are NOT")
print("      empirical uncertainty or confidence intervals.")
print("STEP 9 ENERGY ROBUSTNESS AUDIT: PASS")
print("Saved:", OUTPUT)
