import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--feasibility",
    default="results/service_feasibility_baseline.csv",
)
parser.add_argument(
    "--selection",
    default="results/service_energy_selection_baseline.csv",
)
parser.add_argument(
    "--pairwise",
    default="results/energy_pairwise_repeat_evidence.csv",
)
parser.add_argument(
    "--output",
    default="results/service_energy_selection_evidence_baseline.csv",
)
args = parser.parse_args()

def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)

_, candidates = read_csv(args.feasibility)
selection_fields, selections = read_csv(args.selection)
_, pairwise = read_csv(args.pairwise)

GROUP_KEYS = (
    "service_id",
    "request_profile",
    "device",
    "backend",
    "energy_boundary",
    "throughput_unit",
    "throughput_semantics",
)

PAIR_KEYS = GROUP_KEYS[:5]

IDENTITY_FIELDS = (
    "model",
    "precision",
    "quantization",
    "threads",
    "resolution",
    "batch",
    "context_tokens",
    "output_tokens",
)

def group_key(row):
    return tuple(row[k] for k in GROUP_KEYS)

def candidate_id(row):
    return ";".join(
        f"{field}={row.get(field, '')}"
        for field in IDENTITY_FIELDS
    )

def repeat_id(row):
    threads = (row.get("threads") or "").strip()
    threads = str(int(float(threads))) if threads else "-"
    return f"precision={row['precision']};threads={threads}"

def shape(row):
    try:
        resolution = float(row["resolution"])
        batch = float(row["batch"])
        if not all(math.isfinite(v) for v in (resolution, batch)):
            return None
        return resolution, batch
    except (ValueError, TypeError, KeyError):
        return None

def pair_context(row):
    return tuple(row[k] for k in PAIR_KEYS)

groups = defaultdict(list)

for row in candidates:
    groups[group_key(row)].append(row)

assert len(selections) == len(groups)
assert {group_key(r) for r in selections} == set(groups)

def evidence_for(winner, loser):
    winner_shape = shape(winner)
    loser_shape = shape(loser)

    if (
        winner_shape is None
        or winner_shape != loser_shape
        or winner["throughput_unit"] != "img/s"
    ):
        return None

    pair_ids = {repeat_id(winner), repeat_id(loser)}
    if len(pair_ids) != 2:
        return None

    matches = [
        p for p in pairwise
        if pair_context(p) == pair_context(winner)
        and shape(p) == winner_shape
        and {
            p["lower_median_candidate"],
            p["higher_median_candidate"],
        } == pair_ids
    ]

    if len(matches) != 1:
        return None

    record = matches[0]

    if int(record["repeats_per_candidate"]) != 3:
        return None

    if record["lower_median_candidate"] != repeat_id(winner):
        raise AssertionError(
            "Point-estimate winner conflicts with repeat evidence"
        )

    for candidate, field in (
        (winner, "lower_median_j_per_image"),
        (loser, "higher_median_j_per_image"),
    ):
        if not math.isclose(
            float(candidate["energy_per_item_j"]),
            float(record[field]),
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise AssertionError(
                "Candidate energy differs from raw-repeat median"
            )

    return record

annotated = []

for selection in selections:
    matching = groups[group_key(selection)]

    feasible = [
        row for row in matching
        if row["feasibility_status"] == "FEASIBLE"
    ]

    assert len(feasible) == int(
        selection["feasible_candidates"]
    )

    result = dict(selection)
    result.update({
        "repeat_evidence_status": "",
        "repeat_comparisons_supported": 0,
        "repeat_comparisons_required": 0,
        "repeat_evidence_interpretation": (
            "descriptive_only_no_statistical_guarantee"
        ),
    })

    selected_id = selection["selected_candidate"]

    if not selected_id:
        result["repeat_evidence_status"] = (
            "NOT_APPLICABLE_NO_SELECTION"
        )

    elif len(feasible) == 1:
        assert candidate_id(feasible[0]) == selected_id
        result["repeat_evidence_status"] = (
            "NOT_APPLICABLE_SOLE_FEASIBLE"
        )

    else:
        chosen = [
            row for row in feasible
            if candidate_id(row) == selected_id
        ]

        assert len(chosen) == 1
        winner = chosen[0]

        others = [
            row for row in feasible
            if candidate_id(row) != selected_id
        ]

        result["repeat_comparisons_required"] = len(others)

        # Pairwise evidence identifies precision and thread count.
        # If that identity is ambiguous, do not attach evidence.
        identities = [repeat_id(row) for row in matching]

        if len(identities) != len(set(identities)):
            result["repeat_evidence_status"] = (
                "AMBIGUOUS_REPEAT_IDENTITY"
            )

        else:
            records = [
                evidence_for(winner, other)
                for other in others
            ]

            supported = [r for r in records if r is not None]
            result["repeat_comparisons_supported"] = len(supported)

            if len(supported) != len(others):
                result["repeat_evidence_status"] = (
                    "REPEAT_EVIDENCE_INCOMPLETE"
                )

            elif all(
                r["range_relation"] == "OBSERVED_DISJOINT"
                for r in supported
            ):
                result["repeat_evidence_status"] = (
                    "OBSERVED_DISJOINT_VS_ALL"
                )

            elif any(
                r["range_relation"] == "OBSERVED_OVERLAP"
                for r in supported
            ):
                result["repeat_evidence_status"] = (
                    "OBSERVED_OVERLAP_WITH_SOME"
                )

            else:
                raise ValueError(
                    "Unknown range relation in repeat evidence"
                )

    annotated.append(result)

output = Path(args.output)

fields = [
    *selection_fields,
    "repeat_evidence_status",
    "repeat_comparisons_supported",
    "repeat_comparisons_required",
    "repeat_evidence_interpretation",
]

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=fields, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(annotated)

statuses = Counter(
    r["repeat_evidence_status"] for r in annotated
)

print("=== ENERGY REPEAT EVIDENCE ATTACHMENT ===")
print("Selection groups:", len(annotated))
print("Evidence statuses:", dict(statuses))
print("Saved:", output)
print("NOTE: Selection decisions are unchanged.")

if all(
    r["selection_status"] == "NO_FEASIBLE"
    for r in selections
):
    assert statuses == {
        "NOT_APPLICABLE_NO_SELECTION": len(selections)
    }
    print("BASELINE EVIDENCE ABSTENTION: PASS")
