import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

BASE = Path("configs/partition_manifests_v2")
OUT = Path("results/benchmark_v2_plans")
OUT.mkdir(parents=True, exist_ok=True)

PROFILES = [
    ("resnet50", "stage6"),
    ("resnet50", "block18"),
    ("convnext_base", "compatible35"),
    ("convnext_base", "fine41"),
    ("convnext_large", "compatible35"),
    ("convnext_large", "fine41"),
]

def make_mapping(n, first, cuts):
    last = 0
    segments = []

    for index, end in enumerate((*cuts, n)):
        device = first if index % 2 == 0 else (
            "M" if first == "C" else "C"
        )
        segments.append(device * (end - last))
        last = end

    return "".join(segments)

def generate_mappings(n):
    for first in ("C", "M"):
        for switches in (0, 1, 2):
            for cuts in itertools.combinations(
                range(1, n), switches
            ):
                yield make_mapping(n, first, cuts), cuts

def save_csv(path, rows, fields):
    temp = path.with_suffix(".tmp")

    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)

    temp.replace(path)

catalog = []
core = []
extended = []

for model, profile in PROFILES:
    manifest_path = BASE / f"{model}__{profile}.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)

    groups = manifest["groups"]
    n = len(groups)

    assert n == manifest["group_count"]

    # Cut after Gi is a stage boundary if stage names differ.
    stage_cuts = {
        i for i in range(1, n)
        if groups[i - 1]["stage"] != groups[i]["stage"]
    }

    count = 0

    for mapping, cuts in generate_mappings(n):
        switches = len(cuts)

        if switches == 0:
            category = (
                "full_cpu" if mapping[0] == "C"
                else "full_gpu"
            )
        elif switches == 1:
            category = "single_cut"
        else:
            category = (
                "cpu_island" if mapping[0] == "M"
                else "gpu_island"
            )

        island_width = (
            cuts[1] - cuts[0]
            if switches == 2 else 0
        )

        adjacent_to_stage = (
            switches == 2
            and (
                cuts[0] in stage_cuts
                or cuts[1] in stage_cuts
            )
        )

        # Tier 1: full baselines, stage cuts,
        # and single-block islands next to stage boundaries.
        is_core = (
            switches == 0
            or (
                switches == 1
                and cuts[0] in stage_cuts
            )
            or (
                switches == 2
                and island_width == 1
                and adjacent_to_stage
            )
        )

        # Tier 2: all single cuts, plus selected
        # 1/2/4-block islands at stage boundaries.
        is_extended = (
            switches <= 1
            or (
                switches == 2
                and island_width in (1, 2, 4)
                and adjacent_to_stage
            )
        )

        row = {
            "model": model,
            "profile": profile,
            "group_count": n,
            "mapping": mapping,
            "category": category,
            "internal_switches": switches,
            "cut_after_groups": ",".join(
                f"G{i}" for i in cuts
            ),
            "island_width": island_width,
            "stage_boundary_cuts": sum(
                i in stage_cuts for i in cuts
            ),
            "manifest_sha256":
                hashlib.sha256(raw).hexdigest(),
            "candidate_id": hashlib.sha256(
                f"{model}|{profile}|{mapping}".encode()
            ).hexdigest()[:16],
        }

        catalog.append(row)
        count += 1

        if is_core:
            core.append(row)

        if is_extended:
            extended.append(row)

    expected = 2 * (
        1 + (n - 1)
        + (n - 1) * (n - 2) // 2
    )
    assert count == expected

    print(
        f"{model:17s} {profile:15s} "
        f"groups={n:2d} "
        f"stage_cuts={len(stage_cuts):2d} "
        f"catalog={count:4d}"
    )

fields = list(catalog[0])

save_csv(OUT / "vision_catalog30.csv", catalog, fields)
save_csv(OUT / "vision_core30.csv", core, fields)
save_csv(OUT / "vision_extended30.csv", extended, fields)

print("\n=== MAC-30 VISION CUT-POINT MATRIX ===")
print("Catalog placements:", len(catalog))
print("Extended placements:", len(extended))
print("Core placements:", len(core))

print("\nCORE BY MODEL / PROFILE")
counts = Counter(
    (r["model"], r["profile"]) for r in core
)
for key, count in sorted(counts.items()):
    print(f"{key[0]:17s} {key[1]:15s} {count:4d}")

print("\nSaved in:", OUT)
print("No inference was executed.")
