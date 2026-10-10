import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("results")
PROFILES = {
    "light": (160, 1),
    "medium": (224, 4),
    "heavy": (320, 16),
}

def read_csv(filename):
    with (ROOT / filename).open(
        newline="", encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))

groups = defaultdict(list)
metadata = {}

def add(key, repeat, energy, source):
    assert math.isfinite(energy) and energy > 0
    assert repeat in (1, 2, 3)
    assert repeat not in groups[key], (key, repeat)
    groups[key].append(repeat)
    metadata[(key, repeat)] = (energy, source)

resnet_source = "resnet50_vision_easy_mps_power_500ms.csv"

for row in read_csv(resnet_source):
    profile = row["profile"]
    assert profile in PROFILES, profile

    resolution = int(float(row["resolution"]))
    batch = int(float(row["batch"]))

    assert (resolution, batch) == PROFILES[profile]

    precision = row["precision"]
    assert precision in ("fp16", "fp32")

    key = (
        "vision_easy", profile, "Apple M4", "mps",
        "combined_soc", precision, "", resolution, batch
    )

    add(
        key,
        int(row["repeat"]),
        float(row["energy_image_j"]),
        resnet_source,
    )

intel_source = "convnext_base_vision_medium_msi_cpu_power.csv"

for row in read_csv(intel_source):
    resolution = int(float(row["resolution"]))
    batch = int(float(row["batch"]))

    # Exclude noncanonical 320x320 / batch=1 records.
    if (resolution, batch) != (224, 4):
        continue

    threads = int(float(row["threads"]))
    assert threads in (1, 8, 16)

    key = (
        "vision_medium", "medium", "Intel i7-11800H",
        "cpu", "cpu_package", "fp32",
        str(threads), resolution, batch
    )

    add(
        key,
        int(row["repeat"]),
        float(row["energy_img_j"]),
        intel_source,
    )

assert len(groups) == 9, len(groups)

candidates = read_csv(
    "service_candidate_quality_applicability.csv"
)

fields = [
    "service_id", "request_profile", "device", "backend",
    "energy_boundary", "precision", "threads",
    "resolution", "batch", "repeat_count",
    "energy_unit", "median_j_per_image",
    "mean_j_per_image", "sample_sd_j_per_image",
    "minimum_j_per_image", "maximum_j_per_image",
    "cv_pct", "source_csv",
    "uncertainty_interpretation",
]

summary = []

for key, repeats in sorted(groups.items()):
    assert sorted(repeats) == [1, 2, 3], (key, repeats)

    values = [
        metadata[(key, i)][0]
        for i in (1, 2, 3)
    ]

    sources = {
        metadata[(key, i)][1]
        for i in (1, 2, 3)
    }
    assert len(sources) == 1

    (
        service, profile, device, backend, boundary,
        precision, threads, resolution, batch
    ) = key

    median = st.median(values)

    # Audit: raw repeat median must match scheduler registry.
    matched = [
        r for r in candidates
        if r["service_id"] == service
        and r["request_profile"] == profile
        and r["device"] == device
        and r["backend"] == backend
        and r["energy_boundary"] == boundary
        and r["precision"] == precision
        and int(float(r["resolution"])) == resolution
        and int(float(r["batch"])) == batch
        and (
            not threads
            or int(float(r["threads"])) == int(threads)
        )
    ]

    assert len(matched) == 1, (key, len(matched))

    registry_energy = float(
        matched[0]["energy_per_item_j"]
    )

    assert math.isclose(
        median, registry_energy,
        rel_tol=1e-6, abs_tol=1e-9
    ), (key, median, registry_energy)

    record = dict(zip(fields[:9], key))

    record.update({
        "repeat_count": len(values),
        "energy_unit": "J/image",
        "median_j_per_image": median,
        "mean_j_per_image": st.mean(values),
        "sample_sd_j_per_image": st.stdev(values),
        "minimum_j_per_image": min(values),
        "maximum_j_per_image": max(values),
        "cv_pct": 100 * st.stdev(values) / st.mean(values),
        "source_csv": next(iter(sources)),
        "uncertainty_interpretation": (
            "descriptive_only_n3_no_statistical_guarantee"
        ),
    })

    summary.append(record)

output = ROOT / "energy_uncertainty_summary.csv"

with output.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=fields, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(summary)

print("=== ENERGY UNCERTAINTY SUMMARY ===")
print("Configurations:", len(summary))
print("Raw repeats    :", sum(r["repeat_count"] for r in summary))
print("Registry median audit: PASS")

for row in summary:
    print(
        f"{row['service_id']:14s} "
        f"{row['request_profile']:6s} "
        f"{row['device']:16s} "
        f"{row['precision']:4s} "
        f"threads={row['threads'] or '-':>2s} "
        f"median={row['median_j_per_image']:.6f} "
        f"range=[{row['minimum_j_per_image']:.6f}, "
        f"{row['maximum_j_per_image']:.6f}] "
        f"CV={row['cv_pct']:.2f}%"
    )

print("Saved:", output)
print("Inference: descriptive only; n=3 per configuration.")
