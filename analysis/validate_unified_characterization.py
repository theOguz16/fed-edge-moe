import csv
from collections import Counter
from pathlib import Path

PATH = Path("results/unified_characterization.csv")

with PATH.open(newline="") as f:
    rows = list(csv.DictReader(f))

print("=" * 72)
print("UNIFIED DATASET VALIDATION")
print("=" * 72)

print("Rows:", len(rows))

# ------------------------------------------------------------
# Status
# ------------------------------------------------------------

status = Counter(r["status"] for r in rows)

print("\nSTATUS")
for k, v in sorted(status.items()):
    print(f"{k:20} {v}")

# ------------------------------------------------------------
# Six workload coverage
# ------------------------------------------------------------

print("\nWORKLOAD COVERAGE")

expected = {
    ("text", "easy"),
    ("text", "medium"),
    ("text", "hard"),
    ("vision", "easy"),
    ("vision", "medium"),
    ("vision", "hard"),
}

present = {
    (r["modality"], r["difficulty"])
    for r in rows
}

for key in sorted(expected):
    count = sum(
        1 for r in rows
        if (r["modality"], r["difficulty"]) == key
    )

    print(
        f"{key[0]:6} {key[1]:6} "
        f"{count:4} rows "
        f"{'PASS' if key in present else 'MISSING'}"
    )

# ------------------------------------------------------------
# Model/device/backend coverage
# ------------------------------------------------------------

print("\nMODELS")
for k, v in sorted(Counter(r["model"] for r in rows).items()):
    print(f"{k:30} {v}")

print("\nDEVICES")
for k, v in sorted(Counter(r["device"] for r in rows).items()):
    print(f"{k:30} {v}")

print("\nBACKENDS")
for k, v in sorted(Counter(r["backend"] for r in rows).items()):
    print(f"{k:30} {v}")

print("\nPRECISION")
for k, v in sorted(Counter(r["precision"] for r in rows).items()):
    print(f"{k:30} {v}")

# ------------------------------------------------------------
# Metric availability
# ------------------------------------------------------------

fields = [
    "throughput",
    "latency_sec",
    "power_w",
    "energy_per_item_j",
    "memory_allocated_mb",
    "memory_peak_mb",
]

print("\nMETRIC COVERAGE")

for field in fields:
    n = sum(
        1 for r in rows
        if r[field] not in ("", None)
    )

    pct = 100 * n / len(rows)

    print(
        f"{field:24} "
        f"{n:4}/{len(rows)} "
        f"({pct:5.1f}%)"
    )

# ------------------------------------------------------------
# Energy boundaries
# ------------------------------------------------------------

print("\nENERGY BOUNDARIES")

power_rows = [
    r for r in rows
    if r["measurement_mode"] == "sustained_power"
]

for k, v in sorted(
    Counter(r["energy_boundary"] for r in power_rows).items()
):
    print(f"{k or 'MISSING':30} {v}")

# ------------------------------------------------------------
# Quality
# ------------------------------------------------------------

quality_rows = sum(
    1 for r in rows
    if r["quality_measured"] == "1"
)

print("\nQUALITY")
print("Measured rows:", quality_rows)
print("Not yet measured:", len(rows) - quality_rows)

# ------------------------------------------------------------
# Duplicate canonical configurations
# ------------------------------------------------------------

identity_fields = [
    "modality",
    "difficulty",
    "model",
    "device",
    "backend",
    "precision",
    "quantization",
    "workload",
    "profile",
    "resolution",
    "context_tokens",
    "output_tokens",
    "batch",
    "threads",
    "measurement_mode",
    "source_file",
]

keys = [
    tuple(r[x] for x in identity_fields)
    for r in rows
]

duplicates = [
    k for k, v in Counter(keys).items()
    if v > 1
]

print("\nDUPLICATES")
print("Duplicate canonical keys:", len(duplicates))

# ------------------------------------------------------------
# Basic numerical sanity
# ------------------------------------------------------------

def positive(field):
    bad = []

    for i, r in enumerate(rows, start=2):
        x = r[field]

        if x in ("", None):
            continue

        try:
            if float(x) <= 0:
                bad.append((i, x))
        except ValueError:
            bad.append((i, x))

    return bad


print("\nNUMERICAL SANITY")

for field in (
    "throughput",
    "power_w",
    "energy_per_item_j",
):
    bad = positive(field)

    print(
        f"{field:24} "
        f"{'PASS' if not bad else 'FAIL'}"
        f" ({len(bad)} bad)"
    )

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

critical_ok = (
    len(rows) > 0
    and expected.issubset(present)
    and not duplicates
    and all(r["status"] == "PASS" for r in rows)
)

print()
print("=" * 72)
print(
    "VALIDATION:",
    "PASS" if critical_ok else "REVIEW REQUIRED"
)
print("=" * 72)
