import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

SRC = Path("results/unified_characterization.csv")

SHORT_OUT = Path("results/crossover_analysis_short.csv")
SUSTAINED_OUT = Path("results/crossover_analysis_sustained.csv")
SUMMARY_OUT = Path("results/crossover_analysis_summary.csv")

MAC = "Apple M4"
MSI = "RTX 3050 Laptop"

with SRC.open(newline="") as f:
    rows = list(csv.DictReader(f))


def fnum(x):
    if x in ("", None):
        return None
    return float(x)


def accelerator_row(r):
    return (
        r["device"] in (MAC, MSI)
        and r["backend"] != "cpu"
    )


def pick_value(a, b):
    """
    Prefer Mac-side metadata when present, otherwise use MSI-side.
    This matters for sustained ConvNeXt-Large, where the Mac
    canonical rows have profile labels but resolution/batch may
    be blank while RTX rows retain them.
    """
    if a not in ("", None):
        return a
    return b


# ------------------------------------------------------------
# Pairing keys
# ------------------------------------------------------------

# Short sweeps compare exact matched configurations.
SHORT_PAIR_KEYS = [
    "modality",
    "difficulty",
    "model",
    "precision",
    "quantization",
    "workload",
    "profile",
    "resolution",
    "context_tokens",
    "output_tokens",
    "batch",
    "measurement_mode",
]

# Sustained measurements are canonical representative profiles.
#
# Do NOT join on resolution/batch/context/output here because
# canonical files are not equally populated on both platforms.
# The profile already represents the logical light/medium/heavy
# sustained workload.
SUSTAINED_PAIR_KEYS = [
    "modality",
    "difficulty",
    "model",
    "precision",
    "quantization",
    "profile",
    "measurement_mode",
]

# Keep a consistent output schema for both CSVs.
OUTPUT_CONFIG_FIELDS = [
    "modality",
    "difficulty",
    "model",
    "precision",
    "quantization",
    "workload",
    "profile",
    "resolution",
    "context_tokens",
    "output_tokens",
    "batch",
    "measurement_mode",
]


def build_pairs(mode, pair_keys):
    candidates = [
        r for r in rows
        if accelerator_row(r)
        and r["measurement_mode"] == mode
        and r["throughput"] not in ("", None)
    ]

    groups = defaultdict(dict)

    for r in candidates:
        if mode == "sustained_power":
            key_parts = []

            for k in pair_keys:
                if k == "profile":
                    # Some sustained canonical sources encode
                    # light/medium/heavy under workload instead
                    # of profile. Use whichever is populated.
                    key_parts.append(
                        r.get("profile")
                        or r.get("workload")
                        or ""
                    )
                else:
                    key_parts.append(r[k])

            key = tuple(key_parts)
        else:
            key = tuple(r[k] for k in pair_keys)

        old = groups[key].get(r["device"])

        if old is None:
            groups[key][r["device"]] = r
        else:
            # Canonical duplicate protection:
            # retain the record backed by more repetitions.
            old_n = int(float(old.get("repeat_count") or 0))
            new_n = int(float(r.get("repeat_count") or 0))

            if new_n > old_n:
                groups[key][r["device"]] = r

    paired = []
    unmatched = []

    for key, devices in groups.items():
        if MAC not in devices or MSI not in devices:
            unmatched.append((key, devices))
            continue

        mac = devices[MAC]
        msi = devices[MSI]

        mac_thr = fnum(mac["throughput"])
        msi_thr = fnum(msi["throughput"])

        if mac_thr is None or msi_thr is None:
            continue

        if mac_thr > msi_thr:
            winner = "Mac"
            winner_speedup = mac_thr / msi_thr
        elif msi_thr > mac_thr:
            winner = "MSI"
            winner_speedup = msi_thr / mac_thr
        else:
            winner = "Tie"
            winner_speedup = 1.0

        result = {
            field: pick_value(mac.get(field), msi.get(field))
            for field in OUTPUT_CONFIG_FIELDS
        }

        result.update({
            "mac_backend": mac["backend"],
            "msi_backend": msi["backend"],
            "mac_throughput": mac_thr,
            "msi_throughput": msi_thr,
            "throughput_unit": pick_value(
                mac.get("throughput_unit"),
                msi.get("throughput_unit"),
            ),
            "winner": winner,
            "winner_speedup_x": winner_speedup,
            "msi_over_mac_x": msi_thr / mac_thr,
            "mac_source": mac["source_file"],
            "msi_source": msi["source_file"],
        })

        paired.append(result)

    return paired, unmatched


short, short_unmatched = build_pairs(
    "short_sweep",
    SHORT_PAIR_KEYS,
)

sustained, sustained_unmatched = build_pairs(
    "sustained_power",
    SUSTAINED_PAIR_KEYS,
)


# ------------------------------------------------------------
# Write detailed paired CSVs
# ------------------------------------------------------------

OUT_FIELDS = (
    OUTPUT_CONFIG_FIELDS
    + [
        "mac_backend",
        "msi_backend",
        "mac_throughput",
        "msi_throughput",
        "throughput_unit",
        "winner",
        "winner_speedup_x",
        "msi_over_mac_x",
        "mac_source",
        "msi_source",
    ]
)


def write_csv(path, data):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(data)


write_csv(SHORT_OUT, short)
write_csv(SUSTAINED_OUT, sustained)


# ------------------------------------------------------------
# Summary by model
# ------------------------------------------------------------

summary = []

for label, data in (
    ("short", short),
    ("sustained", sustained),
):
    models = sorted(set(r["model"] for r in data))

    for model in models:
        xs = [
            r for r in data
            if r["model"] == model
        ]

        wins = Counter(
            r["winner"]
            for r in xs
        )

        ratios = [
            r["msi_over_mac_x"]
            for r in xs
        ]

        summary.append({
            "measurement_group": label,
            "model": model,
            "modality": xs[0]["modality"],
            "difficulty": xs[0]["difficulty"],
            "paired_configurations": len(xs),
            "mac_wins": wins["Mac"],
            "msi_wins": wins["MSI"],
            "ties": wins["Tie"],
            "median_msi_over_mac_x": median(ratios),
            "min_msi_over_mac_x": min(ratios),
            "max_msi_over_mac_x": max(ratios),
        })


SUMMARY_FIELDS = [
    "measurement_group",
    "model",
    "modality",
    "difficulty",
    "paired_configurations",
    "mac_wins",
    "msi_wins",
    "ties",
    "median_msi_over_mac_x",
    "min_msi_over_mac_x",
    "max_msi_over_mac_x",
]

with SUMMARY_OUT.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=SUMMARY_FIELDS,
    )
    w.writeheader()
    w.writerows(summary)


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

def report(label, data):
    print()
    print("=" * 88)
    print(label)
    print("=" * 88)

    by_model = defaultdict(list)

    for r in data:
        by_model[r["model"]].append(r)

    total = Counter()

    for model in sorted(by_model):
        xs = by_model[model]

        wins = Counter(
            r["winner"]
            for r in xs
        )

        ratios = sorted(
            r["msi_over_mac_x"]
            for r in xs
        )

        total.update(wins)

        print(
            f"{model:28} | "
            f"pairs {len(xs):2} | "
            f"Mac {wins['Mac']:2} | "
            f"MSI {wins['MSI']:2} | "
            f"median MSI/Mac {median(ratios):5.2f}x | "
            f"range {min(ratios):.2f}-{max(ratios):.2f}x"
        )

    print("-" * 88)

    print(
        f"TOTAL | pairs {len(data)} | "
        f"Mac {total['Mac']} | "
        f"MSI {total['MSI']} | "
        f"Tie {total['Tie']}"
    )


def unmatched_report(label, unmatched):
    print(f"{label:10}: {len(unmatched)}")

    if not unmatched:
        return

    counts = Counter()

    for _, devices in unmatched:
        present = ",".join(sorted(devices.keys()))
        counts[present] += 1

    for devices, count in sorted(counts.items()):
        print(
            f"             {devices or 'none'}: {count}"
        )


report(
    "SHORT-SWEEP CROSSOVER",
    short,
)

report(
    "SUSTAINED CROSSOVER",
    sustained,
)

print()
print("UNMATCHED")

unmatched_report(
    "short",
    short_unmatched,
)

unmatched_report(
    "sustained",
    sustained_unmatched,
)

print()
print("Saved:", SHORT_OUT)
print("Saved:", SUSTAINED_OUT)
print("Saved:", SUMMARY_OUT)
