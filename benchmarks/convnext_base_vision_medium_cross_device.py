import csv
import statistics
from pathlib import Path

MAC_SWEEP = Path(
    "results/convnext_base_vision_medium_mac_mps_sweep.csv"
)
MSI_SWEEP = Path(
    "results/convnext_base_vision_medium_msi_cuda_sweep.csv"
)

MAC_POWER = Path(
    "results/convnext_base_vision_medium_mac_mps_power.csv"
)
MSI_POWER = Path(
    "results/convnext_base_vision_medium_msi_cuda_power.csv"
)

OUT_SWEEP = Path(
    "results/convnext_base_vision_medium_cross_device.csv"
)
OUT_SUSTAINED = Path(
    "results/convnext_base_vision_medium_cross_device_sustained.csv"
)


def load_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------
# 1) SHORT SWEEP CROSSOVER
# ------------------------------------------------------------

mac_rows = load_csv(MAC_SWEEP)
msi_rows = load_csv(MSI_SWEEP)

mac = {}
msi = {}

for row in mac_rows:
    if row["status"] != "PASS":
        continue

    key = (
        row["precision"],
        int(row["resolution"]),
        int(row["batch"]),
    )

    mac.setdefault(key, []).append(
        float(row["throughput_img_s"])
    )


for row in msi_rows:
    if row["status"] != "PASS":
        continue

    key = (
        row["precision"],
        int(row["resolution"]),
        int(row["batch"]),
    )

    msi.setdefault(key, []).append(
        float(row["throughput_img_s"])
    )


sweep_out = []
mac_wins = 0
msi_wins = 0

print("\nSHORT SWEEP CROSSOVER")
print("-" * 88)

for precision in ["fp32", "fp16"]:
    print(f"\n{precision.upper()}")

    for res in [160, 224, 320]:
        for batch in [1, 4, 16]:

            key = (precision, res, batch)

            mac_thr = statistics.median(mac[key])
            msi_thr = statistics.median(msi[key])

            if mac_thr > msi_thr:
                winner = "mac"
                speedup = mac_thr / msi_thr
                mac_wins += 1
            else:
                winner = "msi"
                speedup = msi_thr / mac_thr
                msi_wins += 1

            sweep_out.append({
                "precision": precision,
                "resolution": res,
                "batch": batch,
                "mac_img_s": mac_thr,
                "msi_img_s": msi_thr,
                "winner": winner,
                "winner_speedup": speedup,
            })

            print(
                f"{res}/B{batch:<2} | "
                f"Mac {mac_thr:7.2f} | "
                f"MSI {msi_thr:7.2f} | "
                f"{winner.upper():3} {speedup:.2f}x"
            )


with OUT_SWEEP.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=sweep_out[0].keys(),
    )
    w.writeheader()
    w.writerows(sweep_out)

print("\nSHORT SWEEP WINS")
print(f"Mac: {mac_wins}")
print(f"MSI: {msi_wins}")


# ------------------------------------------------------------
# 2) SUSTAINED REPRESENTATIVE CROSSOVER
# ------------------------------------------------------------

mac_power = load_csv(MAC_POWER)
msi_power = load_csv(MSI_POWER)


def sustained_medians(rows):
    grouped = {}

    for row in rows:
        key = (
            row["precision"],
            row["profile"],
        )

        grouped.setdefault(key, []).append(
            float(row["no_monitor_img_s"])
        )

    return {
        key: statistics.median(values)
        for key, values in grouped.items()
    }


mac_s = sustained_medians(mac_power)
msi_s = sustained_medians(msi_power)

sustained_out = []
mac_s_wins = 0
msi_s_wins = 0

print("\n\nSUSTAINED REPRESENTATIVE CROSSOVER")
print("-" * 88)

for precision in ["fp32", "fp16"]:

    print(f"\n{precision.upper()}")

    for profile in ["light", "medium", "heavy"]:

        key = (precision, profile)

        mac_thr = mac_s[key]
        msi_thr = msi_s[key]

        if mac_thr > msi_thr:
            winner = "mac"
            speedup = mac_thr / msi_thr
            mac_s_wins += 1
        else:
            winner = "msi"
            speedup = msi_thr / mac_thr
            msi_s_wins += 1

        sustained_out.append({
            "precision": precision,
            "profile": profile,
            "mac_no_monitor_img_s": mac_thr,
            "msi_no_monitor_img_s": msi_thr,
            "winner": winner,
            "winner_speedup": speedup,
        })

        print(
            f"{profile:6} | "
            f"Mac {mac_thr:7.2f} | "
            f"MSI {msi_thr:7.2f} | "
            f"{winner.upper():3} {speedup:.2f}x"
        )


with OUT_SUSTAINED.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=sustained_out[0].keys(),
    )
    w.writeheader()
    w.writerows(sustained_out)

print("\nSUSTAINED WINS")
print(f"Mac: {mac_s_wins}")
print(f"MSI: {msi_s_wins}")

print("\nSaved:")
print(OUT_SWEEP)
print(OUT_SUSTAINED)
