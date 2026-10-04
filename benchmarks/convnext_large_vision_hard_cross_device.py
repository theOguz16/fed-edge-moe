import csv
from pathlib import Path

SHORT_OUT = Path("results/convnext_large_vision_hard_cross_device.csv")
SUSTAINED_OUT = Path("results/convnext_large_vision_hard_cross_device_sustained.csv")

mac_short = {
    ("fp32",160,1):38.17, ("fp32",160,4):55.08, ("fp32",160,16):61.40,
    ("fp32",224,1):24.99, ("fp32",224,4):30.07, ("fp32",224,16):30.57,
    ("fp32",320,1):13.94, ("fp32",320,4):15.46, ("fp32",320,16):15.20,

    ("fp16",160,1):46.02, ("fp16",160,4):63.65, ("fp16",160,16):74.64,
    ("fp16",224,1):28.01, ("fp16",224,4):35.79, ("fp16",224,16):37.66,
    ("fp16",320,1):16.04, ("fp16",320,4):18.65, ("fp16",320,16):18.66,
}

msi_short = {
    ("fp32",160,1):48.47, ("fp32",160,4):71.42, ("fp32",160,16):81.91,
    ("fp32",224,1):31.90, ("fp32",224,4):39.25, ("fp32",224,16):40.45,
    ("fp32",320,1):17.53, ("fp32",320,4):20.13, ("fp32",320,16):20.52,

    ("fp16",160,1):98.96, ("fp16",160,4):178.75, ("fp16",160,16):233.47,
    ("fp16",224,1):74.28, ("fp16",224,4):98.71, ("fp16",224,16):125.17,
    ("fp16",320,1):44.47, ("fp16",320,4):52.37, ("fp16",320,16):63.43,
}

rows = []

print("=" * 78)
print("SHORT-SWEEP CROSSOVER")
print("=" * 78)

mac_wins = 0
msi_wins = 0

for key in mac_short:
    precision, resolution, batch = key
    mac = mac_short[key]
    msi = msi_short[key]

    if msi > mac:
        winner = "MSI"
        ratio = msi / mac
        msi_wins += 1
    else:
        winner = "Mac"
        ratio = mac / msi
        mac_wins += 1

    rows.append({
        "precision": precision,
        "resolution": resolution,
        "batch": batch,
        "mac_img_s": mac,
        "msi_img_s": msi,
        "winner": winner,
        "winner_speedup_x": ratio,
    })

    print(
        f"{precision} {resolution}/B{batch:<2} | "
        f"Mac {mac:7.2f} | MSI {msi:7.2f} | "
        f"{winner} {ratio:.2f}x"
    )

with SHORT_OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print()
print(f"SHORT WINS | Mac {mac_wins} | MSI {msi_wins}")


mac_sustained = {
    ("fp32","light"):39.71,
    ("fp32","medium"):28.23,
    ("fp32","heavy"):14.27,
    ("fp16","light"):47.23,
    ("fp16","medium"):35.41,
    ("fp16","heavy"):17.73,
}

msi_sustained = {
    ("fp32","light"):47.63,
    ("fp32","medium"):37.30,
    ("fp32","heavy"):19.66,
    ("fp16","light"):102.79,
    ("fp16","medium"):94.10,
    ("fp16","heavy"):60.91,
}

rows = []

print()
print("=" * 78)
print("SUSTAINED CROSSOVER")
print("=" * 78)

mac_wins = 0
msi_wins = 0

for key in mac_sustained:
    precision, profile = key
    mac = mac_sustained[key]
    msi = msi_sustained[key]

    if msi > mac:
        winner = "MSI"
        ratio = msi / mac
        msi_wins += 1
    else:
        winner = "Mac"
        ratio = mac / msi
        mac_wins += 1

    rows.append({
        "precision": precision,
        "profile": profile,
        "mac_img_s": mac,
        "msi_img_s": msi,
        "winner": winner,
        "winner_speedup_x": ratio,
    })

    print(
        f"{precision} {profile:6} | "
        f"Mac {mac:7.2f} | MSI {msi:7.2f} | "
        f"{winner} {ratio:.2f}x"
    )

with SUSTAINED_OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print()
print(f"SUSTAINED WINS | Mac {mac_wins} | MSI {msi_wins}")

print("\nSaved:", SHORT_OUT)
print("Saved:", SUSTAINED_OUT)
