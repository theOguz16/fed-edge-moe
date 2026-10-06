import csv
import importlib.util
import statistics
import subprocess
import tempfile
from pathlib import Path


FORMAL = Path(
    "benchmarks/convnext_large_vision_hard_mac_qos_formal.py"
)

OUT = Path(
    "results/convnext_large_vision_hard_mac_qos_background_confirm.csv"
)

spec = importlib.util.spec_from_file_location(
    "qos_formal",
    FORMAL,
)

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

mod.DURATION_SEC = 20.0

subprocess.run(
    ["sudo", "-n", "-v"],
    check=True,
)

rows = []

with tempfile.TemporaryDirectory() as tmp:
    temp_dir = Path(tmp)

    for repeat in range(1, 6):
        print(f"\n=== background confirmation R{repeat} ===")

        row = mod.run_one(
            "background",
            repeat,
            temp_dir,
        )

        rows.append(row)

        print(
            f"{row['throughput_img_s']:.3f} img/s | "
            f"{row['mean_power_w']:.2f} W | "
            f"{row['energy_img_j']:.2f} J/img | "
            f"P-core share "
            f"{row['process_pcpu_pct_median']:.1f}% | "
            f"E-active "
            f"{row['e_cluster_active_pct_median']:.1f}% | "
            f"P-active "
            f"{row['p_cluster_active_pct_median']:.1f}%"
        )

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)

print("\nCONFIRMATION MEDIANS")

for key in [
    "throughput_img_s",
    "mean_power_w",
    "energy_img_j",
    "process_pcpu_pct_median",
    "qos_bg_pct",
    "qos_default_pct",
    "e_cluster_active_pct_median",
    "p_cluster_active_pct_median",
    "e_cluster_freq_mhz_median",
    "p_cluster_freq_mhz_median",
]:
    values = [
        float(r[key])
        for r in rows
        if r[key] is not None
    ]

    print(
        f"{key:34} "
        f"{statistics.median(values):10.2f}"
    )

print("\nSaved:", OUT)
