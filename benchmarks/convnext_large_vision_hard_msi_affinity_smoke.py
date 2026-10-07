import csv
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


PLACEMENTS = [
    "unrestricted",
    "four_physical",
    "two_physical_smt",
]

REPEATS = 3
THREADS = 4
DURATION_SEC = 10.0

WORKER = Path(
    "benchmarks/convnext_large_vision_hard_msi_affinity_worker.py"
)

OUT = Path(
    "results/convnext_large_vision_hard_msi_affinity_smoke.csv"
)

rows = []

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    for placement in PLACEMENTS:
        for repeat in range(1, REPEATS + 1):

            result = (
                tmp /
                f"{placement}_{repeat}.json"
            )

            cmd = [
                sys.executable,
                str(WORKER),

                "--placement",
                placement,

                "--threads",
                str(THREADS),

                "--seconds",
                str(DURATION_SEC),

                "--repeat",
                str(repeat),

                "--result",
                str(result),
            ]

            subprocess.run(
                cmd,
                check=True,
            )

            rows.append(
                json.loads(
                    result.read_text()
                )
            )


with OUT.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


print("\nMEDIANS")
print("-" * 70)

for placement in PLACEMENTS:
    group = [
        r for r in rows
        if r["placement"] == placement
    ]

    thr = statistics.median(
        r["throughput_img_s"]
        for r in group
    )

    lat = statistics.median(
        r["latency_ms_img"]
        for r in group
    )

    print(
        f"{placement:17} | "
        f"{thr:7.3f} img/s | "
        f"{lat:8.1f} ms/img"
    )

print("\nSaved:", OUT)
