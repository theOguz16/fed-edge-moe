import csv
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


QOS_MODES = [
    "system",
    "eco",
    "high",
]

REPEATS = 3

WORKER = Path(
    "benchmarks/convnext_large_vision_hard_msi_qos_worker.py"
)

OUT = Path(
    "results/convnext_large_vision_hard_msi_qos_smoke.csv"
)

rows = []

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    for qos in QOS_MODES:
        for repeat in range(1, REPEATS + 1):

            result = (
                tmp /
                f"{qos}_{repeat}.json"
            )

            cmd = [
                sys.executable,
                str(WORKER),

                "--qos",
                qos,

                "--threads",
                "4",

                "--seconds",
                "10",

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
print("-" * 55)

for qos in QOS_MODES:
    group = [
        r for r in rows
        if r["qos"] == qos
    ]

    throughput = statistics.median(
        r["throughput_img_s"]
        for r in group
    )

    latency = statistics.median(
        r["latency_ms_img"]
        for r in group
    )

    print(
        f"{qos:6} | "
        f"{throughput:7.3f} img/s | "
        f"{latency:8.1f} ms/img"
    )

print("\nSaved:", OUT)
