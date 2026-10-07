import csv
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path


PLACEMENTS = [
    "unrestricted",
    "four_physical",
    "two_physical_smt",
]

REPEATS = 3
THREADS = 4
DURATION_SEC = 20.0
POLL_SEC = 1.0

WORKER = Path(
    "benchmarks/convnext_large_vision_hard_msi_affinity_power_worker.py"
)

OUT = Path(
    "results/convnext_large_vision_hard_msi_affinity_power.csv"
)

LHM_URL = "http://localhost:8085/data.json"
CPU_POWER_ID = "/intelcpu/0/power/0"


def get_json():
    with urllib.request.urlopen(
        LHM_URL,
        timeout=3,
    ) as r:
        return json.load(r)


def find_sensor(node, sensor_id):
    if isinstance(node, dict):
        if node.get("SensorId") == sensor_id:
            return node

        for value in node.values():
            found = find_sensor(
                value,
                sensor_id,
            )

            if found is not None:
                return found

    elif isinstance(node, list):
        for value in node:
            found = find_sensor(
                value,
                sensor_id,
            )

            if found is not None:
                return found

    return None


def cpu_power_w():
    node = find_sensor(
        get_json(),
        CPU_POWER_ID,
    )

    if node is None:
        raise RuntimeError(
            "CPU Package power sensor not found"
        )

    return float(
        str(node["Value"])
        .replace(" W", "")
        .replace(",", ".")
        .strip()
    )


def wait_for_file(path, process, timeout=180):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if path.exists():
            return

        if process.poll() is not None:
            output, _ = process.communicate()

            raise RuntimeError(
                f"Worker exited early:\n{output}"
            )

        time.sleep(0.05)

    raise TimeoutError(
        f"Timed out waiting for {path}"
    )


def launch_worker(
    placement,
    directory,
    tag,
):
    result = directory / f"{tag}.json"
    ready = directory / f"{tag}.ready"
    go = directory / f"{tag}.go"
    started = directory / f"{tag}.started"

    cmd = [
        sys.executable,
        str(WORKER),

        "--placement",
        placement,

        "--threads",
        str(THREADS),

        "--seconds",
        str(DURATION_SEC),

        "--result",
        str(result),

        "--ready-file",
        str(ready),

        "--go-file",
        str(go),

        "--started-file",
        str(started),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    wait_for_file(
        ready,
        process,
    )

    return (
        process,
        result,
        go,
        started,
    )


def run_no_monitor(
    placement,
    directory,
    tag,
):
    process, result, go, started = launch_worker(
        placement,
        directory,
        tag,
    )

    go.touch()

    wait_for_file(
        started,
        process,
    )

    output, _ = process.communicate(
        timeout=DURATION_SEC + 120
    )

    if process.returncode != 0:
        raise RuntimeError(output)

    print(output.strip())

    return json.loads(
        result.read_text()
    )


def run_monitored(
    placement,
    directory,
    tag,
):
    process, result, go, started = launch_worker(
        placement,
        directory,
        tag,
    )

    go.touch()

    wait_for_file(
        started,
        process,
    )

    powers = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                powers.append(
                    cpu_power_w()
                )
            except Exception:
                pass

            stop.wait(POLL_SEC)

    thread = threading.Thread(
        target=poll,
        daemon=True,
    )

    thread.start()

    output, _ = process.communicate(
        timeout=DURATION_SEC + 120
    )

    stop.set()
    thread.join(timeout=5)

    if process.returncode != 0:
        raise RuntimeError(output)

    if not powers:
        raise RuntimeError(
            "No LHM power samples collected"
        )

    print(output.strip())

    row = json.loads(
        result.read_text()
    )

    row["mean_power_w"] = statistics.mean(
        powers
    )

    row["median_power_w"] = statistics.median(
        powers
    )

    row["min_power_w"] = min(powers)
    row["max_power_w"] = max(powers)

    row["power_samples"] = len(powers)

    row["energy_img_j"] = (
        row["mean_power_w"]
        / row["throughput_img_s"]
    )

    return row


def main():
    print(
        f"LHM CPU Package: "
        f"{cpu_power_w():.2f} W"
    )

    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)

        for placement in PLACEMENTS:
            for repeat in range(1, REPEATS + 1):

                print(
                    f"\n=== {placement} R{repeat} ==="
                )

                no = run_no_monitor(
                    placement,
                    directory,
                    f"{placement}_r{repeat}_no",
                )

                time.sleep(2)

                mon = run_monitored(
                    placement,
                    directory,
                    f"{placement}_r{repeat}_mon",
                )

                delta_pct = (
                    (
                        mon["throughput_img_s"]
                        - no["throughput_img_s"]
                    )
                    / no["throughput_img_s"]
                    * 100.0
                )

                row = {
                    "placement": placement,
                    "repeat": repeat,
                    "threads": THREADS,

                    "process_affinity_mask":
                        mon["process_affinity_mask"],

                    "no_monitor_img_s":
                        no["throughput_img_s"],

                    "monitored_img_s":
                        mon["throughput_img_s"],

                    "monitor_delta_pct":
                        delta_pct,

                    "latency_ms_img":
                        mon["latency_ms_img"],

                    "mean_power_w":
                        mon["mean_power_w"],

                    "median_power_w":
                        mon["median_power_w"],

                    "min_power_w":
                        mon["min_power_w"],

                    "max_power_w":
                        mon["max_power_w"],

                    "energy_img_j":
                        mon["energy_img_j"],

                    "power_samples":
                        mon["power_samples"],
                }

                rows.append(row)

                print(
                    f"{placement:17} "
                    f"R{repeat} | "
                    f"NO {row['no_monitor_img_s']:.3f} | "
                    f"MON {row['monitored_img_s']:.3f} | "
                    f"{delta_pct:+.2f}% | "
                    f"{row['mean_power_w']:.2f} W | "
                    f"{row['energy_img_j']:.2f} J/img | "
                    f"{row['power_samples']} samples"
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
    print("-" * 100)

    for placement in PLACEMENTS:
        group = [
            r for r in rows
            if r["placement"] == placement
        ]

        def med(key):
            return statistics.median(
                r[key]
                for r in group
            )

        print(
            f"{placement:17} | "
            f"NO {med('no_monitor_img_s'):.3f} | "
            f"MON {med('monitored_img_s'):.3f} | "
            f"delta {med('monitor_delta_pct'):+.2f}% | "
            f"{med('mean_power_w'):.2f} W | "
            f"{med('energy_img_j'):.2f} J/img"
        )

    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
