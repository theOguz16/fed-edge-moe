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


WORKLOADS = [
    "highres_single",
    "batched_load",
]

THREAD_ORDERS = [
    [1, 2, 4, 8, 16],
    [16, 8, 4, 2, 1],
    [4, 8, 16, 1, 2],
]

DURATION_SEC = 20.0
POLL_SEC = 1.0
COOLDOWN_SEC = 3.0

WORKER = Path(
    "benchmarks/"
    "convnext_large_vision_hard_msi_"
    "thread_mechanism_worker.py"
)

OUT = Path(
    "results/"
    "convnext_large_vision_hard_msi_"
    "thread_mechanism.csv"
)

LHM_URL = "http://localhost:8085/data.json"

POWER_ID = "/intelcpu/0/power/0"
LOAD_TOTAL_ID = "/intelcpu/0/load/0"
LOAD_MAX_ID = "/intelcpu/0/load/1"
PACKAGE_TEMP_ID = "/intelcpu/0/temperature/10"

CLOCK_IDS = [
    f"/intelcpu/0/clock/{i}"
    for i in range(1, 9)
]


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


def sensor_float(data, sensor_id):
    node = find_sensor(
        data,
        sensor_id,
    )

    if node is None:
        raise RuntimeError(
            f"Sensor not found: {sensor_id}"
        )

    value = (
        str(node["Value"])
        .replace(",", ".")
        .strip()
        .split()[0]
    )

    return float(value)


def read_metrics():
    data = get_json()

    clocks = [
        sensor_float(
            data,
            sensor_id,
        )
        for sensor_id in CLOCK_IDS
    ]

    return {
        "power_w":
            sensor_float(
                data,
                POWER_ID,
            ),

        "cpu_total_load_pct":
            sensor_float(
                data,
                LOAD_TOTAL_ID,
            ),

        "cpu_core_max_load_pct":
            sensor_float(
                data,
                LOAD_MAX_ID,
            ),

        "mean_core_clock_mhz":
            statistics.mean(clocks),

        "max_core_clock_mhz":
            max(clocks),

        "package_temp_c":
            sensor_float(
                data,
                PACKAGE_TEMP_ID,
            ),
    }


def wait_for_file(
    path,
    process,
    timeout=180,
):
    deadline = (
        time.monotonic()
        + timeout
    )

    while time.monotonic() < deadline:
        if path.exists():
            return

        if process.poll() is not None:
            output, _ = (
                process.communicate()
            )

            raise RuntimeError(
                "Worker exited early:\n"
                + output
            )

        time.sleep(0.05)

    raise TimeoutError(
        f"Timed out waiting for {path}"
    )


def run_one(
    workload,
    threads,
    repeat,
    directory,
):
    tag = (
        f"{workload}_"
        f"{threads}t_"
        f"r{repeat}"
    )

    result = directory / f"{tag}.json"
    ready = directory / f"{tag}.ready"
    go = directory / f"{tag}.go"
    started = directory / f"{tag}.started"

    cmd = [
        sys.executable,
        str(WORKER),

        "--workload",
        workload,

        "--threads",
        str(threads),

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

    go.touch()

    wait_for_file(
        started,
        process,
    )

    samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                samples.append(
                    read_metrics()
                )
            except Exception:
                pass

            stop.wait(POLL_SEC)

    monitor = threading.Thread(
        target=poll,
        daemon=True,
    )

    monitor.start()

    output, _ = process.communicate(
        timeout=DURATION_SEC + 180
    )

    stop.set()
    monitor.join(timeout=5)

    if process.returncode != 0:
        raise RuntimeError(output)

    if not samples:
        raise RuntimeError(
            "No LHM samples collected"
        )

    worker = json.loads(
        result.read_text()
    )

    def mean(key):
        return statistics.mean(
            sample[key]
            for sample in samples
        )

    worker["repeat"] = repeat

    worker["mean_power_w"] = mean(
        "power_w"
    )

    worker["energy_img_j"] = (
        worker["mean_power_w"]
        / worker["throughput_img_s"]
    )

    worker[
        "mean_cpu_total_load_pct"
    ] = mean(
        "cpu_total_load_pct"
    )

    worker[
        "mean_cpu_core_max_load_pct"
    ] = mean(
        "cpu_core_max_load_pct"
    )

    worker[
        "mean_core_clock_mhz"
    ] = mean(
        "mean_core_clock_mhz"
    )

    worker[
        "mean_max_core_clock_mhz"
    ] = mean(
        "max_core_clock_mhz"
    )

    worker[
        "mean_package_temp_c"
    ] = mean(
        "package_temp_c"
    )

    worker["samples"] = len(samples)

    print(output.strip())

    print(
        f"  power={worker['mean_power_w']:.2f} W | "
        f"energy={worker['energy_img_j']:.2f} J/img | "
        f"load={worker['mean_cpu_total_load_pct']:.1f}% | "
        f"clock={worker['mean_core_clock_mhz']:.0f} MHz | "
        f"temp={worker['mean_package_temp_c']:.1f} C"
    )

    return worker


def save(rows):
    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
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


def main():
    first = read_metrics()

    print(
        "LHM OK | "
        f"{first['power_w']:.1f} W | "
        f"{first['cpu_total_load_pct']:.1f}% load | "
        f"{first['package_temp_c']:.1f} C"
    )

    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)

        for repeat, thread_order in enumerate(
            THREAD_ORDERS,
            start=1,
        ):
            workload_order = (
                WORKLOADS
                if repeat % 2 == 1
                else list(reversed(WORKLOADS))
            )

            for workload in workload_order:
                for threads in thread_order:

                    print(
                        f"\n=== "
                        f"{workload} | "
                        f"{threads}t | "
                        f"R{repeat} ==="
                    )

                    row = run_one(
                        workload,
                        threads,
                        repeat,
                        directory,
                    )

                    rows.append(row)
                    save(rows)

                    time.sleep(
                        COOLDOWN_SEC
                    )

    print("\nMEDIANS")
    print("-" * 125)

    for workload in WORKLOADS:
        print(f"\n{workload}")

        workload_rows = [
            r for r in rows
            if r["workload"] == workload
        ]

        medians = {}

        for threads in [
            1, 2, 4, 8, 16
        ]:
            group = [
                r for r in workload_rows
                if r["threads"] == threads
            ]

            def med(key):
                return statistics.median(
                    float(r[key])
                    for r in group
                )

            medians[threads] = {
                "thr":
                    med("throughput_img_s"),

                "power":
                    med("mean_power_w"),

                "energy":
                    med("energy_img_j"),

                "cpueq":
                    med(
                        "process_cpu_equiv_cores"
                    ),

                "load":
                    med(
                        "mean_cpu_total_load_pct"
                    ),

                "clock":
                    med(
                        "mean_core_clock_mhz"
                    ),

                "temp":
                    med(
                        "mean_package_temp_c"
                    ),

                "rss":
                    med("rss_mb"),
            }

        base = medians[1]["thr"]
        previous = None

        for threads in [
            1, 2, 4, 8, 16
        ]:
            m = medians[threads]

            speedup = (
                m["thr"] / base
            )

            efficiency = (
                speedup
                / threads
                * 100.0
            )

            if previous is None:
                marginal = 0.0
            else:
                marginal = (
                    (
                        m["thr"]
                        - previous
                    )
                    / previous
                    * 100.0
                )

            print(
                f"{threads:2}t | "
                f"{m['thr']:6.3f} img/s | "
                f"{m['power']:5.2f} W | "
                f"{m['energy']:6.2f} J/img | "
                f"CPUeq {m['cpueq']:5.2f} | "
                f"load {m['load']:5.1f}% | "
                f"clock {m['clock']:6.0f} MHz | "
                f"temp {m['temp']:4.1f} C | "
                f"speedup {speedup:4.2f}x | "
                f"eff {efficiency:5.1f}% | "
                f"marginal {marginal:+5.1f}%"
            )

            previous = m["thr"]

    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
