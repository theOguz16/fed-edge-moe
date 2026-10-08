import csv
import json
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


CONFIG_ORDERS = [
    [
        "all_cpu",
        "all_mps",
        "stage1_seq",
        "stage1_pipe",
        "stage2_seq",
        "stage2_pipe",
        "stage3_seq",
        "stage3_pipe",
    ],
    [
        "stage3_pipe",
        "stage3_seq",
        "stage2_pipe",
        "stage2_seq",
        "stage1_pipe",
        "stage1_seq",
        "all_mps",
        "all_cpu",
    ],
    [
        "stage1_pipe",
        "stage2_seq",
        "all_cpu",
        "stage3_pipe",
        "all_mps",
        "stage3_seq",
        "stage1_seq",
        "stage2_pipe",
    ],
]


THREADS = 4
DURATION_SEC = 20.0
INTERVAL_MS = 1000
COOLDOWN_SEC = 4.0


WORKER = Path(
    "benchmarks/"
    "convnext_large_vision_hard_mac_"
    "pipeline_power_worker.py"
)


OUT = Path(
    "results/"
    "convnext_large_vision_hard_mac_"
    "pipeline_power.csv"
)


POWER_RE = re.compile(
    r"Combined Power "
    r"\(CPU \+ GPU \+ ANE\):"
    r"\s*([\d.]+)\s*mW"
)


def median_or_none(values):
    if not values:
        return None

    return statistics.median(
        values
    )


def mean_or_none(values):
    if not values:
        return None

    return statistics.mean(
        values
    )


def parse_powermetrics(
    text,
    pid,
):

    blocks = text.split(
        "*** Sampled system activity"
    )

    power_samples = []
    cpu_ms_samples = []
    pcore_share_samples = []


    proc_re = re.compile(
        rf"(?m)^\s*Python\s+"
        rf"{pid}\s+(.+)$"
    )


    for block in blocks:

        power_match = (
            POWER_RE.search(
                block
            )
        )

        if power_match:

            power_samples.append(
                float(
                    power_match.group(1)
                )
                / 1000.0
            )


        proc_match = (
            proc_re.search(
                block
            )
        )

        if not proc_match:
            continue


        tokens = (
            proc_match
            .group(1)
            .split()
        )


        if len(tokens) < 15:
            continue


        try:

            cpu_ms = float(
                tokens[0]
            )

            pcpu_pct = float(
                tokens[14]
            )

        except ValueError:
            continue


        cpu_ms_samples.append(
            cpu_ms
        )

        pcore_share_samples.append(
            pcpu_pct
        )


    # powermetrics is intentionally
    # started before workload release.
    # Remove the initial idle/startup
    # power sample when possible.
    if len(power_samples) >= 4:

        power_samples = (
            power_samples[1:]
        )


    if not power_samples:

        raise RuntimeError(
            "No Combined Power samples"
        )


    return {
        "powermetrics_samples":
            len(power_samples),

        "mean_power_w":
            mean_or_none(
                power_samples
            ),

        "median_power_w":
            median_or_none(
                power_samples
            ),

        "process_cpu_equiv_cores":
            (
                median_or_none(
                    cpu_ms_samples
                )
                / 1000.0
                if cpu_ms_samples
                else None
            ),

        "process_pcore_share_pct":
            median_or_none(
                pcore_share_samples
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

    while (
        time.monotonic()
        < deadline
    ):

        if path.exists():
            return

        if (
            process.poll()
            is not None
        ):

            output, _ = (
                process.communicate()
            )

            raise RuntimeError(
                "Worker exited early:\n"
                + output
            )

        time.sleep(
            0.05
        )


    raise TimeoutError(
        f"Timed out waiting "
        f"for {path}"
    )


def stop_powermetrics(pm):

    if pm.poll() is not None:
        return

    pm.send_signal(
        signal.SIGINT
    )

    try:

        pm.wait(
            timeout=5
        )

    except subprocess.TimeoutExpired:

        pm.kill()

        pm.wait()


def run_one(
    config,
    repeat,
    directory,
):

    tag = (
        f"{config}_r{repeat}"
    )

    ready = (
        directory
        / f"{tag}.ready"
    )

    go = (
        directory
        / f"{tag}.go"
    )

    result_file = (
        directory
        / f"{tag}.json"
    )

    pm_file = (
        directory
        / f"{tag}_powermetrics.txt"
    )


    worker_cmd = [
        sys.executable,
        str(WORKER),

        "--config",
        config,

        "--threads",
        str(THREADS),

        "--seconds",
        str(DURATION_SEC),

        "--result",
        str(result_file),

        "--ready-file",
        str(ready),

        "--go-file",
        str(go),
    ]


    worker = subprocess.Popen(
        worker_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


    wait_for_file(
        ready,
        worker,
        timeout=180,
    )


    ready_data = json.loads(
        ready.read_text()
    )

    pid = int(
        ready_data["pid"]
    )


    with pm_file.open(
        "w"
    ) as pf:

        pm = subprocess.Popen(
            [
                "sudo",
                "-n",
                "powermetrics",

                "--samplers",
                "tasks,cpu_power",

                "--show-process-qos",
                "--show-process-amp",

                "-i",
                str(
                    INTERVAL_MS
                ),
            ],
            stdout=pf,
            stderr=subprocess.STDOUT,
        )


        # Let powermetrics enter its
        # sampling loop before release.
        time.sleep(
            1.2
        )


        go.touch()


        try:

            output, _ = (
                worker.communicate(
                    timeout=
                        DURATION_SEC
                        + 180
                )
            )

        except subprocess.TimeoutExpired:

            worker.kill()

            output, _ = (
                worker.communicate()
            )

            stop_powermetrics(
                pm
            )

            raise RuntimeError(
                "Worker timeout:\n"
                + output
            )


        stop_powermetrics(
            pm
        )


    if worker.returncode != 0:

        raise RuntimeError(
            "Worker failed:\n"
            + output
        )


    if not result_file.exists():

        raise RuntimeError(
            "Missing worker result:\n"
            + output
        )


    worker_result = json.loads(
        result_file.read_text()
    )


    power_result = (
        parse_powermetrics(
            pm_file.read_text(
                errors="replace"
            ),
            pid,
        )
    )


    throughput = (
        worker_result[
            "throughput_img_s"
        ]
    )

    mean_power = (
        power_result[
            "mean_power_w"
        ]
    )

    energy_img = (
        mean_power
        / throughput
    )


    row = {
        "config":
            config,

        "repeat":
            repeat,

        "threads":
            THREADS,

        "resolution":
            worker_result[
                "resolution"
            ],

        "batch":
            worker_result[
                "batch"
            ],

        "images":
            worker_result[
                "images"
            ],

        "elapsed_sec":
            worker_result[
                "elapsed_sec"
            ],

        "throughput_img_s":
            throughput,

        "median_latency_ms":
            worker_result[
                "median_latency_ms"
            ],

        "p95_latency_ms":
            worker_result[
                "p95_latency_ms"
            ],

        "mean_combined_power_w":
            mean_power,

        "median_combined_power_w":
            power_result[
                "median_power_w"
            ],

        "energy_img_j":
            energy_img,

        "mps_current_alloc_mb":
            worker_result[
                "mps_current_alloc_mb"
            ],

        "mps_driver_alloc_mb":
            worker_result[
                "mps_driver_alloc_mb"
            ],

        "process_cpu_equiv_cores":
            power_result[
                "process_cpu_equiv_cores"
            ],

        "process_pcore_share_pct":
            power_result[
                "process_pcore_share_pct"
            ],

        "powermetrics_samples":
            power_result[
                "powermetrics_samples"
            ],
    }


    pshare = (
        row[
            "process_pcore_share_pct"
        ]
    )

    pshare_text = (
        f"{pshare:5.1f}%"
        if pshare is not None
        else "  n/a "
    )


    print(
        f"{config:12} "
        f"R{repeat} | "
        f"{throughput:6.2f} img/s | "
        f"{row['median_latency_ms']:7.1f} ms | "
        f"{mean_power:5.2f} W | "
        f"{energy_img:6.3f} J/img | "
        f"MPS "
        f"{row['mps_current_alloc_mb']:7.1f} MB | "
        f"Pshare {pshare_text}"
    )


    return row


# Verify cached sudo credential.
subprocess.run(
    [
        "sudo",
        "-n",
        "-v",
    ],
    check=True,
)


rows = []


with tempfile.TemporaryDirectory(
    prefix="mac_pipeline_power_"
) as temp:

    directory = Path(
        temp
    )


    for repeat, order in enumerate(
        CONFIG_ORDERS,
        start=1,
    ):

        print(
            f"\n===== REPEAT "
            f"{repeat} ====="
        )


        for config in order:

            # Refresh sudo timestamp
            # while the formal sweep runs.
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "-v",
                ],
                check=True,
            )


            row = run_one(
                config,
                repeat,
                directory,
            )

            rows.append(
                row
            )

            time.sleep(
                COOLDOWN_SEC
            )


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
        fieldnames=
            rows[0].keys(),
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print(
    "\nMEDIAN SUMMARY"
)

print(
    "-" * 145
)


configs = [
    "all_cpu",
    "all_mps",
    "stage1_seq",
    "stage1_pipe",
    "stage2_seq",
    "stage2_pipe",
    "stage3_seq",
    "stage3_pipe",
]


points = {}


for config in configs:

    group = [
        r
        for r in rows
        if r["config"] == config
    ]


    def med(key):

        values = [
            float(r[key])
            for r in group
            if r[key] is not None
        ]

        if not values:
            return None

        return statistics.median(
            values
        )


    throughput = med(
        "throughput_img_s"
    )

    energy = med(
        "energy_img_j"
    )

    points[
        config
    ] = (
        throughput,
        energy,
    )


    print(
        f"{config:12} | "
        f"thr "
        f"{throughput:6.2f} | "
        f"lat "
        f"{med('median_latency_ms'):7.1f} ms | "
        f"P "
        f"{med('mean_combined_power_w'):5.2f} W | "
        f"E "
        f"{energy:6.3f} J/img | "
        f"MPS "
        f"{med('mps_current_alloc_mb'):7.1f} MB | "
        f"driver "
        f"{med('mps_driver_alloc_mb'):7.1f} MB"
    )


print(
    "\nENERGY-THROUGHPUT "
    "PARETO"
)

print(
    "-" * 75
)


for config, (
    throughput,
    energy,
) in points.items():

    dominated = False


    for other, (
        other_thr,
        other_energy,
    ) in points.items():

        if other == config:
            continue


        no_worse = (
            other_thr
            >= throughput
            and
            other_energy
            <= energy
        )

        strict = (
            other_thr
            > throughput
            or
            other_energy
            < energy
        )


        if (
            no_worse
            and
            strict
        ):

            dominated = True
            break


    print(
        f"{config:12} | "
        f"{throughput:6.2f} img/s | "
        f"{energy:6.3f} J/img | "
        f"{'dominated' if dominated else 'PARETO'}"
    )


print(
    "\nSaved:",
    OUT,
)
