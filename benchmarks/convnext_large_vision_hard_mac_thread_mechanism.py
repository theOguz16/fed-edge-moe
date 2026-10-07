import argparse
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


WORKLOADS = {
    "highres_single": (320, 1),
    "batched_load": (224, 2),
}

FORMAL_THREAD_ORDERS = [
    [1, 2, 4, 6, 8, 10],
    [10, 8, 6, 4, 2, 1],
    [4, 6, 8, 10, 1, 2],
]

WORKER = Path(
    "benchmarks/convnext_large_vision_hard_mac_qos_worker.py"
)

OUT = Path(
    "results/convnext_large_vision_hard_mac_thread_mechanism.csv"
)

INTERVAL_MS = 1000

POWER_RE = re.compile(
    r"Combined Power \(CPU \+ GPU \+ ANE\):"
    r"\s*([\d.]+)\s*mW"
)

E_FREQ_RE = re.compile(
    r"E-Cluster HW active frequency:\s*([\d.]+)\s*MHz"
)

E_RES_RE = re.compile(
    r"E-Cluster HW active residency:\s*([\d.]+)%"
)

P_FREQ_RE = re.compile(
    r"P-Cluster HW active frequency:\s*([\d.]+)\s*MHz"
)

P_RES_RE = re.compile(
    r"P-Cluster HW active residency:\s*([\d.]+)%"
)


def median_or_none(values):
    if not values:
        return None
    return statistics.median(values)


def mean_or_none(values):
    if not values:
        return None
    return statistics.mean(values)


def parse_powermetrics(text, pid):
    blocks = text.split(
        "*** Sampled system activity"
    )

    samples = []

    proc_re = re.compile(
        rf"(?m)^\s*Python\s+{pid}\s+(.+)$"
    )

    for block in blocks:
        match = proc_re.search(block)

        if not match:
            continue

        tokens = match.group(1).split()

        # Same column layout already validated
        # by the Mac QoS formal experiment.
        if len(tokens) < 15:
            continue

        try:
            cpu_ms = float(tokens[0])

            bg = float(tokens[8])
            util = float(tokens[9])
            default = float(tokens[10])
            uinit = float(tokens[11])
            uintr = float(tokens[12])

            pcpu_ms = float(tokens[13])
            pcpu_pct = float(tokens[14])

        except ValueError:
            continue

        # Exclude startup / waiting samples.
        if cpu_ms < 500.0:
            continue

        power_match = POWER_RE.search(block)

        e_freq_match = E_FREQ_RE.search(block)
        e_res_match = E_RES_RE.search(block)

        p_freq_match = P_FREQ_RE.search(block)
        p_res_match = P_RES_RE.search(block)

        samples.append({
            "cpu_ms_s": cpu_ms,

            "qos_bg_ms_s": bg,
            "qos_util_ms_s": util,
            "qos_default_ms_s": default,
            "qos_uinit_ms_s": uinit,
            "qos_uintr_ms_s": uintr,

            "pcpu_ms_s": pcpu_ms,
            "pcpu_pct": pcpu_pct,

            "power_w":
                float(power_match.group(1)) / 1000.0
                if power_match
                else None,

            "e_freq_mhz":
                float(e_freq_match.group(1))
                if e_freq_match
                else None,

            "e_active_pct":
                float(e_res_match.group(1))
                if e_res_match
                else None,

            "p_freq_mhz":
                float(p_freq_match.group(1))
                if p_freq_match
                else None,

            "p_active_pct":
                float(p_res_match.group(1))
                if p_res_match
                else None,
        })

    if not samples:
        raise RuntimeError(
            f"No valid powermetrics samples for PID {pid}"
        )

    def vals(key):
        return [
            s[key]
            for s in samples
            if s[key] is not None
        ]

    bg_sum = sum(vals("qos_bg_ms_s"))
    util_sum = sum(vals("qos_util_ms_s"))
    default_sum = sum(vals("qos_default_ms_s"))
    uinit_sum = sum(vals("qos_uinit_ms_s"))
    uintr_sum = sum(vals("qos_uintr_ms_s"))

    qos_total = (
        bg_sum
        + util_sum
        + default_sum
        + uinit_sum
        + uintr_sum
    )

    def share(value):
        if qos_total <= 0:
            return 0.0

        return 100.0 * value / qos_total

    pcpu_pct = median_or_none(
        vals("pcpu_pct")
    )

    return {
        "powermetrics_samples":
            len(samples),

        "cpu_ms_s_median":
            median_or_none(
                vals("cpu_ms_s")
            ),

        "process_cpu_equiv_cores":
            median_or_none(
                vals("cpu_ms_s")
            ) / 1000.0,

        "process_pcpu_ms_s_median":
            median_or_none(
                vals("pcpu_ms_s")
            ),

        "process_pcore_share_pct":
            pcpu_pct,

        "process_ecore_share_pct":
            100.0 - pcpu_pct,

        "qos_bg_pct":
            share(bg_sum),

        "qos_util_pct":
            share(util_sum),

        "qos_default_pct":
            share(default_sum),

        "qos_uinit_pct":
            share(uinit_sum),

        "qos_uintr_pct":
            share(uintr_sum),

        "mean_power_w":
            mean_or_none(
                vals("power_w")
            ),

        "e_cluster_freq_mhz_median":
            median_or_none(
                vals("e_freq_mhz")
            ),

        "e_cluster_active_pct_median":
            median_or_none(
                vals("e_active_pct")
            ),

        "p_cluster_freq_mhz_median":
            median_or_none(
                vals("p_freq_mhz")
            ),

        "p_cluster_active_pct_median":
            median_or_none(
                vals("p_active_pct")
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
    resolution,
    batch,
    threads,
    repeat,
    duration,
    temp_dir,
):
    tag = (
        f"{workload}_"
        f"{threads}t_"
        f"r{repeat}"
    )

    ready = temp_dir / f"{tag}.ready"
    go = temp_dir / f"{tag}.go"
    result = temp_dir / f"{tag}.json"
    pm_file = temp_dir / f"{tag}_powermetrics.txt"

    worker_cmd = [
        sys.executable,
        str(WORKER),

        "--qos",
        "default",

        "--threads",
        str(threads),

        "--resolution",
        str(resolution),

        "--batch",
        str(batch),

        "--seconds",
        str(duration),

        "--result",
        str(result),

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
    )

    ready_info = json.loads(
        ready.read_text()
    )

    pid = int(
        ready_info["pid"]
    )

    with pm_file.open("w") as pf:
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
                str(INTERVAL_MS),
            ],
            stdout=pf,
            stderr=subprocess.STDOUT,
        )

        # Start telemetry before releasing inference.
        time.sleep(1.2)

        go.touch()

        output, _ = worker.communicate(
            timeout=duration + 180
        )

        try:
            pm.send_signal(
                signal.SIGINT
            )
            pm.wait(timeout=8)

        except subprocess.TimeoutExpired:
            pm.terminate()
            pm.wait(timeout=5)

    if worker.returncode != 0:
        raise RuntimeError(
            "Worker failed:\n"
            + output
        )

    data = json.loads(
        result.read_text()
    )

    pm_data = parse_powermetrics(
        pm_file.read_text(
            errors="replace"
        ),
        pid,
    )

    row = {
        "workload": workload,
        "repeat": repeat,

        **data,
        **pm_data,
    }

    throughput = row[
        "throughput_img_s"
    ]

    power = row[
        "mean_power_w"
    ]

    row["energy_img_j"] = (
        power / throughput
        if power is not None
        and throughput > 0
        else None
    )

    print(output.strip())

    print(
        f"  power={row['mean_power_w']:.2f} W | "
        f"energy={row['energy_img_j']:.2f} J/img | "
        f"CPUeq={row['process_cpu_equiv_cores']:.2f} | "
        f"P-share={row['process_pcore_share_pct']:.1f}% | "
        f"E-active={row['e_cluster_active_pct_median']:.1f}% | "
        f"P-active={row['p_cluster_active_pct_median']:.1f}% | "
        f"E-freq={row['e_cluster_freq_mhz_median']:.0f} MHz | "
        f"P-freq={row['p_cluster_freq_mhz_median']:.0f} MHz"
    )

    return row


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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    subprocess.run(
        [
            "sudo",
            "-n",
            "-v",
        ],
        check=True,
    )

    if args.smoke:
        duration = 6.0

        plans = [
            (
                1,
                [
                    (
                        "highres_single",
                        [1, 4, 10],
                    )
                ],
            )
        ]

    else:
        duration = 20.0

        plans = []

        for repeat, order in enumerate(
            FORMAL_THREAD_ORDERS,
            start=1,
        ):
            workload_order = (
                list(WORKLOADS)
                if repeat % 2 == 1
                else list(
                    reversed(
                        list(WORKLOADS)
                    )
                )
            )

            plans.append(
                (
                    repeat,
                    [
                        (
                            workload,
                            order,
                        )
                        for workload
                        in workload_order
                    ],
                )
            )

    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)

        for repeat, workload_plans in plans:
            for workload, order in workload_plans:
                resolution, batch = (
                    WORKLOADS[workload]
                )

                for threads in order:
                    print(
                        f"\n=== "
                        f"{workload} | "
                        f"{threads}t | "
                        f"R{repeat} ==="
                    )

                    row = run_one(
                        workload,
                        resolution,
                        batch,
                        threads,
                        repeat,
                        duration,
                        temp_dir,
                    )

                    rows.append(row)

                    if not args.smoke:
                        save(rows)

                    time.sleep(2)

    if args.smoke:
        print(
            "\nSmoke test complete."
        )
        return

    print("\nMEDIANS")
    print("-" * 145)

    for workload in WORKLOADS:
        print(
            f"\n{workload}"
        )

        workload_rows = [
            r for r in rows
            if r["workload"]
            == workload
        ]

        medians = {}

        for threads in [
            1, 2, 4, 6, 8, 10
        ]:
            group = [
                r
                for r in workload_rows
                if r["threads"]
                == threads
            ]

            def med(key):
                values = [
                    float(r[key])
                    for r in group
                    if r[key]
                    is not None
                ]

                return statistics.median(
                    values
                )

            medians[threads] = {
                "thr":
                    med(
                        "throughput_img_s"
                    ),

                "power":
                    med(
                        "mean_power_w"
                    ),

                "energy":
                    med(
                        "energy_img_j"
                    ),

                "cpu":
                    med(
                        "process_cpu_equiv_cores"
                    ),

                "p_share":
                    med(
                        "process_pcore_share_pct"
                    ),

                "e_active":
                    med(
                        "e_cluster_active_pct_median"
                    ),

                "p_active":
                    med(
                        "p_cluster_active_pct_median"
                    ),

                "e_freq":
                    med(
                        "e_cluster_freq_mhz_median"
                    ),

                "p_freq":
                    med(
                        "p_cluster_freq_mhz_median"
                    ),

                "rss":
                    med(
                        "process_rss_peak_mb"
                    ),
            }

        base = medians[1]["thr"]
        previous = None

        for threads in [
            1, 2, 4, 6, 8, 10
        ]:
            m = medians[threads]

            speedup = (
                m["thr"]
                / base
            )

            efficiency = (
                speedup
                / threads
                * 100.0
            )

            marginal = (
                0.0
                if previous is None
                else (
                    (
                        m["thr"]
                        - previous
                    )
                    / previous
                    * 100.0
                )
            )

            print(
                f"{threads:2}t | "
                f"{m['thr']:6.3f} img/s | "
                f"{m['power']:5.2f} W | "
                f"{m['energy']:6.2f} J/img | "
                f"CPUeq {m['cpu']:4.2f} | "
                f"Pshare {m['p_share']:5.1f}% | "
                f"Eact {m['e_active']:5.1f}% | "
                f"Pact {m['p_active']:5.1f}% | "
                f"Efreq {m['e_freq']:4.0f} | "
                f"Pfreq {m['p_freq']:4.0f} MHz | "
                f"speedup {speedup:4.2f}x | "
                f"eff {efficiency:5.1f}% | "
                f"marginal {marginal:+5.1f}%"
            )

            previous = m["thr"]

    print(
        "\nSaved:",
        OUT,
    )


if __name__ == "__main__":
    main()
