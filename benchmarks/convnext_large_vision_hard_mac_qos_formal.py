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


QOS_CLASSES = [
    "background",
    "utility",
    "default",
    "userInitiated",
]

REPEATS = 3
THREADS = 4
RESOLUTION = 320
BATCH = 1
DURATION_SEC = 10.0
INTERVAL_MS = 1000

WORKER = Path(
    "benchmarks/convnext_large_vision_hard_mac_qos_worker.py"
)

OUT = Path(
    "results/convnext_large_vision_hard_mac_qos_formal.csv"
)

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
    blocks = text.split("*** Sampled system activity")

    samples = []

    proc_re = re.compile(
        rf"(?m)^\s*Python\s+{pid}\s+(.+)$"
    )

    for block in blocks:
        match = proc_re.search(block)

        if not match:
            continue

        tokens = match.group(1).split()

        # Expected columns after PID:
        #
        # 0 CPU ms/s
        # 1 User%
        # 2-3 deadlines
        # 4-5 wakeups
        # 6 Disabled
        # 7 Maint
        # 8 BG
        # 9 Util
        # 10 Default
        # 11 U-Init
        # 12 U-Intr
        # 13 PCPU ms/s
        # 14 %PCPU
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

        # Ignore waiting/startup samples.
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
                if power_match else None,

            "e_freq_mhz":
                float(e_freq_match.group(1))
                if e_freq_match else None,

            "e_active_pct":
                float(e_res_match.group(1))
                if e_res_match else None,

            "p_freq_mhz":
                float(p_freq_match.group(1))
                if p_freq_match else None,

            "p_active_pct":
                float(p_res_match.group(1))
                if p_res_match else None,
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

    return {
        "powermetrics_samples":
            len(samples),

        "cpu_ms_s_median":
            median_or_none(vals("cpu_ms_s")),

        "process_pcpu_ms_s_median":
            median_or_none(vals("pcpu_ms_s")),

        "process_pcpu_pct_median":
            median_or_none(vals("pcpu_pct")),

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
            mean_or_none(vals("power_w")),

        "e_cluster_freq_mhz_median":
            median_or_none(vals("e_freq_mhz")),

        "e_cluster_active_pct_median":
            median_or_none(vals("e_active_pct")),

        "p_cluster_freq_mhz_median":
            median_or_none(vals("p_freq_mhz")),

        "p_cluster_active_pct_median":
            median_or_none(vals("p_active_pct")),
    }


def wait_for_file(path, process, timeout=60):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if path.exists():
            return

        rc = process.poll()

        if rc is not None:
            raise RuntimeError(
                f"Worker exited before ready-file, rc={rc}"
            )

        time.sleep(0.05)

    raise TimeoutError(
        f"Timed out waiting for {path}"
    )


def run_one(qos, repeat, temp_dir):
    ready = temp_dir / f"{qos}_{repeat}.ready"
    go = temp_dir / f"{qos}_{repeat}.go"
    result = temp_dir / f"{qos}_{repeat}.json"
    pm_file = temp_dir / f"{qos}_{repeat}_powermetrics.txt"

    worker_cmd = [
        sys.executable,
        str(WORKER),
        "--qos",
        qos,
        "--threads",
        str(THREADS),
        "--resolution",
        str(RESOLUTION),
        "--batch",
        str(BATCH),
        "--seconds",
        str(DURATION_SEC),
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
        timeout=120,
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

        # Ensure powermetrics is sampling before inference.
        time.sleep(1.2)

        go.touch()

        output, _ = worker.communicate(
            timeout=DURATION_SEC + 60
        )

        # Capture a final interval.
        time.sleep(0.5)

        try:
            pm.send_signal(signal.SIGINT)
            pm.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pm.terminate()
            pm.wait()

    print(output.strip())

    if worker.returncode != 0:
        raise RuntimeError(
            f"Worker failed: {qos} R{repeat}"
        )

    data = json.loads(
        result.read_text()
    )

    pm_data = parse_powermetrics(
        pm_file.read_text(
            errors="replace"
        ),
        data["pid"],
    )

    data.update(pm_data)

    power = data["mean_power_w"]
    throughput = data["throughput_img_s"]

    data["energy_img_j"] = (
        power / throughput
        if power is not None and throughput > 0
        else None
    )

    data["repeat"] = repeat

    return data


def main():
    subprocess.run(
        ["sudo", "-n", "-v"],
        check=True,
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        temp_dir = Path(tmp)

        for qos in QOS_CLASSES:
            for repeat in range(1, REPEATS + 1):

                print()
                print(
                    f"=== {qos} R{repeat} ==="
                )

                row = run_one(
                    qos,
                    repeat,
                    temp_dir,
                )

                rows.append(row)

                print(
                    f"{qos:14} R{repeat} | "
                    f"{row['throughput_img_s']:.3f} img/s | "
                    f"{row['mean_power_w']:.2f} W | "
                    f"{row['energy_img_j']:.3f} J/img | "
                    f"P-core {row['process_pcpu_pct_median']:.1f}% | "
                    f"QoS "
                    f"BG {row['qos_bg_pct']:.1f}% "
                    f"Util {row['qos_util_pct']:.1f}% "
                    f"Def {row['qos_default_pct']:.1f}% "
                    f"UInit {row['qos_uinit_pct']:.1f}%"
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

    print()
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
