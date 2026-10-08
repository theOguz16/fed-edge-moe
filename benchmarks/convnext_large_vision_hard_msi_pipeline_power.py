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


try:
    import pynvml
except ImportError:
    raise RuntimeError(
        "pynvml missing"
    )


DURATION_SEC = 20.0
POLL_SEC = 0.5
COOLDOWN_SEC = 4.0

THREADS = 4


WORKER = Path(
    "benchmarks/"
    "convnext_large_vision_hard_msi_"
    "pipeline_power_worker.py"
)

OUT = Path(
    "results/"
    "convnext_large_vision_hard_msi_"
    "pipeline_power.csv"
)


LHM_URL = (
    "http://localhost:8085/"
    "data.json"
)

CPU_POWER_ID = (
    "/intelcpu/0/power/0"
)

CPU_LOAD_ID = (
    "/intelcpu/0/load/0"
)

CPU_TEMP_ID = (
    "/intelcpu/0/"
    "temperature/10"
)


BASE_ORDER = [
    "all_cpu",
    "all_gpu",
    "stage1_seq",
    "stage1_pipe",
    "stage2_seq",
    "stage2_pipe",
    "stage3_seq",
    "stage3_pipe",
]


ORDERS = [
    BASE_ORDER,

    list(
        reversed(
            BASE_ORDER
        )
    ),

    [
        "stage1_pipe",
        "stage3_seq",
        "all_gpu",
        "stage2_pipe",
        "all_cpu",
        "stage1_seq",
        "stage3_pipe",
        "stage2_seq",
    ],
]


def get_lhm_json():

    with urllib.request.urlopen(
        LHM_URL,
        timeout=3,
    ) as response:

        return json.load(
            response
        )


def find_sensor(
    node,
    sensor_id,
):

    if isinstance(
        node,
        dict,
    ):

        if (
            node.get("SensorId")
            == sensor_id
        ):
            return node

        for value in (
            node.values()
        ):

            found = find_sensor(
                value,
                sensor_id,
            )

            if found is not None:
                return found


    elif isinstance(
        node,
        list,
    ):

        for value in node:

            found = find_sensor(
                value,
                sensor_id,
            )

            if found is not None:
                return found

    return None


def sensor_float(
    data,
    sensor_id,
):

    node = find_sensor(
        data,
        sensor_id,
    )

    if node is None:

        raise RuntimeError(
            "Sensor not found: "
            + sensor_id
        )

    value = (
        str(node["Value"])
        .replace(",", ".")
        .strip()
        .split()[0]
    )

    return float(
        value
    )


def cpu_sample():

    data = get_lhm_json()

    return {
        "cpu_power_w":
            sensor_float(
                data,
                CPU_POWER_ID,
            ),

        "cpu_load_pct":
            sensor_float(
                data,
                CPU_LOAD_ID,
            ),

        "cpu_temp_c":
            sensor_float(
                data,
                CPU_TEMP_ID,
            ),
    }


pynvml.nvmlInit()

gpu_handle = (
    pynvml
    .nvmlDeviceGetHandleByIndex(
        0
    )
)


def gpu_sample():

    power = (
        pynvml
        .nvmlDeviceGetPowerUsage(
            gpu_handle
        )
        / 1000.0
    )

    util = (
        pynvml
        .nvmlDeviceGetUtilizationRates(
            gpu_handle
        )
        .gpu
    )

    memory = (
        pynvml
        .nvmlDeviceGetMemoryInfo(
            gpu_handle
        )
        .used
        / 1024**2
    )

    temp = (
        pynvml
        .nvmlDeviceGetTemperature(
            gpu_handle,
            pynvml
            .NVML_TEMPERATURE_GPU,
        )
    )

    return {
        "gpu_power_w":
            power,

        "gpu_util_pct":
            util,

        "gpu_mem_used_mb":
            memory,

        "gpu_temp_c":
            temp,
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
        f"Timed out: {path}"
    )


def run_one(
    config,
    repeat,
    directory,
):

    tag = (
        f"{config}_r{repeat}"
    )

    result_file = (
        directory
        / f"{tag}.json"
    )

    ready_file = (
        directory
        / f"{tag}.ready"
    )

    go_file = (
        directory
        / f"{tag}.go"
    )

    started_file = (
        directory
        / f"{tag}.started"
    )


    command = [
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
        str(ready_file),

        "--go-file",
        str(go_file),

        "--started-file",
        str(started_file),
    ]


    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


    wait_for_file(
        ready_file,
        process,
    )


    samples = []

    stop = threading.Event()


    def poll():

        while (
            not started_file.exists()
            and
            not stop.is_set()
        ):
            time.sleep(
                0.01
            )

        while not stop.is_set():

            try:

                cpu = cpu_sample()
                gpu = gpu_sample()

                samples.append({
                    "time":
                        time.perf_counter(),

                    **cpu,
                    **gpu,

                    "combined_power_w":
                        (
                            cpu[
                                "cpu_power_w"
                            ]
                            +
                            gpu[
                                "gpu_power_w"
                            ]
                        ),
                })

            except Exception as exc:

                print(
                    "monitor warning:",
                    exc,
                )

            stop.wait(
                POLL_SEC
            )


    poll_thread = (
        threading.Thread(
            target=poll,
            daemon=True,
        )
    )

    poll_thread.start()


    go_file.touch()


    wait_for_file(
        started_file,
        process,
        timeout=30,
    )


    try:

        output, _ = (
            process.communicate(
                timeout=
                    DURATION_SEC
                    + 180
            )
        )

    except subprocess.TimeoutExpired:

        process.kill()

        output, _ = (
            process.communicate()
        )

        raise RuntimeError(
            "Worker timeout:\n"
            + output
        )

    finally:

        stop.set()

        poll_thread.join(
            timeout=3
        )


    if process.returncode != 0:

        raise RuntimeError(
            "Worker failed:\n"
            + output
        )


    if not result_file.exists():

        raise RuntimeError(
            "Worker produced "
            "no result:\n"
            + output
        )


    if not samples:

        raise RuntimeError(
            "No power samples "
            f"for {config}"
        )


    worker_result = json.loads(
        result_file.read_text()
    )


    throughput = (
        worker_result[
            "throughput_img_s"
        ]
    )


    mean_cpu_power = (
        statistics.mean(
            s["cpu_power_w"]
            for s in samples
        )
    )

    mean_gpu_power = (
        statistics.mean(
            s["gpu_power_w"]
            for s in samples
        )
    )

    mean_combined_power = (
        statistics.mean(
            s["combined_power_w"]
            for s in samples
        )
    )


    cpu_energy_img = (
        mean_cpu_power
        / throughput
    )

    gpu_energy_img = (
        mean_gpu_power
        / throughput
    )

    combined_energy_img = (
        mean_combined_power
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

        "mean_cpu_package_power_w":
            mean_cpu_power,

        "mean_gpu_board_power_w":
            mean_gpu_power,

        "mean_compute_power_w":
            mean_combined_power,

        "cpu_energy_img_j":
            cpu_energy_img,

        "gpu_energy_img_j":
            gpu_energy_img,

        "compute_energy_img_j":
            combined_energy_img,

        "mean_cpu_load_pct":
            statistics.mean(
                s["cpu_load_pct"]
                for s in samples
            ),

        "peak_cpu_temp_c":
            max(
                s["cpu_temp_c"]
                for s in samples
            ),

        "mean_gpu_util_pct":
            statistics.mean(
                s["gpu_util_pct"]
                for s in samples
            ),

        "peak_gpu_temp_c":
            max(
                s["gpu_temp_c"]
                for s in samples
            ),

        "nvml_peak_used_mb":
            max(
                s["gpu_mem_used_mb"]
                for s in samples
            ),

        "torch_cuda_allocated_mb":
            worker_result[
                "cuda_allocated_mb"
            ],

        "torch_cuda_peak_mb":
            worker_result[
                "cuda_peak_mb"
            ],

        "power_samples":
            len(samples),
    }


    print(
        f"{config:12} "
        f"R{repeat} | "
        f"{throughput:6.2f} img/s | "
        f"{row['median_latency_ms']:7.1f} ms | "
        f"CPU {mean_cpu_power:5.1f} W | "
        f"GPU {mean_gpu_power:5.1f} W | "
        f"SUM {mean_combined_power:5.1f} W | "
        f"{combined_energy_img:6.3f} J/img | "
        f"VRAM "
        f"{row['torch_cuda_allocated_mb']:6.1f} MB"
    )


    return row


print(
    "Checking monitors..."
)

cpu_check = cpu_sample()
gpu_check = gpu_sample()

print(
    f"LHM CPU package: "
    f"{cpu_check['cpu_power_w']:.2f} W"
)

print(
    f"NVML GPU board: "
    f"{gpu_check['gpu_power_w']:.2f} W"
)


rows = []


with tempfile.TemporaryDirectory(
    prefix="pipeline_power_"
) as temp:

    directory = Path(
        temp
    )

    for repeat, order in enumerate(
        ORDERS,
        start=1,
    ):

        print(
            f"\n===== REPEAT "
            f"{repeat} ====="
        )

        for config in order:

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
    "-" * 150
)


median_points = {}


for config in BASE_ORDER:

    group = [
        r
        for r in rows
        if r["config"] == config
    ]


    def med(key):

        return statistics.median(
            float(r[key])
            for r in group
        )


    throughput = med(
        "throughput_img_s"
    )

    energy = med(
        "compute_energy_img_j"
    )

    median_points[
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
        f"CPU "
        f"{med('mean_cpu_package_power_w'):5.1f} W | "
        f"GPU "
        f"{med('mean_gpu_board_power_w'):5.1f} W | "
        f"SUM "
        f"{med('mean_compute_power_w'):5.1f} W | "
        f"E "
        f"{energy:6.3f} J/img | "
        f"VRAM "
        f"{med('torch_cuda_allocated_mb'):6.1f} MB"
    )


print(
    "\nENERGY-THROUGHPUT "
    "PARETO FRONT"
)

print(
    "-" * 80
)


for config, (
    throughput,
    energy,
) in median_points.items():

    dominated = False

    for other, (
        other_thr,
        other_energy,
    ) in median_points.items():

        if other == config:
            continue

        no_worse = (
            other_thr >= throughput
            and
            other_energy <= energy
        )

        strictly_better = (
            other_thr > throughput
            or
            other_energy < energy
        )

        if (
            no_worse
            and
            strictly_better
        ):

            dominated = True
            break


    status = (
        "PARETO"
        if not dominated
        else "dominated"
    )

    print(
        f"{config:12} | "
        f"{throughput:6.2f} img/s | "
        f"{energy:6.3f} J/img | "
        f"{status}"
    )


print(
    "\nSaved:",
    OUT,
)


pynvml.nvmlShutdown()
