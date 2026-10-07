import argparse
import ctypes
import json
import os
import time
from ctypes import wintypes
from pathlib import Path

from windows_qos import (
    set_process_qos,
    get_process_qos_state,
)


WORKLOADS = {
    "highres_single": (320, 1),
    "batched_load": (224, 2),
}


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


kernel32 = ctypes.WinDLL(
    "kernel32",
    use_last_error=True,
)

psapi = ctypes.WinDLL(
    "psapi",
    use_last_error=True,
)

GetCurrentProcess = kernel32.GetCurrentProcess
GetCurrentProcess.restype = wintypes.HANDLE

GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
    wintypes.DWORD,
]
GetProcessMemoryInfo.restype = wintypes.BOOL


def memory_info():
    c = PROCESS_MEMORY_COUNTERS_EX()
    c.cb = ctypes.sizeof(c)

    ok = GetProcessMemoryInfo(
        GetCurrentProcess(),
        ctypes.byref(c),
        c.cb,
    )

    if not ok:
        raise OSError(
            ctypes.get_last_error(),
            "GetProcessMemoryInfo failed",
        )

    mb = 1024 * 1024

    return {
        "rss_mb":
            c.WorkingSetSize / mb,

        "peak_rss_mb":
            c.PeakWorkingSetSize / mb,

        "private_mb":
            c.PrivateUsage / mb,
    }


def wait_for_file(path, timeout=120):
    deadline = time.monotonic() + timeout

    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for {path}"
            )

        time.sleep(0.05)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--workload",
        required=True,
        choices=WORKLOADS,
    )

    p.add_argument(
        "--threads",
        required=True,
        type=int,
    )

    p.add_argument(
        "--seconds",
        type=float,
        default=20.0,
    )

    p.add_argument("--result", required=True)
    p.add_argument("--ready-file", required=True)
    p.add_argument("--go-file", required=True)
    p.add_argument("--started-file", required=True)

    args = p.parse_args()

    # Fix scheduling policy. Thread count is our main variable.
    set_process_qos("high")
    qos = get_process_qos_state()

    os.environ["OMP_NUM_THREADS"] = str(
        args.threads
    )

    os.environ["MKL_NUM_THREADS"] = str(
        args.threads
    )

    import torch

    from torchvision.models import (
        convnext_large,
        ConvNeXt_Large_Weights,
    )

    torch.set_num_threads(args.threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    res, batch = WORKLOADS[
        args.workload
    ]

    model = convnext_large(
        weights=ConvNeXt_Large_Weights.DEFAULT
    )

    model.eval()

    x = torch.randn(
        batch,
        3,
        res,
        res,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        for _ in range(3):
            _ = model(x)

    ready = Path(args.ready_file)
    go = Path(args.go_file)
    started = Path(args.started_file)

    ready.write_text("ready")
    wait_for_file(go)

    iterations = 0
    images = 0

    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    started.write_text("started")

    with torch.inference_mode():
        while True:
            _ = model(x)

            iterations += 1
            images += batch

            if (
                time.perf_counter()
                - wall_start
                >= args.seconds
            ):
                break

    elapsed = (
        time.perf_counter()
        - wall_start
    )

    cpu_time = (
        time.process_time()
        - cpu_start
    )

    mem = memory_info()

    row = {
        "workload": args.workload,
        "resolution": res,
        "batch": batch,
        "threads": args.threads,

        "duration_sec": elapsed,
        "iterations": iterations,
        "images": images,

        "throughput_img_s":
            images / elapsed,

        "batch_latency_ms":
            elapsed / iterations * 1000.0,

        "image_time_ms":
            elapsed / images * 1000.0,

        "process_cpu_time_sec":
            cpu_time,

        "process_cpu_equiv_cores":
            cpu_time / elapsed,

        "rss_mb":
            mem["rss_mb"],

        "peak_rss_mb":
            mem["peak_rss_mb"],

        "private_mb":
            mem["private_mb"],

        "qos_controlled":
            qos[
                "execution_speed_controlled"
            ],

        "qos_throttled":
            qos[
                "execution_speed_throttled"
            ],
    }

    Path(args.result).write_text(
        json.dumps(
            row,
            indent=2,
        )
    )

    print(
        f"{args.workload:14} "
        f"{args.threads:2}t | "
        f"{row['throughput_img_s']:.3f} img/s | "
        f"CPUeq={row['process_cpu_equiv_cores']:.2f} | "
        f"RSS={row['rss_mb']:.0f} MB"
    )


if __name__ == "__main__":
    main()
