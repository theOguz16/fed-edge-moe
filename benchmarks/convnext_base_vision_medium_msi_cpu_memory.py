import argparse
import csv
import ctypes
import json
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

MODEL = "ConvNeXt-Base"
DEVICE = "Intel i7-11800H"
THREADS = (1, 8, 16)
RESOLUTION = 224
BATCH = 4
WARMUP = 2
INFERENCE_CALLS = 3

DEFAULT_OUTPUT = (
    "results/convnext_base_vision_medium_msi_cpu_memory.csv"
)

MARKER = "CONVNEXT_MEMORY_JSON:"


def read_windows_memory():
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

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)

    success = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )

    if not success:
        raise ctypes.WinError(ctypes.get_last_error())

    return {
        "peak_working_set_mib":
            counters.PeakWorkingSetSize / 1024**2,
        "private_commit_mib":
            counters.PrivateUsage / 1024**2,
    }


def worker(threads):
    if sys.platform != "win32":
        raise RuntimeError("This measurement requires Windows")

    import winreg
    import torch
    import torchvision
    from torchvision.models import (
        convnext_base,
        ConvNeXt_Base_Weights,
    )

    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
    ) as key:
        cpu_name = winreg.QueryValueEx(
            key, "ProcessorNameString"
        )[0]

    if "i7-11800H" not in cpu_name:
        raise RuntimeError(
            f"Unexpected CPU: {cpu_name}"
        )

    torch.set_num_threads(threads)
    torch.manual_seed(0)

    weights = ConvNeXt_Base_Weights.DEFAULT

    # Do not accidentally include a model download in
    # a process-memory measurement.
    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(weights.url).name
    )

    if not checkpoint.is_file():
        raise RuntimeError(
            f"Model weights not cached: {checkpoint}"
        )

    samples = []

    def sample():
        samples.append(read_windows_memory())

    model = convnext_base(weights=weights)
    model.eval()

    assert next(model.parameters()).dtype == torch.float32
    assert next(model.parameters()).device.type == "cpu"

    sample()

    x = torch.randn(
        BATCH, 3, RESOLUTION, RESOLUTION,
        dtype=torch.float32,
        device="cpu",
    )
    sample()

    with torch.inference_mode():
        for _ in range(WARMUP):
            y = model(x)
            assert tuple(y.shape) == (BATCH, 1000)
            sample()

        for _ in range(INFERENCE_CALLS):
            y = model(x)
            assert tuple(y.shape) == (BATCH, 1000)
            sample()

    last = read_windows_memory()
    samples.append(last)

    result = {
        "model": MODEL,
        "device": DEVICE,
        "backend": "cpu",
        "precision": "fp32",
        "workload": "batched_load",
        "profile": "medium",
        "resolution": RESOLUTION,
        "batch": BATCH,
        "threads": threads,
        "peak_working_set_mib": max(
            s["peak_working_set_mib"] for s in samples
        ),
        "max_sampled_private_commit_mib": max(
            s["private_commit_mib"] for s in samples
        ),
        "memory_measurement":
            "windows_psapi_process_peak_working_set",
        "private_commit_semantics":
            "sampled_at_phase_boundaries",
        "warmup": WARMUP,
        "inference_calls": INFERENCE_CALLS,
        "weights": "ConvNeXt_Base_Weights.DEFAULT",
        "cpu_name": cpu_name,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "exit_code": 0,
    }

    print(MARKER + json.dumps(result))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--threads",
        choices=("all", "1", "8", "16"),
        default="all",
    )
    parser.add_argument(
        "--repeats", type=int, default=3
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--worker", action="store_true"
    )

    args = parser.parse_args()

    if args.worker:
        if args.threads == "all":
            parser.error("Worker requires a single thread count")
        worker(int(args.threads))
        return

    if sys.platform != "win32":
        parser.error("Run measurements on the Windows MSI machine")

    if args.repeats < 1:
        parser.error("--repeats must be positive")

    selected = (
        THREADS
        if args.threads == "all"
        else (int(args.threads),)
    )

    records = []
    script = str(Path(__file__).resolve())

    for threads in selected:
        for repeat in range(1, args.repeats + 1):
            completed = subprocess.run(
                [
                    sys.executable,
                    script,
                    "--worker",
                    "--threads", str(threads),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            payloads = [
                line[len(MARKER):]
                for line in completed.stdout.splitlines()
                if line.startswith(MARKER)
            ]

            if completed.returncode != 0 or len(payloads) != 1:
                raise RuntimeError(
                    f"Memory worker failed: "
                    f"threads={threads}, repeat={repeat}\n"
                    + completed.stderr[-3000:]
                    + "\n"
                    + completed.stdout[-1000:]
                )

            record = json.loads(payloads[0])
            assert record["threads"] == threads
            assert record["exit_code"] == 0

            record["repeat"] = repeat
            records.append(record)

            print(
                f"{threads:2}t | repeat={repeat} | "
                f"peak working set="
                f"{record['peak_working_set_mib']:.1f} MiB | "
                f"sampled private commit="
                f"{record['max_sampled_private_commit_mib']:.1f} MiB",
                flush=True,
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)

    assert len(records) == len(selected) * args.repeats

    print("\nCONVNEXT-BASE WINDOWS CPU MEMORY: PASS")
    print("Records:", len(records))
    print("Saved:", out)


if __name__ == "__main__":
    main()
