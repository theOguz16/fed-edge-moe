import argparse
import csv
import ctypes
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from ctypes import wintypes
from pathlib import Path

MODEL = "ggml-org/Qwen3-1.7B-GGUF:Q4_K_M"
EXE = Path("tools/llama-cuda/bin/llama-batched-bench.exe")

WORKLOADS = {
    "long_single": (128, 128, 1),
    "batch": (128, 64, 4),
}

THREADS = [1, 4, 8, 16]

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

kernel32.OpenProcess.argtypes = [
    wintypes.DWORD, wintypes.BOOL, wintypes.DWORD
]
kernel32.OpenProcess.restype = wintypes.HANDLE

kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL

psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
    wintypes.DWORD,
]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

parser = argparse.ArgumentParser()
parser.add_argument(
    "--workload",
    choices=["all", *WORKLOADS],
    default="all",
)
parser.add_argument("--threads", default="all")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument(
    "--output",
    default="results/qwen3_q4_msi_cpu_memory.csv",
)
args = parser.parse_args()

if not EXE.is_file():
    parser.error(f"Executable not found: {EXE}")

if args.repeats < 1:
    parser.error("--repeats must be positive")

if args.threads == "all":
    selected_threads = THREADS
else:
    try:
        selected_threads = [int(args.threads)]
    except ValueError:
        parser.error("--threads must be 1, 4, 8, 16 or all")
    if selected_threads[0] not in THREADS:
        parser.error("--threads must be 1, 4, 8, 16 or all")

selected_workloads = (
    WORKLOADS
    if args.workload == "all"
    else {args.workload: WORKLOADS[args.workload]}
)

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "model",
    "backend",
    "quantization",
    "workload",
    "context_tokens",
    "output_tokens",
    "batch",
    "threads",
    "repeat",
    "peak_working_set_mib",
    "max_sampled_private_commit_mib",
    "memory_measurement",
    "poll_interval_s",
    "samples",
    "exit_code",
]

summaries = defaultdict(list)
completed = 0

with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for workload, (ctx, output, batch) in selected_workloads.items():
        for threads in selected_threads:
            for repeat in range(1, args.repeats + 1):

                cmd = [
                    str(EXE),
                    "-hf", MODEL,
                    "-ngl", "0",
                    "-t", str(threads),
                    "-tb", str(threads),
                    "-npp", str(ctx),
                    "-ntg", str(output),
                    "-npl", str(batch),
                ]

                print(
                    f"{workload} | threads={threads} | "
                    f"repeat={repeat}/{args.repeats}",
                    flush=True,
                )

                with tempfile.TemporaryDirectory() as temp:
                    log_path = Path(temp) / "benchmark.log"

                    with log_path.open("wb") as log:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )

                        handle = kernel32.OpenProcess(
                            0x0400 | 0x0010,
                            False,
                            proc.pid,
                        )

                        if not handle:
                            proc.terminate()
                            proc.wait()
                            raise ctypes.WinError(
                                ctypes.get_last_error()
                            )

                        peak_ws = 0
                        max_private = 0
                        samples = 0

                        try:
                            while True:
                                counters = PROCESS_MEMORY_COUNTERS_EX()
                                counters.cb = ctypes.sizeof(counters)

                                ok = psapi.GetProcessMemoryInfo(
                                    handle,
                                    ctypes.byref(counters),
                                    counters.cb,
                                )

                                if ok:
                                    peak_ws = max(
                                        peak_ws,
                                        counters.PeakWorkingSetSize,
                                    )
                                    max_private = max(
                                        max_private,
                                        counters.PrivateUsage,
                                    )
                                    samples += 1

                                if proc.poll() is not None:
                                    break

                                time.sleep(0.2)

                            returncode = proc.wait()

                        finally:
                            kernel32.CloseHandle(handle)

                    if returncode != 0:
                        log_text = log_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                        raise RuntimeError(
                            f"Benchmark failed ({returncode}):\n"
                            + log_text[-2500:]
                        )

                if samples == 0 or peak_ws == 0:
                    raise RuntimeError("No valid memory samples")

                rss_mib = peak_ws / (1024 ** 2)
                private_mib = max_private / (1024 ** 2)

                row = {
                    "model": MODEL,
                    "backend": "cpu",
                    "quantization": "q4_k_m",
                    "workload": workload,
                    "context_tokens": ctx,
                    "output_tokens": output,
                    "batch": batch,
                    "threads": threads,
                    "repeat": repeat,
                    "peak_working_set_mib": rss_mib,
                    "max_sampled_private_commit_mib": private_mib,
                    "memory_measurement": "windows_psapi",
                    "poll_interval_s": 0.2,
                    "samples": samples,
                    "exit_code": returncode,
                }

                writer.writerow(row)
                f.flush()

                summaries[(workload, threads)].append(rss_mib)
                completed += 1

                print(
                    f"  Peak WS={rss_mib:.1f} MiB | "
                    f"sampled private={private_mib:.1f} MiB | "
                    f"samples={samples}",
                    flush=True,
                )

print("\nMEMORY SUMMARY")

for (workload, threads), values in sorted(summaries.items()):
    med = statistics.median(values)
    spread = 100 * (max(values) - min(values)) / med

    print(
        f"{workload:12s} threads={threads:2d} | "
        f"median={med:8.1f} MiB | "
        f"max={max(values):8.1f} MiB | "
        f"spread={spread:5.1f}%"
    )

print(f"\nCompleted: {completed}")
print(f"Saved: {out}")
