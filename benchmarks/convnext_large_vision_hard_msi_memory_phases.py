import argparse
import csv
import gc
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

from resource_monitor import MemorySampler, memory_snapshot

try:
    import pynvml
except ImportError:
    raise RuntimeError(
        "pynvml missing. Install with: "
        "python -m pip install nvidia-ml-py"
    )


PROFILES = {
    "light": (160, 1),
    "medium": (224, 4),
    "heavy": (320, 16),
}

PRECISIONS = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}

REPEATS = 3
INFERENCE_SEC = 5.0

OUT = Path(
    "results/convnext_large_vision_hard_msi_memory_phases.csv"
)

MB = 1024 ** 2


def sync():
    torch.cuda.synchronize()


def stable_host_snapshot(samples=7, interval_sec=0.10):
    """
    Median host-memory snapshot to reduce short-lived OS noise.
    """
    rss = []
    used = []
    available = []

    for _ in range(samples):
        m = memory_snapshot(include_children=False)

        rss.append(m["process_rss_mb"])
        used.append(m["system_ram_used_mb"])
        available.append(m["system_ram_available_mb"])

        time.sleep(interval_sec)

    return {
        "process_rss_mb": statistics.median(rss),
        "system_ram_used_mb": statistics.median(used),
        "system_ram_available_mb": statistics.median(available),
    }


def cuda_snapshot(handle):
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return {
        "cuda_allocated_mb":
            torch.cuda.memory_allocated() / MB,

        "cuda_reserved_mb":
            torch.cuda.memory_reserved() / MB,

        "cuda_peak_allocated_mb":
            torch.cuda.max_memory_allocated() / MB,

        "cuda_peak_reserved_mb":
            torch.cuda.max_memory_reserved() / MB,

        "nvml_used_mb":
            meminfo.used / MB,
    }


def run_for_duration(model, x, batch, seconds):
    images = 0

    sync()
    start = time.perf_counter()

    with torch.inference_mode():
        while True:
            _ = model(x)
            images += batch

            if images % (batch * 4) == 0:
                sync()

                if time.perf_counter() - start >= seconds:
                    break

    sync()
    elapsed = time.perf_counter() - start

    return images / elapsed


def worker(precision, profile, repeat, result_path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    dtype = PRECISIONS[precision]
    res, batch = PROFILES[profile]

    device = torch.device("cuda")
    weights = ConvNeXt_Large_Weights.DEFAULT

    # --------------------------------------------------------
    # A. Before model load
    # --------------------------------------------------------

    gc.collect()
    torch.cuda.empty_cache()
    sync()

    a = stable_host_snapshot()
    a_cuda = cuda_snapshot(handle)

    # --------------------------------------------------------
    # B. CPU model loaded
    # --------------------------------------------------------

    model = convnext_large(weights=weights)
    model.eval()

    gc.collect()

    b = stable_host_snapshot()
    b_cuda = cuda_snapshot(handle)

    # --------------------------------------------------------
    # C. Model moved to CUDA
    # --------------------------------------------------------

    model = model.to(
        device=device,
        dtype=dtype,
    )

    sync()
    gc.collect()

    c = stable_host_snapshot()
    c_cuda = cuda_snapshot(handle)

    # --------------------------------------------------------
    # D. Input allocated
    # --------------------------------------------------------

    x = torch.randn(
        batch,
        3,
        res,
        res,
        device=device,
        dtype=dtype,
    )

    sync()

    d = stable_host_snapshot()
    d_cuda = cuda_snapshot(handle)

    # --------------------------------------------------------
    # E. Warm-up completed
    # --------------------------------------------------------

    with torch.inference_mode():
        for _ in range(5):
            _ = model(x)

    sync()

    e = stable_host_snapshot()
    e_cuda = cuda_snapshot(handle)

    # --------------------------------------------------------
    # F. Sustained inference peak
    # --------------------------------------------------------

    torch.cuda.reset_peak_memory_stats()
    sync()

    f_pre_cuda = cuda_snapshot(handle)

    sampler = MemorySampler(
        interval_sec=0.10,
        include_children=False,
    )

    sampler.start()

    throughput = run_for_duration(
        model,
        x,
        batch,
        INFERENCE_SEC,
    )

    f_ram = sampler.stop()
    f_cuda = cuda_snapshot(handle)

    row = {
        "precision": precision,
        "profile": profile,
        "resolution": res,
        "batch": batch,
        "repeat": repeat,

        "throughput_img_s": throughput,

        # Host memory stages
        "rss_A_before_model_mb":
            a["process_rss_mb"],

        "rss_B_cpu_model_mb":
            b["process_rss_mb"],

        "rss_C_device_model_mb":
            c["process_rss_mb"],

        "rss_D_input_mb":
            d["process_rss_mb"],

        "rss_E_warmup_mb":
            e["process_rss_mb"],

        "rss_F_pre_inference_mb":
            f_ram["process_rss_pre_mb"],

        "rss_F_inference_peak_mb":
            f_ram["process_rss_peak_mb"],

        # Host RSS deltas
        "rss_delta_A_B_model_load_mb":
            b["process_rss_mb"]
            - a["process_rss_mb"],

        "rss_delta_B_C_device_transfer_mb":
            c["process_rss_mb"]
            - b["process_rss_mb"],

        "rss_delta_C_D_input_mb":
            d["process_rss_mb"]
            - c["process_rss_mb"],

        "rss_delta_D_E_warmup_mb":
            e["process_rss_mb"]
            - d["process_rss_mb"],

        "rss_delta_E_F_inference_mb":
            f_ram["process_rss_peak_mb"]
            - e["process_rss_mb"],

        # CUDA / NVML stage C
        "cuda_C_allocated_mb":
            c_cuda["cuda_allocated_mb"],

        "cuda_C_reserved_mb":
            c_cuda["cuda_reserved_mb"],

        "nvml_C_used_mb":
            c_cuda["nvml_used_mb"],

        # Stage D
        "cuda_D_allocated_mb":
            d_cuda["cuda_allocated_mb"],

        "cuda_D_reserved_mb":
            d_cuda["cuda_reserved_mb"],

        "nvml_D_used_mb":
            d_cuda["nvml_used_mb"],

        # Stage E
        "cuda_E_allocated_mb":
            e_cuda["cuda_allocated_mb"],

        "cuda_E_reserved_mb":
            e_cuda["cuda_reserved_mb"],

        "cuda_E_peak_allocated_mb":
            e_cuda["cuda_peak_allocated_mb"],

        "nvml_E_used_mb":
            e_cuda["nvml_used_mb"],

        # Stage F
        "cuda_F_pre_allocated_mb":
            f_pre_cuda["cuda_allocated_mb"],

        "cuda_F_pre_reserved_mb":
            f_pre_cuda["cuda_reserved_mb"],

        "cuda_F_allocated_mb":
            f_cuda["cuda_allocated_mb"],

        "cuda_F_reserved_mb":
            f_cuda["cuda_reserved_mb"],

        "cuda_F_peak_allocated_mb":
            f_cuda["cuda_peak_allocated_mb"],

        "cuda_F_peak_reserved_mb":
            f_cuda["cuda_peak_reserved_mb"],

        "nvml_F_used_mb":
            f_cuda["nvml_used_mb"],

        # Diagnostic system-wide RAM
        "system_used_A_mb":
            a["system_ram_used_mb"],

        "system_used_E_mb":
            e["system_ram_used_mb"],

        "system_available_E_mb":
            e["system_ram_available_mb"],

        "memory_sample_count":
            f_ram["memory_sample_count"],
    }

    Path(result_path).write_text(
        json.dumps(row)
    )

    pynvml.nvmlShutdown()


def parent():
    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    script = Path(__file__).resolve()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        for precision in PRECISIONS:
            for profile in PROFILES:
                for repeat in range(1, REPEATS + 1):

                    result_path = (
                        tmp
                        / f"{precision}_{profile}_{repeat}.json"
                    )

                    print(
                        f"RUN {precision:4} "
                        f"{profile:6} "
                        f"R{repeat}"
                    )

                    subprocess.run(
                        [
                            sys.executable,
                            str(script),
                            "--worker",
                            "--precision",
                            precision,
                            "--profile",
                            profile,
                            "--repeat",
                            str(repeat),
                            "--result",
                            str(result_path),
                        ],
                        check=True,
                    )

                    row = json.loads(
                        result_path.read_text()
                    )

                    rows.append(row)

                    print(
                        f"  "
                        f"A {row['rss_A_before_model_mb']:.0f} MB | "
                        f"B {row['rss_B_cpu_model_mb']:.0f} MB | "
                        f"C {row['rss_C_device_model_mb']:.0f} MB | "
                        f"D {row['rss_D_input_mb']:.0f} MB | "
                        f"E {row['rss_E_warmup_mb']:.0f} MB | "
                        f"F {row['rss_F_inference_peak_mb']:.0f} MB | "
                        f"CUDA {row['cuda_F_allocated_mb']:.0f} MB | "
                        f"peak {row['cuda_F_peak_allocated_mb']:.0f} MB | "
                        f"NVML {row['nvml_F_used_mb']:.0f} MB | "
                        f"{row['throughput_img_s']:.2f} img/s"
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker",
        action="store_true",
    )

    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
    )

    parser.add_argument(
        "--profile",
        choices=PROFILES,
    )

    parser.add_argument(
        "--repeat",
        type=int,
    )

    parser.add_argument(
        "--result",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.worker:
        worker(
            args.precision,
            args.profile,
            args.repeat,
            args.result,
        )
    else:
        parent()
