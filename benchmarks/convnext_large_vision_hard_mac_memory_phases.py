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
    "results/convnext_large_vision_hard_mac_memory_phases.csv"
)

MB = 1024 ** 2


def sync():
    torch.mps.synchronize()


def stable_host_snapshot(samples=7, interval_sec=0.10):
    """
    Median snapshot to reduce short-lived macOS RSS noise.
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


def mps_snapshot():
    current = (
        torch.mps.current_allocated_memory() / MB
    )

    driver = None

    if hasattr(torch.mps, "driver_allocated_memory"):
        driver = (
            torch.mps.driver_allocated_memory() / MB
        )

    return {
        "mps_current_mb": current,
        "mps_driver_mb": driver,
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
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    dtype = PRECISIONS[precision]
    res, batch = PROFILES[profile]

    device = torch.device("mps")
    weights = ConvNeXt_Large_Weights.DEFAULT

    # --------------------------------------------------------
    # A. Before model load
    # --------------------------------------------------------

    a = stable_host_snapshot()

    # --------------------------------------------------------
    # B. CPU model loaded
    # --------------------------------------------------------

    model = convnext_large(weights=weights)
    model.eval()

    gc.collect()

    b = stable_host_snapshot()

    # --------------------------------------------------------
    # C. Model moved to accelerator
    # --------------------------------------------------------

    model = model.to(
        device=device,
        dtype=dtype,
    )

    sync()
    gc.collect()

    c = stable_host_snapshot()
    c_mps = mps_snapshot()

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
    d_mps = mps_snapshot()

    # --------------------------------------------------------
    # E. Warm-up completed
    # --------------------------------------------------------

    with torch.inference_mode():
        for _ in range(5):
            _ = model(x)

    sync()

    e = stable_host_snapshot()
    e_mps = mps_snapshot()

    # --------------------------------------------------------
    # F. Sustained inference peak
    # --------------------------------------------------------

    sampler = MemorySampler(
        interval_sec=0.10,
        include_children=False,
    )

    torch.mps.empty_cache()
    sync()

    # Do not clear model/input allocations; only unused cache.
    f_pre_mps = mps_snapshot()

    sampler.start()

    throughput = run_for_duration(
        model,
        x,
        batch,
        INFERENCE_SEC,
    )

    f_ram = sampler.stop()
    f_mps = mps_snapshot()

    row = {
        "precision": precision,
        "profile": profile,
        "resolution": res,
        "batch": batch,
        "repeat": repeat,

        "throughput_img_s": throughput,

        # A
        "rss_A_before_model_mb":
            a["process_rss_mb"],

        # B
        "rss_B_cpu_model_mb":
            b["process_rss_mb"],

        # C
        "rss_C_device_model_mb":
            c["process_rss_mb"],

        # D
        "rss_D_input_mb":
            d["process_rss_mb"],

        # E
        "rss_E_warmup_mb":
            e["process_rss_mb"],

        # F
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

        # MPS allocations
        "mps_C_current_mb":
            c_mps["mps_current_mb"],

        "mps_C_driver_mb":
            c_mps["mps_driver_mb"],

        "mps_D_current_mb":
            d_mps["mps_current_mb"],

        "mps_D_driver_mb":
            d_mps["mps_driver_mb"],

        "mps_E_current_mb":
            e_mps["mps_current_mb"],

        "mps_E_driver_mb":
            e_mps["mps_driver_mb"],

        "mps_F_pre_current_mb":
            f_pre_mps["mps_current_mb"],

        "mps_F_pre_driver_mb":
            f_pre_mps["mps_driver_mb"],

        "mps_F_current_mb":
            f_mps["mps_current_mb"],

        "mps_F_driver_mb":
            f_mps["mps_driver_mb"],

        # System-wide values are diagnostic only.
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
                        f"MPS {row['mps_F_current_mb']:.0f} MB | "
                        f"{row['throughput_img_s']:.2f} img/s"
                    )

                    # Save after every run.
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
