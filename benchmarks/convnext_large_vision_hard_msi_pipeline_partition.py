import csv
import gc
import os
import statistics
import time

from windows_qos import set_process_qos

# Keep scheduling policy fixed.
set_process_qos("high")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
from torchvision.models import (
    convnext_large,
    ConvNeXt_Large_Weights,
)


THREADS = 4
RESOLUTION = 320
BATCH = 1

REPEATS = 3
ITERS = 8
WARMUP = 3

OUT = (
    "results/"
    "convnext_large_vision_hard_msi_"
    "pipeline_partition.csv"
)

CONFIGS = [
    ("all_cpu", None),

    ("after_stage1", 1),
    ("after_stage2", 3),
    ("after_stage3", 5),
    ("after_stage4", 7),

    ("all_gpu", None),
]


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available"
    )


torch.set_num_threads(THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def param_count(module):
    return sum(
        p.numel()
        for p in module.parameters()
    )


def output_tail(model, z):
    z = model.avgpool(z)
    return model.classifier(z)


def build_split(model, cut):
    children = list(
        model.features.children()
    )

    left = torch.nn.Sequential(
        *children[:cut + 1]
    )

    right = torch.nn.Sequential(
        *children[cut + 1:]
    )

    return left, right


def run_all_cpu(model, x):
    start = time.perf_counter()

    with torch.inference_mode():
        y = model(x)

    total = (
        time.perf_counter()
        - start
    ) * 1000.0

    return {
        "output": y,
        "cpu_stage_ms": total,
        "transfer_ms": 0.0,
        "gpu_stage_ms": 0.0,
        "total_ms": total,
    }


def run_all_gpu(model, x):
    torch.cuda.synchronize()

    start = time.perf_counter()

    x_gpu = x.to(
        "cuda",
        non_blocking=False,
    )

    torch.cuda.synchronize()

    transfer_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        y = model(x_gpu)

    torch.cuda.synchronize()

    gpu_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return {
        "output": y.cpu(),
        "cpu_stage_ms": 0.0,
        "transfer_ms": transfer_ms,
        "gpu_stage_ms": gpu_ms,
        "total_ms":
            transfer_ms + gpu_ms,
    }


def run_split(
    model,
    left,
    right,
    x,
):
    start = time.perf_counter()

    with torch.inference_mode():
        z = left(x)

    cpu_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    torch.cuda.synchronize()

    start = time.perf_counter()

    z_gpu = z.to(
        "cuda",
        non_blocking=False,
    )

    torch.cuda.synchronize()

    transfer_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        z_gpu = right(z_gpu)
        y = output_tail(
            model,
            z_gpu,
        )

    torch.cuda.synchronize()

    gpu_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return {
        "output": y.cpu(),
        "cpu_stage_ms": cpu_ms,
        "transfer_ms": transfer_ms,
        "gpu_stage_ms": gpu_ms,
        "total_ms":
            cpu_ms
            + transfer_ms
            + gpu_ms,
    }


rows = []

x = torch.randn(
    BATCH,
    3,
    RESOLUTION,
    RESOLUTION,
    dtype=torch.float32,
)


for config, cut in CONFIGS:

    print(
        f"\n=== {config} ==="
    )

    gc.collect()
    torch.cuda.empty_cache()

    model = convnext_large(
        weights=
        ConvNeXt_Large_Weights.DEFAULT
    )

    model.eval()

    # Reference before any modules
    # are moved to CUDA.
    with torch.inference_mode():
        reference = model(x)

    left = None
    right = None

    if config == "all_cpu":

        cpu_params = param_count(model)
        gpu_params = 0

        def once():
            return run_all_cpu(
                model,
                x,
            )

    elif config == "all_gpu":

        cpu_params = 0
        gpu_params = param_count(model)

        model = model.to("cuda")

        def once():
            return run_all_gpu(
                model,
                x,
            )

    else:

        left, right = build_split(
            model,
            cut,
        )

        cpu_params = param_count(left)

        gpu_params = (
            param_count(right)
            + param_count(model.avgpool)
            + param_count(model.classifier)
        )

        left = left.to("cpu")
        right = right.to("cuda")

        model.avgpool = (
            model.avgpool.to("cuda")
        )

        model.classifier = (
            model.classifier.to("cuda")
        )

        def once():
            return run_split(
                model,
                left,
                right,
                x,
            )

    # Warm-up.
    for _ in range(WARMUP):
        _ = once()

    # Correctness after actual
    # heterogeneous execution.
    check = once()

    diff = (
        check["output"]
        - reference
    ).abs()

    max_abs_diff = (
        diff.max().item()
    )

    mean_abs_diff = (
        diff.mean().item()
    )

    top1_match = int(
        check["output"]
        .argmax(dim=1)
        .item()
        ==
        reference
        .argmax(dim=1)
        .item()
    )

    cuda_alloc_mb = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    for repeat in range(
        1,
        REPEATS + 1,
    ):

        cpu_times = []
        transfer_times = []
        gpu_times = []
        total_times = []

        torch.cuda.reset_peak_memory_stats()

        for _ in range(ITERS):

            result = once()

            cpu_times.append(
                result["cpu_stage_ms"]
            )

            transfer_times.append(
                result["transfer_ms"]
            )

            gpu_times.append(
                result["gpu_stage_ms"]
            )

            total_times.append(
                result["total_ms"]
            )

        med_cpu = statistics.median(
            cpu_times
        )

        med_transfer = statistics.median(
            transfer_times
        )

        med_gpu = statistics.median(
            gpu_times
        )

        med_total = statistics.median(
            total_times
        )

        throughput = (
            BATCH
            / (med_total / 1000.0)
        )

        peak_cuda_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        row = {
            "config": config,
            "cut_after_feature": cut,
            "repeat": repeat,

            "threads": THREADS,
            "resolution": RESOLUTION,
            "batch": BATCH,

            "cpu_params_m":
                cpu_params / 1e6,

            "gpu_params_m":
                gpu_params / 1e6,

            "cpu_stage_ms":
                med_cpu,

            "transfer_ms":
                med_transfer,

            "gpu_stage_ms":
                med_gpu,

            "total_ms":
                med_total,

            "throughput_img_s":
                throughput,

            "cuda_allocated_mb":
                cuda_alloc_mb,

            "cuda_peak_allocated_mb":
                peak_cuda_mb,

            "max_abs_diff":
                max_abs_diff,

            "mean_abs_diff":
                mean_abs_diff,

            "top1_match_reference":
                top1_match,
        }

        rows.append(row)

        print(
            f"R{repeat} | "
            f"CPU {med_cpu:7.2f} ms | "
            f"H2D {med_transfer:6.2f} ms | "
            f"GPU {med_gpu:7.2f} ms | "
            f"TOTAL {med_total:7.2f} ms | "
            f"{throughput:6.2f} img/s | "
            f"VRAM {cuda_alloc_mb:7.1f} MB | "
            f"top1={top1_match}"
        )

    del model
    del left
    del right

    gc.collect()
    torch.cuda.empty_cache()


with open(
    OUT,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


print("\nMEDIANS")
print("-" * 110)

for config, _ in CONFIGS:

    group = [
        r for r in rows
        if r["config"] == config
    ]

    def med(key):
        return statistics.median(
            float(r[key])
            for r in group
        )

    print(
        f"{config:14} | "
        f"CPU {med('cpu_stage_ms'):7.2f} | "
        f"H2D {med('transfer_ms'):6.2f} | "
        f"GPU {med('gpu_stage_ms'):7.2f} | "
        f"TOTAL {med('total_ms'):7.2f} ms | "
        f"{med('throughput_img_s'):6.2f} img/s | "
        f"VRAM {med('cuda_allocated_mb'):7.1f} MB"
    )


print("\nSaved:", OUT)
