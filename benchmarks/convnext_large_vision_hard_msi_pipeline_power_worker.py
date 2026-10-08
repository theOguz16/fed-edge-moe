import argparse
import json
import os
import queue
import statistics
import threading
import time
from pathlib import Path

from windows_qos import set_process_qos


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        required=True,
        choices=[
            "all_cpu",
            "all_gpu",
            "stage1_seq",
            "stage1_pipe",
            "stage2_seq",
            "stage2_pipe",
            "stage3_seq",
            "stage3_pipe",
        ],
    )

    p.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    p.add_argument(
        "--seconds",
        type=float,
        default=20.0,
    )

    p.add_argument(
        "--result",
        required=True,
    )

    p.add_argument(
        "--ready-file",
        required=True,
    )

    p.add_argument(
        "--go-file",
        required=True,
    )

    p.add_argument(
        "--started-file",
        required=True,
    )

    return p.parse_args()


args = parse_args()

# Keep scheduling policy fixed.
set_process_qos("high")

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


RESOLUTION = 320
BATCH = 1

CUTS = {
    "stage1": 1,
    "stage2": 3,
    "stage3": 5,
}


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available"
    )


torch.set_num_threads(
    args.threads
)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


x_cpu = torch.randn(
    BATCH,
    3,
    RESOLUTION,
    RESOLUTION,
    dtype=torch.float32,
)


model = convnext_large(
    weights=
    ConvNeXt_Large_Weights.DEFAULT
)

model.eval()


uses_gpu = (
    args.config != "all_cpu"
)

is_pipeline = (
    args.config.endswith("_pipe")
)

is_split = (
    args.config.startswith("stage")
)


left = None
right = None
avgpool = None
classifier = None


if args.config == "all_gpu":

    model = model.to("cuda")


elif is_split:

    stage_name = args.config.split("_")[0]

    cut = CUTS[
        stage_name
    ]

    children = list(
        model.features.children()
    )

    left = torch.nn.Sequential(
        *children[:cut + 1]
    ).to("cpu")

    right = torch.nn.Sequential(
        *children[cut + 1:]
    ).to("cuda")

    avgpool = model.avgpool.to(
        "cuda"
    )

    classifier = model.classifier.to(
        "cuda"
    )


def gpu_tail(z_gpu):
    z_gpu = right(z_gpu)
    z_gpu = avgpool(z_gpu)
    return classifier(z_gpu)


def one_serial():
    start = time.perf_counter()

    with torch.inference_mode():

        if args.config == "all_cpu":

            _ = model(x_cpu)


        elif args.config == "all_gpu":

            x_gpu = x_cpu.to(
                "cuda",
                non_blocking=False,
            )

            _ = model(x_gpu)

            torch.cuda.synchronize()


        else:

            z = left(x_cpu)

            z_gpu = z.to(
                "cuda",
                non_blocking=False,
            )

            _ = gpu_tail(z_gpu)

            torch.cuda.synchronize()

    end = time.perf_counter()

    return (
        end - start
    ) * 1000.0


def run_serial(seconds):

    latencies = []

    images = 0

    wall_start = time.perf_counter()

    deadline = (
        wall_start
        + seconds
    )

    while (
        time.perf_counter()
        < deadline
    ):

        latency = one_serial()

        latencies.append(
            latency
        )

        images += BATCH

    if uses_gpu:
        torch.cuda.synchronize()

    wall_end = time.perf_counter()

    elapsed = (
        wall_end
        - wall_start
    )

    return {
        "images": images,
        "elapsed_sec": elapsed,
        "throughput_img_s":
            images / elapsed,

        "median_latency_ms":
            statistics.median(
                latencies
            ),

        "p95_latency_ms":
            sorted(latencies)[
                min(
                    len(latencies) - 1,
                    int(
                        0.95
                        * len(latencies)
                    ),
                )
            ],
    }


def run_pipeline(seconds):

    q = queue.Queue(
        maxsize=2
    )

    latencies = []

    completed = 0

    completed_lock = (
        threading.Lock()
    )

    wall_start = time.perf_counter()

    deadline = (
        wall_start
        + seconds
    )


    def producer():

        index = 0

        with torch.inference_mode():

            while (
                time.perf_counter()
                < deadline
            ):

                start = (
                    time.perf_counter()
                )

                z = left(
                    x_cpu
                )

                q.put(
                    (
                        index,
                        start,
                        z,
                    )
                )

                index += 1

        q.put(None)


    def consumer():

        nonlocal completed

        with torch.inference_mode():

            while True:

                item = q.get()

                if item is None:
                    break

                (
                    _,
                    start,
                    z,
                ) = item

                z_gpu = z.to(
                    "cuda",
                    non_blocking=False,
                )

                _ = gpu_tail(
                    z_gpu
                )

                torch.cuda.synchronize()

                end = (
                    time.perf_counter()
                )

                latencies.append(
                    (
                        end - start
                    )
                    * 1000.0
                )

                with completed_lock:
                    completed += BATCH


    producer_thread = (
        threading.Thread(
            target=producer,
            name="cpu-stage",
        )
    )

    consumer_thread = (
        threading.Thread(
            target=consumer,
            name="gpu-stage",
        )
    )


    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    torch.cuda.synchronize()

    wall_end = time.perf_counter()

    elapsed = (
        wall_end
        - wall_start
    )

    return {
        "images": completed,

        "elapsed_sec":
            elapsed,

        "throughput_img_s":
            completed / elapsed,

        "median_latency_ms":
            statistics.median(
                latencies
            ),

        "p95_latency_ms":
            sorted(latencies)[
                min(
                    len(latencies) - 1,
                    int(
                        0.95
                        * len(latencies)
                    ),
                )
            ],
    }


def run_measurement(seconds):

    if is_pipeline:
        return run_pipeline(
            seconds
        )

    return run_serial(
        seconds
    )


# Warm-up before measurement.
_ = run_measurement(
    2.0
)


if uses_gpu:

    torch.cuda.synchronize()

    cuda_allocated_mb = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    torch.cuda.reset_peak_memory_stats()

else:

    cuda_allocated_mb = 0.0


ready_file = Path(
    args.ready_file
)

go_file = Path(
    args.go_file
)

started_file = Path(
    args.started_file
)

result_file = Path(
    args.result
)


ready_file.touch()


while not go_file.exists():
    time.sleep(0.02)


started_file.touch()


result = run_measurement(
    args.seconds
)


if uses_gpu:

    torch.cuda.synchronize()

    cuda_peak_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

else:

    cuda_peak_mb = 0.0


result.update({
    "config":
        args.config,

    "threads":
        args.threads,

    "resolution":
        RESOLUTION,

    "batch":
        BATCH,

    "cuda_allocated_mb":
        cuda_allocated_mb,

    "cuda_peak_mb":
        cuda_peak_mb,
})


result_file.write_text(
    json.dumps(
        result,
        indent=2,
    )
)


print(
    json.dumps(
        result,
        indent=2,
    )
)
