import argparse
import ctypes
import json
import os
import queue
import statistics
import threading
import time
from pathlib import Path


QOS_DEFAULT = 0x15


def set_default_qos():
    libc = ctypes.CDLL(
        "/usr/lib/libSystem.B.dylib",
        use_errno=True,
    )

    setter = libc.pthread_set_qos_class_self_np
    setter.argtypes = [
        ctypes.c_uint,
        ctypes.c_int,
    ]
    setter.restype = ctypes.c_int

    rc = setter(
        QOS_DEFAULT,
        0,
    )

    if rc != 0:
        raise RuntimeError(
            f"QoS set failed: {rc}"
        )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        required=True,
        choices=[
            "all_cpu",
            "all_mps",
            "stage1_seq",
            "stage1_pipe",
            "stage2_pipe",
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

    return p.parse_args()


args = parse_args()

set_default_qos()

os.environ["OMP_NUM_THREADS"] = str(
    args.threads
)

os.environ["VECLIB_MAXIMUM_THREADS"] = str(
    args.threads
)


import torch

from torchvision.models import (
    convnext_large,
    ConvNeXt_Large_Weights,
)


if not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS unavailable"
    )


torch.set_num_threads(
    args.threads
)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


RESOLUTION = 320
BATCH = 1

CUTS = {
    "stage1": 1,
    "stage2": 3,
    "stage3": 5,
}


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


is_pipeline = (
    args.config.endswith("_pipe")
)

is_split = (
    args.config.startswith("stage")
)

uses_mps = (
    args.config != "all_cpu"
)


left = None
right = None
avgpool = None
classifier = None


if args.config == "all_mps":

    model = model.to(
        "mps"
    )


elif is_split:

    stage_name = (
        args.config.split("_")[0]
    )

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
    ).to("mps")

    avgpool = model.avgpool.to(
        "mps"
    )

    classifier = (
        model.classifier.to(
            "mps"
        )
    )


def tail(z):
    z = right(z)
    z = avgpool(z)
    return classifier(z)


def percentile95(values):
    ordered = sorted(
        values
    )

    index = min(
        len(ordered) - 1,
        int(
            0.95
            * len(ordered)
        ),
    )

    return ordered[index]


def run_serial(seconds):

    latencies = []
    images = 0

    start = time.perf_counter()

    deadline = (
        start + seconds
    )

    with torch.inference_mode():

        while (
            time.perf_counter()
            < deadline
        ):

            t0 = time.perf_counter()

            if args.config == "all_cpu":

                _ = model(
                    x_cpu
                )


            elif args.config == "all_mps":

                x_mps = x_cpu.to(
                    "mps"
                )

                _ = model(
                    x_mps
                )

                torch.mps.synchronize()


            else:

                z = left(
                    x_cpu
                )

                z = z.to(
                    "mps"
                )

                _ = tail(
                    z
                )

                torch.mps.synchronize()


            t1 = time.perf_counter()

            latencies.append(
                (
                    t1 - t0
                ) * 1000.0
            )

            images += BATCH


    if uses_mps:
        torch.mps.synchronize()


    end = time.perf_counter()

    elapsed = (
        end - start
    )


    return {
        "images":
            images,

        "elapsed_sec":
            elapsed,

        "throughput_img_s":
            images / elapsed,

        "median_latency_ms":
            statistics.median(
                latencies
            ),

        "p95_latency_ms":
            percentile95(
                latencies
            ),
    }


def run_pipeline(seconds):

    q = queue.Queue(
        maxsize=2
    )

    latencies = []

    completed = 0

    start = time.perf_counter()

    deadline = (
        start + seconds
    )


    def producer():

        with torch.inference_mode():

            while (
                time.perf_counter()
                < deadline
            ):

                item_start = (
                    time.perf_counter()
                )

                z = left(
                    x_cpu
                )

                q.put(
                    (
                        item_start,
                        z,
                    )
                )

        q.put(None)


    def consumer():

        nonlocal completed

        with torch.inference_mode():

            while True:

                item = q.get()

                if item is None:
                    break

                item_start, z = item

                z_mps = z.to(
                    "mps"
                )

                _ = tail(
                    z_mps
                )

                torch.mps.synchronize()

                item_end = (
                    time.perf_counter()
                )

                latencies.append(
                    (
                        item_end
                        - item_start
                    )
                    * 1000.0
                )

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
            name="mps-stage",
        )
    )


    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    torch.mps.synchronize()

    end = time.perf_counter()

    elapsed = (
        end - start
    )


    return {
        "images":
            completed,

        "elapsed_sec":
            elapsed,

        "throughput_img_s":
            completed / elapsed,

        "median_latency_ms":
            statistics.median(
                latencies
            ),

        "p95_latency_ms":
            percentile95(
                latencies
            ),
    }


def run(seconds):

    if is_pipeline:
        return run_pipeline(
            seconds
        )

    return run_serial(
        seconds
    )


# Warm-up / compilation before
# formal power measurement.
_ = run(
    2.0
)


if uses_mps:

    torch.mps.synchronize()

    mps_current_mb = (
        torch.mps
        .current_allocated_memory()
        / 1024**2
    )

    mps_driver_mb = (
        torch.mps
        .driver_allocated_memory()
        / 1024**2
    )

else:

    mps_current_mb = 0.0
    mps_driver_mb = 0.0


ready = Path(
    args.ready_file
)

go = Path(
    args.go_file
)

ready.write_text(
    json.dumps({
        "pid": os.getpid(),
        "config": args.config,
    })
)


deadline = (
    time.monotonic()
    + 120
)

while not go.exists():

    if (
        time.monotonic()
        > deadline
    ):
        raise TimeoutError(
            "Timed out waiting "
            "for go-file"
        )

    time.sleep(
        0.05
    )


result = run(
    args.seconds
)


if uses_mps:

    torch.mps.synchronize()

    mps_current_mb = (
        torch.mps
        .current_allocated_memory()
        / 1024**2
    )

    mps_driver_mb = (
        torch.mps
        .driver_allocated_memory()
        / 1024**2
    )


result.update({
    "pid":
        os.getpid(),

    "config":
        args.config,

    "threads":
        args.threads,

    "resolution":
        RESOLUTION,

    "batch":
        BATCH,

    "mps_current_alloc_mb":
        mps_current_mb,

    "mps_driver_alloc_mb":
        mps_driver_mb,
})


Path(
    args.result
).write_text(
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
