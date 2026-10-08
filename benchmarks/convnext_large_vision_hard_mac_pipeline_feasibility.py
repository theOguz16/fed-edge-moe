import csv
import gc
import os
import queue
import statistics
import threading
import time

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import torch
from torchvision.models import (
    convnext_large,
    ConvNeXt_Large_Weights,
)


THREADS = 4
RESOLUTION = 320
MICROBATCH_SIZE = 1

NUM_MICROBATCHES = 12
REPEATS = 3
WARMUP_MICROBATCHES = 3

OUT = (
    "results/"
    "convnext_large_vision_hard_mac_"
    "pipeline_feasibility.csv"
)

CUTS = [
    ("after_stage1", 1),
    ("after_stage2", 3),
    ("after_stage3", 5),
]


if not torch.backends.mps.is_available():
    raise RuntimeError("MPS unavailable")


torch.set_num_threads(THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def cleanup():
    gc.collect()

    try:
        torch.mps.empty_cache()
    except Exception:
        pass


def mps_memory():
    try:
        current = (
            torch.mps.current_allocated_memory()
            / 1024**2
        )

        driver = (
            torch.mps.driver_allocated_memory()
            / 1024**2
        )

        return current, driver

    except Exception:
        return 0.0, 0.0


def make_inputs(n):
    return [
        torch.randn(
            MICROBATCH_SIZE,
            3,
            RESOLUTION,
            RESOLUTION,
            dtype=torch.float32,
        )
        for _ in range(n)
    ]


def build_split(model, cut):
    children = list(
        model.features.children()
    )

    left = torch.nn.Sequential(
        *children[:cut + 1]
    ).to("cpu")

    right = torch.nn.Sequential(
        *children[cut + 1:]
    ).to("mps")

    avgpool = model.avgpool.to("mps")
    classifier = model.classifier.to("mps")

    return (
        left,
        right,
        avgpool,
        classifier,
    )


def gpu_tail(
    right,
    avgpool,
    classifier,
    z,
):
    z = right(z)
    z = avgpool(z)
    return classifier(z)


def run_all_cpu(
    model,
    inputs,
):
    latencies = []
    outputs = []

    wall_start = time.perf_counter()

    with torch.inference_mode():

        for x in inputs:

            start = time.perf_counter()

            y = model(x)

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )

            outputs.append(
                y.detach().cpu()
            )

    wall_end = time.perf_counter()

    return {
        "wall_ms":
            (wall_end - wall_start)
            * 1000,

        "throughput_img_s":
            len(inputs)
            * MICROBATCH_SIZE
            / (wall_end - wall_start),

        "median_e2e_latency_ms":
            statistics.median(
                latencies
            ),

        "outputs":
            outputs,
    }


def run_all_mps(
    model,
    inputs,
):
    latencies = []
    outputs = []

    torch.mps.synchronize()

    wall_start = time.perf_counter()

    with torch.inference_mode():

        for x in inputs:

            start = time.perf_counter()

            x_mps = x.to(
                "mps"
            )

            y = model(
                x_mps
            )

            torch.mps.synchronize()

            end = time.perf_counter()

            latencies.append(
                (end - start) * 1000
            )

            outputs.append(
                y.detach().cpu()
            )

    torch.mps.synchronize()

    wall_end = time.perf_counter()

    return {
        "wall_ms":
            (wall_end - wall_start)
            * 1000,

        "throughput_img_s":
            len(inputs)
            * MICROBATCH_SIZE
            / (wall_end - wall_start),

        "median_e2e_latency_ms":
            statistics.median(
                latencies
            ),

        "outputs":
            outputs,
    }


def run_sequential(
    left,
    right,
    avgpool,
    classifier,
    inputs,
):
    timeline = []
    outputs = []

    wall_start = time.perf_counter()

    with torch.inference_mode():

        for i, x in enumerate(inputs):

            cpu_start = time.perf_counter()

            z = left(x)

            cpu_end = time.perf_counter()

            h2d_start = time.perf_counter()

            z_mps = z.to(
                "mps"
            )

            torch.mps.synchronize()

            h2d_end = time.perf_counter()

            gpu_start = time.perf_counter()

            y = gpu_tail(
                right,
                avgpool,
                classifier,
                z_mps,
            )

            torch.mps.synchronize()

            gpu_end = time.perf_counter()

            outputs.append(
                y.detach().cpu()
            )

            timeline.append({
                "index": i,

                "cpu_start":
                    cpu_start,

                "cpu_end":
                    cpu_end,

                "h2d_start":
                    h2d_start,

                "h2d_end":
                    h2d_end,

                "gpu_start":
                    gpu_start,

                "gpu_end":
                    gpu_end,
            })

    torch.mps.synchronize()

    wall_end = time.perf_counter()

    return (
        wall_start,
        wall_end,
        timeline,
        outputs,
    )


def run_pipeline(
    left,
    right,
    avgpool,
    classifier,
    inputs,
):
    q = queue.Queue(
        maxsize=2
    )

    timeline = {
        i: {}
        for i in range(
            len(inputs)
        )
    }

    outputs = [
        None
        for _ in inputs
    ]

    wall_start = time.perf_counter()


    def producer():

        with torch.inference_mode():

            for i, x in enumerate(
                inputs
            ):

                cpu_start = (
                    time.perf_counter()
                )

                z = left(x)

                cpu_end = (
                    time.perf_counter()
                )

                timeline[i][
                    "cpu_start"
                ] = cpu_start

                timeline[i][
                    "cpu_end"
                ] = cpu_end

                q.put(
                    (i, z)
                )

        q.put(None)


    def consumer():

        with torch.inference_mode():

            while True:

                item = q.get()

                if item is None:
                    break

                i, z = item

                h2d_start = (
                    time.perf_counter()
                )

                z_mps = z.to(
                    "mps"
                )

                torch.mps.synchronize()

                h2d_end = (
                    time.perf_counter()
                )

                gpu_start = (
                    time.perf_counter()
                )

                y = gpu_tail(
                    right,
                    avgpool,
                    classifier,
                    z_mps,
                )

                torch.mps.synchronize()

                gpu_end = (
                    time.perf_counter()
                )

                timeline[i][
                    "h2d_start"
                ] = h2d_start

                timeline[i][
                    "h2d_end"
                ] = h2d_end

                timeline[i][
                    "gpu_start"
                ] = gpu_start

                timeline[i][
                    "gpu_end"
                ] = gpu_end

                outputs[i] = (
                    y.detach().cpu()
                )


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

    wall_end = time.perf_counter()

    ordered_timeline = [
        {
            "index": i,
            **timeline[i],
        }
        for i in range(
            len(inputs)
        )
    ]

    return (
        wall_start,
        wall_end,
        ordered_timeline,
        outputs,
    )


def summarize(
    wall_start,
    wall_end,
    timeline,
):
    elapsed = (
        wall_end
        - wall_start
    )

    samples = (
        len(timeline)
        * MICROBATCH_SIZE
    )

    throughput = (
        samples / elapsed
    )

    cpu_ms = [
        (
            t["cpu_end"]
            - t["cpu_start"]
        ) * 1000
        for t in timeline
    ]

    h2d_ms = [
        (
            t["h2d_end"]
            - t["h2d_start"]
        ) * 1000
        for t in timeline
    ]

    gpu_ms = [
        (
            t["gpu_end"]
            - t["gpu_start"]
        ) * 1000
        for t in timeline
    ]

    e2e_ms = [
        (
            t["gpu_end"]
            - t["cpu_start"]
        ) * 1000
        for t in timeline
    ]

    return {
        "wall_ms":
            elapsed * 1000,

        "throughput_img_s":
            throughput,

        "effective_ms_per_img":
            1000.0
            / throughput,

        "median_cpu_stage_ms":
            statistics.median(
                cpu_ms
            ),

        "median_h2d_ms":
            statistics.median(
                h2d_ms
            ),

        "median_gpu_stage_ms":
            statistics.median(
                gpu_ms
            ),

        "median_e2e_latency_ms":
            statistics.median(
                e2e_ms
            ),
    }


rows = []


#
# Baseline: all CPU
#

print(
    "\n"
    + "=" * 90
)

print("all_cpu")

print(
    "=" * 90
)

cleanup()

model = convnext_large(
    weights=
    ConvNeXt_Large_Weights.DEFAULT
)

model.eval()

warm = make_inputs(
    WARMUP_MICROBATCHES
)

_ = run_all_cpu(
    model,
    warm,
)

for repeat in range(
    1,
    REPEATS + 1,
):

    inputs = make_inputs(
        NUM_MICROBATCHES
    )

    result = run_all_cpu(
        model,
        inputs,
    )

    row = {
        "config":
            "all_cpu",

        "mode":
            "baseline",

        "repeat":
            repeat,

        "throughput_img_s":
            result[
                "throughput_img_s"
            ],

        "effective_ms_per_img":
            1000.0
            / result[
                "throughput_img_s"
            ],

        "median_e2e_latency_ms":
            result[
                "median_e2e_latency_ms"
            ],

        "median_cpu_stage_ms":
            result[
                "median_e2e_latency_ms"
            ],

        "median_h2d_ms":
            0.0,

        "median_gpu_stage_ms":
            0.0,

        "mps_current_alloc_mb":
            0.0,

        "mps_driver_alloc_mb":
            0.0,

        "max_abs_diff_seq_vs_pipe":
            0.0,
    }

    rows.append(
        row
    )

    print(
        f"R{repeat} | "
        f"{row['throughput_img_s']:6.2f} img/s | "
        f"{row['median_e2e_latency_ms']:7.2f} ms"
    )


del model

cleanup()


#
# Baseline: all MPS
#

print(
    "\n"
    + "=" * 90
)

print("all_mps")

print(
    "=" * 90
)

model = convnext_large(
    weights=
    ConvNeXt_Large_Weights.DEFAULT
)

model.eval()

model = model.to(
    "mps"
)

warm = make_inputs(
    WARMUP_MICROBATCHES
)

_ = run_all_mps(
    model,
    warm,
)

for repeat in range(
    1,
    REPEATS + 1,
):

    inputs = make_inputs(
        NUM_MICROBATCHES
    )

    result = run_all_mps(
        model,
        inputs,
    )

    current_mb, driver_mb = (
        mps_memory()
    )

    row = {
        "config":
            "all_mps",

        "mode":
            "baseline",

        "repeat":
            repeat,

        "throughput_img_s":
            result[
                "throughput_img_s"
            ],

        "effective_ms_per_img":
            1000.0
            / result[
                "throughput_img_s"
            ],

        "median_e2e_latency_ms":
            result[
                "median_e2e_latency_ms"
            ],

        "median_cpu_stage_ms":
            0.0,

        "median_h2d_ms":
            0.0,

        "median_gpu_stage_ms":
            result[
                "median_e2e_latency_ms"
            ],

        "mps_current_alloc_mb":
            current_mb,

        "mps_driver_alloc_mb":
            driver_mb,

        "max_abs_diff_seq_vs_pipe":
            0.0,
    }

    rows.append(
        row
    )

    print(
        f"R{repeat} | "
        f"{row['throughput_img_s']:6.2f} img/s | "
        f"{row['median_e2e_latency_ms']:7.2f} ms | "
        f"MPS current "
        f"{current_mb:7.1f} MB | "
        f"driver "
        f"{driver_mb:7.1f} MB"
    )


del model

cleanup()


#
# Split configurations
#

for cut_name, cut in CUTS:

    print(
        "\n"
        + "=" * 90
    )

    print(
        cut_name
    )

    print(
        "=" * 90
    )

    cleanup()

    model = convnext_large(
        weights=
        ConvNeXt_Large_Weights.DEFAULT
    )

    model.eval()

    (
        left,
        right,
        avgpool,
        classifier,
    ) = build_split(
        model,
        cut,
    )

    warm = make_inputs(
        WARMUP_MICROBATCHES
    )

    _ = run_sequential(
        left,
        right,
        avgpool,
        classifier,
        warm,
    )

    _ = run_pipeline(
        left,
        right,
        avgpool,
        classifier,
        warm,
    )


    for repeat in range(
        1,
        REPEATS + 1,
    ):

        inputs = make_inputs(
            NUM_MICROBATCHES
        )

        seq = run_sequential(
            left,
            right,
            avgpool,
            classifier,
            inputs,
        )

        pipe = run_pipeline(
            left,
            right,
            avgpool,
            classifier,
            inputs,
        )

        seq_summary = summarize(
            seq[0],
            seq[1],
            seq[2],
        )

        pipe_summary = summarize(
            pipe[0],
            pipe[1],
            pipe[2],
        )


        max_abs_diff = 0.0

        for a, b in zip(
            seq[3],
            pipe[3],
        ):

            diff = (
                a - b
            ).abs().max().item()

            max_abs_diff = max(
                max_abs_diff,
                diff,
            )


        current_mb, driver_mb = (
            mps_memory()
        )


        speedup = (
            pipe_summary[
                "throughput_img_s"
            ]
            /
            seq_summary[
                "throughput_img_s"
            ]
        )


        for mode, summary in [
            (
                "sequential",
                seq_summary,
            ),
            (
                "pipeline",
                pipe_summary,
            ),
        ]:

            rows.append({
                "config":
                    cut_name,

                "mode":
                    mode,

                "repeat":
                    repeat,

                **summary,

                "mps_current_alloc_mb":
                    current_mb,

                "mps_driver_alloc_mb":
                    driver_mb,

                "max_abs_diff_seq_vs_pipe":
                    max_abs_diff,
            })


        print(
            f"R{repeat} | "
            f"SEQ "
            f"{seq_summary['throughput_img_s']:6.2f} img/s | "
            f"{seq_summary['median_e2e_latency_ms']:7.2f} ms"
        )

        print(
            f"   -> PIPE "
            f"{pipe_summary['throughput_img_s']:6.2f} img/s | "
            f"{pipe_summary['median_e2e_latency_ms']:7.2f} ms | "
            f"{speedup:5.2f}x | "
            f"diff {max_abs_diff:.3e} | "
            f"MPS {current_mb:7.1f} MB"
        )


    del model
    del left
    del right
    del avgpool
    del classifier

    cleanup()


with open(
    OUT,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            dict.fromkeys(
                key
                for row in rows
                for key in row.keys()
            )
        ),
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


print(
    "\nMEDIAN SUMMARY"
)

print(
    "-" * 140
)


groups = [
    ("all_cpu", "baseline"),
    ("all_mps", "baseline"),

    ("after_stage1", "sequential"),
    ("after_stage1", "pipeline"),

    ("after_stage2", "sequential"),
    ("after_stage2", "pipeline"),

    ("after_stage3", "sequential"),
    ("after_stage3", "pipeline"),
]


for config, mode in groups:

    group = [
        r
        for r in rows
        if (
            r["config"] == config
            and
            r["mode"] == mode
        )
    ]


    def med(key):
        return statistics.median(
            float(r[key])
            for r in group
        )


    print(
        f"{config:14} "
        f"{mode:10} | "
        f"thr "
        f"{med('throughput_img_s'):6.2f} | "
        f"eff "
        f"{med('effective_ms_per_img'):7.2f} ms/img | "
        f"e2e "
        f"{med('median_e2e_latency_ms'):7.2f} ms | "
        f"CPU "
        f"{med('median_cpu_stage_ms'):7.2f} | "
        f"move "
        f"{med('median_h2d_ms'):6.2f} | "
        f"MPS "
        f"{med('median_gpu_stage_ms'):7.2f} | "
        f"alloc "
        f"{med('mps_current_alloc_mb'):7.1f} MB | "
        f"driver "
        f"{med('mps_driver_alloc_mb'):7.1f} MB"
    )


print(
    "\nSaved:",
    OUT,
)
