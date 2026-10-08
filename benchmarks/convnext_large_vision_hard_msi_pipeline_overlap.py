import csv
import gc
import os
import queue
import statistics
import threading
import time

from windows_qos import set_process_qos

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

MICROBATCH_SIZE = 1
NUM_MICROBATCHES = 12

REPEATS = 3
WARMUP_MICROBATCHES = 3

OUT = (
    "results/"
    "convnext_large_vision_hard_msi_"
    "pipeline_overlap.csv"
)

CUTS = [
    ("after_stage1", 1),
    ("after_stage2", 3),
    ("after_stage3", 5),
]


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")


torch.set_num_threads(THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def build_split(model, cut):
    children = list(
        model.features.children()
    )

    left = torch.nn.Sequential(
        *children[:cut + 1]
    ).to("cpu")

    right = torch.nn.Sequential(
        *children[cut + 1:]
    ).to("cuda")

    avgpool = model.avgpool.to("cuda")
    classifier = model.classifier.to("cuda")

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
    z_gpu,
):
    z_gpu = right(z_gpu)
    z_gpu = avgpool(z_gpu)
    return classifier(z_gpu)


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

            torch.cuda.synchronize()

            h2d_start = time.perf_counter()

            z_gpu = z.to(
                "cuda",
                non_blocking=False,
            )

            torch.cuda.synchronize()

            h2d_end = time.perf_counter()

            gpu_start = time.perf_counter()

            y = gpu_tail(
                right,
                avgpool,
                classifier,
                z_gpu,
            )

            torch.cuda.synchronize()

            gpu_end = time.perf_counter()

            outputs.append(
                y.detach()
            )

            timeline.append({
                "index": i,
                "cpu_start": cpu_start,
                "cpu_end": cpu_end,
                "h2d_start": h2d_start,
                "h2d_end": h2d_end,
                "gpu_start": gpu_start,
                "gpu_end": gpu_end,
            })

    torch.cuda.synchronize()

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
    q = queue.Queue(maxsize=2)

    timeline = {
        i: {}
        for i in range(len(inputs))
    }

    outputs = [None] * len(inputs)

    wall_start = time.perf_counter()

    def producer():

        with torch.inference_mode():

            for i, x in enumerate(inputs):

                cpu_start = time.perf_counter()

                z = left(x)

                cpu_end = time.perf_counter()

                timeline[i][
                    "cpu_start"
                ] = cpu_start

                timeline[i][
                    "cpu_end"
                ] = cpu_end

                q.put((i, z))

        q.put(None)

    def consumer():

        with torch.inference_mode():

            while True:

                item = q.get()

                if item is None:
                    break

                i, z = item

                torch.cuda.synchronize()

                h2d_start = time.perf_counter()

                z_gpu = z.to(
                    "cuda",
                    non_blocking=False,
                )

                torch.cuda.synchronize()

                h2d_end = time.perf_counter()

                gpu_start = time.perf_counter()

                y = gpu_tail(
                    right,
                    avgpool,
                    classifier,
                    z_gpu,
                )

                torch.cuda.synchronize()

                gpu_end = time.perf_counter()

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

                outputs[i] = y.detach()

    producer_thread = threading.Thread(
        target=producer,
        name="cpu-stage",
    )

    consumer_thread = threading.Thread(
        target=consumer,
        name="gpu-stage",
    )

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    torch.cuda.synchronize()

    wall_end = time.perf_counter()

    ordered_timeline = [
        {
            "index": i,
            **timeline[i],
        }
        for i in range(len(inputs))
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
    wall_s = (
        wall_end
        - wall_start
    )

    samples = (
        len(timeline)
        * MICROBATCH_SIZE
    )

    throughput = (
        samples / wall_s
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

    first_output_ms = (
        min(
            t["gpu_end"]
            for t in timeline
        )
        - wall_start
    ) * 1000

    return {
        "wall_ms":
            wall_s * 1000,

        "throughput_img_s":
            throughput,

        "effective_ms_per_img":
            1000 / throughput,

        "median_cpu_stage_ms":
            statistics.median(cpu_ms),

        "median_h2d_ms":
            statistics.median(h2d_ms),

        "median_gpu_stage_ms":
            statistics.median(gpu_ms),

        "median_e2e_latency_ms":
            statistics.median(e2e_ms),

        "first_output_ms":
            first_output_ms,
    }


rows = []


for cut_name, cut in CUTS:

    print(
        f"\n{'=' * 90}\n"
        f"{cut_name}\n"
        f"{'=' * 90}"
    )

    gc.collect()
    torch.cuda.empty_cache()

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

    warm_inputs = make_inputs(
        WARMUP_MICROBATCHES
    )

    _ = run_sequential(
        left,
        right,
        avgpool,
        classifier,
        warm_inputs,
    )

    _ = run_pipeline(
        left,
        right,
        avgpool,
        classifier,
        warm_inputs,
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

            row = {
                "cut":
                    cut_name,

                "cut_after_feature":
                    cut,

                "mode":
                    mode,

                "repeat":
                    repeat,

                "threads":
                    THREADS,

                "resolution":
                    RESOLUTION,

                "microbatch_size":
                    MICROBATCH_SIZE,

                "num_microbatches":
                    NUM_MICROBATCHES,

                **summary,

                "pipeline_speedup":
                    speedup,

                "max_abs_diff_seq_vs_pipe":
                    max_abs_diff,
            }

            rows.append(row)

        print(
            f"R{repeat} | "
            f"SEQ "
            f"{seq_summary['throughput_img_s']:6.2f} img/s | "
            f"{seq_summary['median_e2e_latency_ms']:7.2f} ms e2e"
        )

        print(
            f"   -> PIPE "
            f"{pipe_summary['throughput_img_s']:6.2f} img/s | "
            f"{pipe_summary['median_e2e_latency_ms']:7.2f} ms e2e | "
            f"speedup {speedup:5.2f}x | "
            f"diff {max_abs_diff:.3e}"
        )

    del model
    del left
    del right
    del avgpool
    del classifier

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


print(
    "\n\nMEDIAN SUMMARY"
)

print("-" * 125)

for cut_name, _ in CUTS:

    for mode in [
        "sequential",
        "pipeline",
    ]:

        group = [
            r
            for r in rows
            if (
                r["cut"] == cut_name
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
            f"{cut_name:14} "
            f"{mode:10} | "
            f"thr "
            f"{med('throughput_img_s'):6.2f} | "
            f"eff "
            f"{med('effective_ms_per_img'):7.2f} ms/img | "
            f"e2e "
            f"{med('median_e2e_latency_ms'):7.2f} ms | "
            f"CPU "
            f"{med('median_cpu_stage_ms'):7.2f} | "
            f"H2D "
            f"{med('median_h2d_ms'):6.2f} | "
            f"GPU "
            f"{med('median_gpu_stage_ms'):7.2f}"
        )


print(
    "\nSaved:",
    OUT,
)
