import argparse
import json
import os
import time
from pathlib import Path

from windows_qos import (
    set_process_qos,
    get_process_qos_state,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--qos",
        required=True,
        choices=["system", "eco", "high"],
    )

    p.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    p.add_argument(
        "--seconds",
        type=float,
        default=10.0,
    )

    p.add_argument(
        "--repeat",
        type=int,
        required=True,
    )

    p.add_argument(
        "--result",
        required=True,
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Set process QoS BEFORE importing torch.
    set_process_qos(args.qos)
    qos_state = get_process_qos_state()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)

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

    model = convnext_large(
        weights=ConvNeXt_Large_Weights.DEFAULT
    )

    model.eval()

    x = torch.randn(
        1,
        3,
        320,
        320,
    )

    with torch.inference_mode():
        for _ in range(3):
            _ = model(x)

    images = 0

    start = time.perf_counter()

    with torch.inference_mode():
        while True:
            _ = model(x)
            images += 1

            if (
                time.perf_counter() - start
                >= args.seconds
            ):
                break

    elapsed = time.perf_counter() - start

    result = {
        "qos": args.qos,
        "repeat": args.repeat,
        "threads": args.threads,
        "duration_sec": elapsed,
        "images": images,

        "throughput_img_s":
            images / elapsed,

        "latency_ms_img":
            elapsed / images * 1000.0,

        "qos_controlled":
            qos_state[
                "execution_speed_controlled"
            ],

        "qos_throttled":
            qos_state[
                "execution_speed_throttled"
            ],
    }

    Path(args.result).write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        f"{args.qos:6} "
        f"R{args.repeat} | "
        f"{result['throughput_img_s']:.3f} img/s | "
        f"{result['latency_ms_img']:.1f} ms/img | "
        f"controlled={result['qos_controlled']} | "
        f"throttled={result['qos_throttled']}"
    )


if __name__ == "__main__":
    main()
