import argparse
import json
import os
import time
from pathlib import Path

from windows_affinity import (
    set_affinity,
    get_affinity,
)

from windows_qos import (
    set_process_qos,
    get_process_qos_state,
)


PLACEMENTS = {
    "unrestricted": None,
    "four_physical": [0, 2, 4, 6],
    "two_physical_smt": [0, 1, 2, 3],
}


def wait_for_file(path, timeout=120):
    deadline = time.monotonic() + timeout

    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for {path}"
            )

        time.sleep(0.05)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--placement",
        required=True,
        choices=PLACEMENTS,
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

    p.add_argument("--result", required=True)
    p.add_argument("--ready-file", required=True)
    p.add_argument("--go-file", required=True)
    p.add_argument("--started-file", required=True)

    args = p.parse_args()

    # Keep QoS fixed so placement is the main variable.
    set_process_qos("high")

    cpus = PLACEMENTS[args.placement]

    if cpus is not None:
        set_affinity(cpus)

    affinity = get_affinity()
    qos = get_process_qos_state()

    # Configure before torch creates worker infrastructure.
    os.environ["OMP_NUM_THREADS"] = str(
        args.threads
    )

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
        dtype=torch.float32,
    )

    # Warm-up excluded.
    with torch.inference_mode():
        for _ in range(3):
            _ = model(x)

    ready = Path(args.ready_file)
    go = Path(args.go_file)
    started = Path(args.started_file)

    ready.write_text("ready")

    wait_for_file(go)

    started.write_text("started")

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

    row = {
        "placement": args.placement,
        "allowed_cpus": cpus,

        "process_affinity_mask":
            affinity["process_mask"],

        "threads": args.threads,

        "duration_sec": elapsed,
        "images": images,

        "throughput_img_s":
            images / elapsed,

        "latency_ms_img":
            elapsed / images * 1000.0,

        "qos_controlled":
            qos["execution_speed_controlled"],

        "qos_throttled":
            qos["execution_speed_throttled"],
    }

    Path(args.result).write_text(
        json.dumps(
            row,
            indent=2,
        )
    )

    print(
        f"{args.placement:17} | "
        f"{row['throughput_img_s']:.3f} img/s | "
        f"{row['latency_ms_img']:.1f} ms/img | "
        f"mask=0x{row['process_affinity_mask']:x}"
    )


if __name__ == "__main__":
    main()
