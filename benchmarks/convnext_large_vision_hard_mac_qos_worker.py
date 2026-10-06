import argparse
import ctypes
import json
import os
import time
from pathlib import Path


QOS_CLASSES = {
    "background": 0x09,
    "utility": 0x11,
    "default": 0x15,
    "userInitiated": 0x19,
}

QOS_NAMES = {
    value: key
    for key, value in QOS_CLASSES.items()
}


def set_current_thread_qos(name):
    qos_value = QOS_CLASSES[name]

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

    rc = setter(qos_value, 0)

    if rc != 0:
        raise RuntimeError(
            f"pthread_set_qos_class_self_np failed: {rc}"
        )

    libc.pthread_self.restype = ctypes.c_void_p
    thread = libc.pthread_self()

    getter = libc.pthread_get_qos_class_np
    getter.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_int),
    ]
    getter.restype = ctypes.c_int

    actual_qos = ctypes.c_uint()
    relative_priority = ctypes.c_int()

    rc = getter(
        thread,
        ctypes.byref(actual_qos),
        ctypes.byref(relative_priority),
    )

    if rc != 0:
        raise RuntimeError(
            f"pthread_get_qos_class_np failed: {rc}"
        )

    return {
        "requested": name,
        "actual_value": actual_qos.value,
        "actual_name": QOS_NAMES.get(
            actual_qos.value,
            f"unknown_0x{actual_qos.value:x}",
        ),
        "relative_priority": relative_priority.value,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qos",
        required=True,
        choices=QOS_CLASSES,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=320,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--result",
        required=True,
    )

    parser.add_argument(
        "--ready-file",
    )

    parser.add_argument(
        "--go-file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set QoS before importing torch so that PyTorch's
    # worker infrastructure is initialized after QoS setup.
    qos_info = set_current_thread_qos(args.qos)

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.threads)

    import torch
    from torchvision.models import (
        convnext_large,
        ConvNeXt_Large_Weights,
    )

    from resource_monitor import MemorySampler

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
        args.batch,
        3,
        args.resolution,
        args.resolution,
    )

    with torch.inference_mode():
        for _ in range(3):
            _ = model(x)

    # Signal that model load + warm-up are complete.
    if args.ready_file:
        Path(args.ready_file).write_text(
            json.dumps({
                "pid": os.getpid(),
                "qos": args.qos,
            })
        )

    # Formal runner starts powermetrics, then releases us.
    if args.go_file:
        deadline = time.monotonic() + 60.0
        go_path = Path(args.go_file)

        while not go_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for measurement go-file"
                )

            time.sleep(0.05)

    sampler = MemorySampler(
        interval_sec=0.2,
        include_children=False,
    )

    sampler.start()

    images = 0
    calls = 0

    start = time.perf_counter()

    with torch.inference_mode():
        while True:
            _ = model(x)

            images += args.batch
            calls += 1

            elapsed = time.perf_counter() - start

            if elapsed >= args.seconds:
                break

    elapsed = time.perf_counter() - start
    memory = sampler.stop()

    result = {
        "pid": os.getpid(),
        "qos_requested": qos_info["requested"],
        "qos_actual_main_thread": qos_info["actual_name"],
        "qos_actual_value": qos_info["actual_value"],
        "qos_relative_priority":
            qos_info["relative_priority"],

        "threads": args.threads,
        "resolution": args.resolution,
        "batch": args.batch,

        "duration_sec": elapsed,
        "calls": calls,
        "images": images,

        "throughput_img_s":
            images / elapsed,

        "latency_per_image_ms":
            (elapsed / images) * 1000.0,

        **memory,
    }

    Path(args.result).write_text(
        json.dumps(result, indent=2)
    )

    print(
        f"PID {result['pid']} | "
        f"QoS requested={result['qos_requested']} | "
        f"actual={result['qos_actual_main_thread']} | "
        f"{args.threads}t | "
        f"{result['throughput_img_s']:.2f} img/s | "
        f"{result['process_rss_peak_mb']:.0f} MB RSS"
    )


if __name__ == "__main__":
    main()
