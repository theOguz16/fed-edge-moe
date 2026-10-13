import argparse
import csv
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--worker", action="store_true")
p.add_argument("--summary-only", action="store_true")
p.add_argument("--out", required=True)
p.add_argument("--threads", nargs="+", type=int,
               default=[1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64])
p.add_argument("--thread", type=int, default=4)
p.add_argument("--repeat", type=int, default=1)
p.add_argument("--duration", type=float, default=20)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--size", type=int, default=320)
p.add_argument("--batch", type=int, default=1)
p.add_argument("--weights", choices=["pretrained", "random"],
               default="pretrained")
args = p.parse_args()

root = Path(args.out)
root.mkdir(parents=True, exist_ok=True)

def worker():
    import torch
    import torchvision
    from torchvision.models import (
        convnext_large, ConvNeXt_Large_Weights
    )

    torch.set_num_threads(args.thread)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)

    weights = (
        ConvNeXt_Large_Weights.IMAGENET1K_V1
        if args.weights == "pretrained" else None
    )

    model = convnext_large(weights=weights).eval()
    x = torch.randn(args.batch, 3, args.size, args.size)

    with torch.inference_mode():
        for _ in range(args.warmup):
            y = model(x)

        if not torch.isfinite(y).all().item():
            raise RuntimeError("Non-finite model output")

        elapsed = []
        start = time.perf_counter()

        while True:
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            elapsed.append(t1 - t0)

            if t1 - start >= args.duration and len(elapsed) >= 3:
                break

        wall = t1 - start

    values = sorted(t * 1000 for t in elapsed)
    pos = 0.95 * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    p95 = values[lo] + (values[hi] - values[lo]) * (pos - lo)

    row = {
        "host": platform.node(),
        "os": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "weights": args.weights,
        "model": "convnext_large",
        "precision": "fp32",
        "device": "cpu",
        "threads": args.thread,
        "repeat": args.repeat,
        "size": args.size,
        "batch": args.batch,
        "window_s": wall,
        "calls": len(elapsed),
        "throughput_img_s": args.batch * len(elapsed) / wall,
        "latency_p50_ms": statistics.median(values),
        "latency_p95_ms": p95,
    }

    dest = root / f"t{args.thread}_r{args.repeat}.csv"
    temp = root / f"t{args.thread}_r{args.repeat}.tmp"

    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    temp.replace(dest)

    print(
        f"{args.thread} threads | repeat={args.repeat} | "
        f"{row['throughput_img_s']:.4f} img/s | "
        f"P50={row['latency_p50_ms']:.2f}ms"
    )

def summarize():
    rows = []
    for path in root.glob("t*_r*.csv"):
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))

    if not rows:
        print("No measurements yet")
        return

    by_thread = {}
    for r in rows:
        by_thread.setdefault(int(r["threads"]), []).append(
            float(r["throughput_img_s"])
        )

    summary = []
    for thread, values in sorted(by_thread.items()):
        summary.append({
            "threads": thread,
            "median_throughput": statistics.median(values),
            "min_throughput": min(values),
            "max_throughput": max(values),
            "repeats": len(values),
        })

    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print("\n===== THREAD SCALING SUMMARY =====")
    for r in summary:
        print(
            f"{r['threads']:2d} threads | "
            f"{r['median_throughput']:.4f} img/s "
            f"| repeats={r['repeats']}"
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [r["threads"] for r in summary]
        y = [r["median_throughput"] for r in summary]
        low = [r["median_throughput"] - r["min_throughput"]
               for r in summary]
        high = [r["max_throughput"] - r["median_throughput"]
                for r in summary]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(x, y, yerr=[low, high],
                    marker="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xlabel("CPU intra-op threads")
        ax.set_ylabel("Throughput (images/s)")
        ax.set_title("ANDASIS ConvNeXt-Large CPU Scaling")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / "thread_scaling.png", dpi=250)
        plt.close(fig)
        print("Graph:", root / "thread_scaling.png")
    except ImportError:
        print("Install matplotlib to generate the graph.")

def run():
    if args.duration <= 0 or args.repeats <= 0:
        p.error("Duration and repeats must be positive.")

    if hasattr(os, "sched_getaffinity"):
        available = len(os.sched_getaffinity(0))
    else:
        available = (
            os.process_cpu_count()
            if hasattr(os, "process_cpu_count")
            else os.cpu_count()
        )

    print("Host:", platform.node())
    print("OS:", platform.platform())
    print("Accessible logical CPUs (estimate):", available)

    threads = sorted(set(
        t for t in args.threads if t > 0
        and (available is None or t <= available)
    ))

    if not threads:
        raise RuntimeError("No valid thread counts")

    jobs = [
        (t, r)
        for t in threads
        for r in range(1, args.repeats + 1)
    ]
    random.Random(42).shuffle(jobs)

    for i, (thread, repeat) in enumerate(jobs, 1):
        dest = root / f"t{thread}_r{repeat}.csv"
        if dest.exists():
            with dest.open(newline="") as f:
                existing = list(csv.DictReader(f))
            if (len(existing) == 1
                and int(existing[0]["threads"]) == thread
                and int(existing[0]["repeat"]) == repeat
                and int(existing[0]["size"]) == args.size
                and int(existing[0]["batch"]) == args.batch
                and existing[0]["weights"] == args.weights):
                print(f"[{i}/{len(jobs)}] SKIP {dest.name}")
                continue
            raise RuntimeError(f"Existing file mismatch: {dest}")

        print(f"[{i}/{len(jobs)}] thread={thread}, repeat={repeat}",
              flush=True)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(thread)
        env["MKL_NUM_THREADS"] = str(thread)

        subprocess.run([
            sys.executable, str(Path(__file__).resolve()),
            "--worker",
            "--out", str(root),
            "--thread", str(thread),
            "--repeat", str(repeat),
            "--duration", str(args.duration),
            "--warmup", str(args.warmup),
            "--size", str(args.size),
            "--batch", str(args.batch),
            "--weights", args.weights,
        ], check=True, env=env)

    summarize()

if args.worker:
    worker()
elif args.summary_only:
    summarize()
else:
    run()
