import csv
import hashlib
import json
import os
import queue
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
BASE = Path("results/benchmark_v3").resolve()

SOURCE_WORKER = BASE / (
    "energy_pilot/t20_4_20261017T190349Z_2438/t20_4.py"
)
SOURCE_CHECKS = SOURCE_WORKER.parent / "checksums.json"
SOURCE_DB = BASE / (
    "dataset_extensions/t20_9_20261017T195103Z_2438/"
    "dataset_v3_with_energy.sqlite"
)
T05 = BASE / (
    "validation/t05_20261015T173344Z_2438/metadata.json"
)
HOLDOUT = BASE / (
    "holdout_thread_confirmation/"
    "t12_20261015T192923Z_2438/holdout_images.json"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/"
    "imagenetv2-matched-frequency-format-val"
).resolve()

MAPPINGS = {
    "mps_cpu_cut3": ("mps", "cpu", 3, "AAACCC"),
    "mps_cpu_cut5": ("mps", "cpu", 5, "AAAAAC"),
    "cpu_mps_cut3": ("cpu", "mps", 3, "CCCAAA"),
}
BATCH = 8
MICROBATCHES = (1, 2)
REPEATS = (1, 2)
WARMUPS = 2
TIMED_ROUNDS = 10

CASES = [
    (name, mb, repeat)
    for repeat in REPEATS
    for name in MAPPINGS
    for mb in MICROBATCHES
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")

    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)

def verify_source():
    assert SOURCE_WORKER.is_file()
    assert SOURCE_DB.is_file()
    assert T05.is_file()
    assert HOLDOUT.is_file()

    checks = json.loads(SOURCE_CHECKS.read_text())
    assert sha(SOURCE_WORKER) == checks["t20_4.py"]

    db = sqlite3.connect(
        f"file:{SOURCE_DB}?mode=ro", uri=True
    )
    try:
        assert db.execute(
            "PRAGMA quick_check"
        ).fetchone()[0] == "ok"

        states = dict(db.execute(
            "SELECT state,COUNT(*) FROM tasks GROUP BY state"
        ).fetchall())

        assert states == {
            "completed": 6,
            "planned": 327330,
        }

        assert db.execute(
            "SELECT COUNT(*) FROM energy_measurements"
        ).fetchone()[0] == 18
    finally:
        db.close()

    assert shutil.disk_usage(ROOT).free > 5 * 1024**3

def build_inputs(weights):
    import torch
    from PIL import Image

    records = json.loads(HOLDOUT.read_text())
    assert isinstance(records, list) and records

    tensors = []
    checksums = []

    for i in range(BATCH):
        record = records[i % len(records)]
        path = IMAGE_ROOT / record["path"]

        assert path.is_file()
        assert sha(path) == record["sha256"]

        with Image.open(path) as image:
            tensor = weights.transforms()(
                image.convert("RGB")
            )

        tensors.append(tensor)
        checksums.append(record["sha256"])

    return torch.stack(tensors), checksums


def t26a_set_qos(name):
    import ctypes
    values = {
        "background": 0x09,
        "utility": 0x11,
        "default": 0x15,
        "user_initiated": 0x19,
    }
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    setter = lib.pthread_set_qos_class_self_np
    getter = lib.qos_class_self
    setter.argtypes = [ctypes.c_uint, ctypes.c_int]
    setter.restype = ctypes.c_int
    getter.restype = ctypes.c_uint
    code = setter(values[name], 0)
    if code != 0 or getter() != values[name]:
        raise RuntimeError("QOS_VERIFICATION_FAILED")

def t26a_power_pairs(raw):
    import re
    import math
    cpu = list(re.finditer(
        r"CPU Power:\s*([\d.]+)\s*mW", raw, re.I
    ))
    pairs = []
    for i, match in enumerate(cpu):
        stop = (
            cpu[i+1].start()
            if i+1 < len(cpu) else len(raw)
        )
        gpu = re.search(
            r"GPU Power:\s*([\d.]+)\s*mW",
            raw[match.end():stop], re.I
        )
        if gpu:
            a = float(match.group(1))/1000
            b = float(gpu.group(1))/1000
            if (
                math.isfinite(a) and math.isfinite(b)
                and a >= 0 and b >= 0
            ):
                pairs.append((a,b))
    return pairs

def worker(mapping_name, mb_size, repeat, result_path):
    import torch
    from torch import nn
    from torchvision.models import (
        ResNet50_Weights,
        resnet50,
    )

    assert (
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        == "0"
    )
    assert torch.backends.mps.is_available()

    t26a_set_qos(os.environ["T26A_QOS"])
    torch.set_num_threads(4)
    torch.manual_seed(2026)

    dev1, dev2, cut, bits = MAPPINGS[mapping_name]

    weights = ResNet50_Weights.IMAGENET1K_V2
    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    expected_sha = json.loads(T05.read_text())[
        "models"
    ]["resnet50"]["weight_sha256"]

    assert checkpoint.is_file()
    assert sha(checkpoint) == expected_sha

    model = resnet50(weights=None).eval()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    del state

    inputs, image_shas = build_inputs(weights)

    with torch.inference_mode():
        reference = model(inputs).detach().clone()

    stages = [
        nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        ),
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        nn.Sequential(
            model.avgpool,
            nn.Flatten(1),
            model.fc,
        ),
    ]

    for i, stage in enumerate(stages):
        stage.to(dev1 if i < cut else dev2)

    # Complete model placement before starting worker threads.
    torch.mps.synchronize()

    first = stages[:cut]
    second = stages[cut:]

    microbatches = [
        inputs[i:i + mb_size]
        for i in range(0, BATCH, mb_size)
    ]

    def first_stage(index, x):
        start = time.monotonic_ns()

        y = x.to(dev1)

        for layer in first:
            y = layer(y)

        if dev1 == "mps":
            torch.mps.synchronize()

        compute_end = time.monotonic_ns()

        transfer_start = compute_end

        if dev2 == "mps":
            # CPU->MPS: producer handles CPU only.
            # GPU transfer belongs to the consumer thread.
            transfer_end = transfer_start
        else:
            if y.device.type != dev2:
                y = y.to(dev2)
            transfer_end = time.monotonic_ns()

        return y, {
            "index": index,
            "stage1_start_ns": start,
            "stage1_end_ns": compute_end,
            "transfer_start_ns": transfer_start,
            "transfer_end_ns": transfer_end,
        }

    def second_stage(index, y, first_event):
        # Only the consumer submits GPU work in CPU->MPS mode.
        if dev2 == "mps" and y.device.type != "mps":
            transfer_start = time.monotonic_ns()
            y = y.to("mps")
            torch.mps.synchronize()
            transfer_end = time.monotonic_ns()

            first_event = dict(first_event)
            first_event["transfer_start_ns"] = transfer_start
            first_event["transfer_end_ns"] = transfer_end

        # Stage-2 compute starts after the transfer.
        start = time.monotonic_ns()

        for layer in second:
            y = layer(y)

        if dev2 == "mps":
            torch.mps.synchronize()

        compute_end = time.monotonic_ns()

        y = y.to("cpu").detach().clone()
        finished = time.monotonic_ns()

        event = dict(first_event)
        event.update({
            "stage2_start_ns": start,
            "stage2_end_ns": compute_end,
            "output_ready_ns": finished,
        })

        return y, event

    def sequential():
        results = [None] * len(microbatches)
        events = []

        started = time.monotonic_ns()

        with torch.inference_mode():
            for index, x in enumerate(microbatches):
                y, first_event = first_stage(index, x)
                output, event = second_stage(
                    index, y, first_event
                )
                results[index] = output
                events.append(event)

        ended = time.monotonic_ns()

        return torch.cat(results), events, started, ended

    def pipeline():
        channel = queue.Queue(maxsize=2)
        stop = threading.Event()
        errors = []
        results = [None] * len(microbatches)
        events = [None] * len(microbatches)

        def producer():
            try:
                with torch.inference_mode():
                    for index, x in enumerate(microbatches):
                        if stop.is_set():
                            break

                        y, event = first_stage(index, x)

                        while not stop.is_set():
                            try:
                                channel.put(
                                    (index, y, event),
                                    timeout=0.25
                                )
                                break
                            except queue.Full:
                                continue

                    while not stop.is_set():
                        try:
                            channel.put(None, timeout=0.25)
                            break
                        except queue.Full:
                            continue

            except BaseException:
                errors.append(traceback.format_exc())
                stop.set()

        def consumer():
            try:
                with torch.inference_mode():
                    while True:
                        try:
                            item = channel.get(timeout=0.25)
                        except queue.Empty:
                            if stop.is_set():
                                return
                            continue

                        if item is None:
                            return

                        index, y, first_event = item

                        output, event = second_stage(
                            index, y, first_event
                        )
                        results[index] = output
                        events[index] = event

            except BaseException:
                errors.append(traceback.format_exc())
                stop.set()

        producer_thread = threading.Thread(
            target=producer, daemon=True
        )
        consumer_thread = threading.Thread(
            target=consumer, daemon=True
        )

        started = time.monotonic_ns()

        consumer_thread.start()
        producer_thread.start()

        producer_thread.join(timeout=120)
        consumer_thread.join(timeout=120)

        if (
            producer_thread.is_alive()
            or consumer_thread.is_alive()
        ):
            stop.set()
            raise RuntimeError(
                "Pipeline thread timeout/deadlock"
            )

        if errors:
            raise RuntimeError(
                "Pipeline worker failure:\n"
                + "\n".join(errors)
            )

        assert all(x is not None for x in results)
        assert all(x is not None for x in events)

        ended = time.monotonic_ns()

        return torch.cat(results), events, started, ended

    def overlap_ns(events):
        overlap = 0

        for a in events:
            for b in events:
                if a["index"] == b["index"]:
                    continue

                left = max(
                    a["stage1_start_ns"],
                    b["stage2_start_ns"],
                )
                right = min(
                    a["stage1_end_ns"],
                    b["stage2_end_ns"],
                )

                overlap += max(0, right - left)

        return overlap

    def run_once(mode):
        if mode == "sequential":
            return sequential()
        return pipeline()

    mode_order = (
        ["sequential", "pipeline"]
        if repeat == 1
        else ["pipeline", "sequential"]
    )

    measurements = {}

    for mode in mode_order:
        for _ in range(WARMUPS):
            output, _, _, _ = run_once(mode)

            assert torch.allclose(
                output, reference,
                rtol=1e-3, atol=1e-3
            )

        rounds = []
        power_path = result_path.with_name(
            "powermetrics_" + mode + ".txt"
        )
        with power_path.open("w") as sensor_file:
            sensor = subprocess.Popen(
                [
                    "sudo", "-n", "powermetrics",
                    "--samplers", "cpu_power,gpu_power",
                    "-i", "1000", "-n", "2000",
                ],
                stdout=sensor_file,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            mode_started = time.monotonic()
            try:
                while (
                    len(rounds) < TIMED_ROUNDS
                    or time.monotonic() - mode_started < 10.0
                ):
                    round_index = len(rounds)
                output, events, start, end = run_once(mode)

                difference = (
                    output - reference
                ).abs().max().item()

                numerical_pass = bool(torch.allclose(
                    output, reference,
                    rtol=1e-3, atol=1e-3
                ))
                top1_pass = bool(torch.equal(
                    output.argmax(dim=1),
                    reference.argmax(dim=1)
                ))

                assert numerical_pass
                assert top1_pass

                duration_s = (end - start) / 1e9
                overlap = overlap_ns(events)

                assert len(events) == len(microbatches)
                assert duration_s > 0

                micro_latencies_ms = [
                    (
                        event["output_ready_ns"]
                        - event["stage1_start_ns"]
                    ) / 1e6
                    for event in events
                ]

                transfer_ms = [
                    (
                        event["transfer_end_ns"]
                        - event["transfer_start_ns"]
                    ) / 1e6
                    for event in events
                ]

                rounds.append({
                    "round": round_index + 1,
                    "wall_time_ms": duration_s * 1000,
                    "throughput_images_s": BATCH / duration_s,
                    "stage_compute_overlap_ms": overlap / 1e6,
                    "microbatch_latency_median_ms":
                        statistics.median(micro_latencies_ms),
                    "transfer_median_ms":
                        statistics.median(transfer_ms),
                    "max_abs_error": float(difference),
                    "numerical_pass": numerical_pass,
                    "top1_agreement": top1_pass,
                    "events": events,
                })
            finally:
                power_window_s = (
                    time.monotonic() - mode_started
                )
                if sensor.poll() is None:
                    sensor.send_signal(signal.SIGINT)
                try:
                    sensor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    sensor.kill()
                    sensor.wait()

        pairs = t26a_power_pairs(
            power_path.read_text(errors="replace")
        )
        if len(pairs) < 2:
            raise RuntimeError(
                "POWER_MISSING "
                f"mode={mode} samples={len(pairs)} "
                f"sensor_exit={sensor.returncode}"
            )

        cpu_w = statistics.mean(x[0] for x in pairs)
        gpu_w = statistics.mean(x[1] for x in pairs)
        sustained_ips = (
            BATCH * len(rounds) / power_window_s
        )
        assert sustained_ips > 0

        measurements[mode] = {
            "rounds": rounds,
            "power_cpu_w": cpu_w,
            "power_gpu_w": gpu_w,
            "power_sample_count": len(pairs),
            "power_window_s": power_window_s,
            "power_raw_sha256": sha(power_path),
            "sustained_throughput_images_s": sustained_ips,
            "gross_energy_j_image": (cpu_w+gpu_w)/sustained_ips,
            "throughput_median_images_s": statistics.median(
                x["throughput_images_s"] for x in rounds
            ),
            "batch_wall_median_ms": statistics.median(
                x["wall_time_ms"] for x in rounds
            ),
            "microbatch_latency_median_ms": statistics.median(
                x["microbatch_latency_median_ms"]
                for x in rounds
            ),
            "transfer_median_ms": statistics.median(
                x["transfer_median_ms"] for x in rounds
            ),
            "stage_overlap_median_ms": statistics.median(
                x["stage_compute_overlap_ms"]
                for x in rounds
            ),
            "rounds_with_positive_overlap": sum(
                x["stage_compute_overlap_ms"] > 0
                for x in rounds
            ),
        }

    seq = measurements["sequential"]
    pipe = measurements["pipeline"]

    gain = (
        pipe["throughput_median_images_s"]
        / seq["throughput_median_images_s"]
    )

    save(result_path, {
        "protocol": "T26A_RESNET_PIPELINE_ENERGY_V1",
        "status": "numerical_validation_pass",
        "model": "resnet50",
        "hardware": "apple_m4",
        "precision": "fp32",
        "cpu_threads": 4,
        "batch_size": BATCH,
        "microbatch_size": mb_size,
        "microbatch_count": len(microbatches),
        "mapping": mapping_name,
        "mapping_bits": bits,
        "cut": cut,
        "stage1_device": dev1,
        "stage2_device": dev2,
        "stage_queue_maxsize": 2,
        "independent_process_repeat": repeat,
        "mode_order": mode_order,
        "warmups_per_mode": WARMUPS,
        "timed_rounds_per_mode": TIMED_ROUNDS,
        "weights_sha256": expected_sha,
        "input_image_sha256_list": image_shas,
        "reference_top1_indices":
            reference.argmax(dim=1).tolist(),
        "measurements": measurements,
        "pipeline_speedup_over_sequential": gain,
        "pipeline_has_actual_separate_threads": True,
        "hardware_kernel_overlap_proven": False,
        "timeline_observation":
            "CPU wall-clock stage intervals with MPS synchronization",
        "power_measured": True,
        "energy_measured": True,
        "measurement_scope":
            "instrumented_repeated_8_image_batches",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    })

    print(
        "CASE DONE:",
        mapping_name,
        "microbatch=", mb_size,
        "repeat=", repeat,
        f"seq={seq['throughput_median_images_s']:.2f}",
        f"pipe={pipe['throughput_median_images_s']:.2f}",
        f"speedup={gain:.3f}x",
        f"overlap={pipe['stage_overlap_median_ms']:.3f}ms",
        flush=True,
    )

def case_folder(name, mb, repeat):
    return (
        ROOT
        / f"repeat_{repeat}"
        / f"{name}_micro{mb}"
    )

def verify_completed(folder, name, mb, repeat):
    path = folder / "summary.json"

    if not path.exists():
        return None

    summary = json.loads(path.read_text())

    assert summary["mapping"] == name
    assert summary["microbatch_size"] == mb
    assert summary["independent_process_repeat"] == repeat

    worker_file = ROOT / summary["worker_relative_path"]
    assert sha(worker_file) == summary["worker_sha256"]

    worker_data = json.loads(worker_file.read_text())

    assert worker_data["status"] == "numerical_validation_pass"
    assert worker_data["mapping"] == name
    assert worker_data["microbatch_size"] == mb
    assert (
        worker_data["independent_process_repeat"]
        == repeat
    )

    return summary

def run_case(name, mb, repeat):
    folder = case_folder(name, mb, repeat)
    folder.mkdir(parents=True, exist_ok=True)

    done = verify_completed(folder, name, mb, repeat)

    if done is not None:
        print(
            "SKIP VERIFIED:", name,
            "microbatch=", mb,
            "repeat=", repeat,
            flush=True,
        )
        return done

    attempt = folder / (
        "attempt_" + uuid.uuid4().hex
    )
    attempt.mkdir()

    output_path = attempt / "result.json"
    stdout_path = attempt / "stdout.txt"
    stderr_path = attempt / "stderr.txt"

    print(
        "RUN:", name,
        "microbatch=", mb,
        "repeat=", repeat,
        flush=True,
    )

    with stdout_path.open("w") as stdout, \
         stderr_path.open("w") as stderr:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "worker",
                name,
                str(mb),
                str(repeat),
                str(output_path),
            ],
            stdout=stdout,
            stderr=stderr,
            env={
                **os.environ,
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            },
            start_new_session=True,
        )

        try:
            code = proc.wait(timeout=360)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise

    if code != 0 or not output_path.is_file():
        failure = {
            "protocol": "T21_failure_record",
            "mapping": name,
            "microbatch": mb,
            "repeat": repeat,
            "returncode": code,
            "stderr_relative_path": str(
                stderr_path.relative_to(ROOT)
            ),
            "stdout_relative_path": str(
                stdout_path.relative_to(ROOT)
            ),
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }
        save(attempt / "failure.json", failure)

        print(
            "FAILED — raw logs saved:",
            attempt.relative_to(ROOT),
            flush=True,
        )
        return None

    data = json.loads(output_path.read_text())

    assert data["status"] == "numerical_validation_pass"
    assert data["mapping"] == name
    assert data["microbatch_size"] == mb
    assert data["independent_process_repeat"] == repeat

    summary = {
        "protocol": "T26A_RESNET_PIPELINE_ENERGY_V1",
        "mapping": name,
        "microbatch_size": mb,
        "independent_process_repeat": repeat,
        "worker_relative_path": str(
            output_path.relative_to(ROOT)
        ),
        "worker_sha256": sha(output_path),
        "stdout_sha256": sha(stdout_path),
        "stderr_sha256": sha(stderr_path),
        "sequential_throughput_images_s":
            data["measurements"]["sequential"][
                "throughput_median_images_s"
            ],
        "pipeline_throughput_images_s":
            data["measurements"]["pipeline"][
                "throughput_median_images_s"
            ],
        "pipeline_speedup":
            data["pipeline_speedup_over_sequential"],
        "pipeline_stage_overlap_median_ms":
            data["measurements"]["pipeline"][
                "stage_overlap_median_ms"
            ],
        "pipeline_positive_overlap_rounds":
            data["measurements"]["pipeline"][
                "rounds_with_positive_overlap"
            ],
        "measurement_status": "validated_exploratory",
    }

    # The completed marker is written last.
    save(folder / "summary.json", summary)

    print(
        f"COMMITTED: seq={summary['sequential_throughput_images_s']:.2f}"
        f" pipe={summary['pipeline_throughput_images_s']:.2f}"
        f" speedup={summary['pipeline_speedup']:.3f}x"
        f" overlap={summary['pipeline_stage_overlap_median_ms']:.3f} ms",
        flush=True,
    )

    return summary

def gather():
    rows = []

    for name, mb, repeat in CASES:
        item = verify_completed(
            case_folder(name, mb, repeat),
            name, mb, repeat
        )
        if item is not None:
            rows.append(item)

    return rows

def export_csv(rows):
    if not rows:
        return

    fields = [
        "mapping",
        "microbatch_size",
        "independent_process_repeat",
        "sequential_throughput_images_s",
        "pipeline_throughput_images_s",
        "pipeline_speedup",
        "pipeline_stage_overlap_median_ms",
        "pipeline_positive_overlap_rounds",
        "measurement_status",
        "worker_relative_path",
        "worker_sha256",
    ]

    path = ROOT / "pipeline_measurements.csv"
    temp = ROOT / "pipeline_measurements.csv.tmp"

    with temp.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp, path)

def create_timeline(rows):
    target = next(
        (
            r for r in rows
            if r["mapping"] == "cpu_mps_cut3"
            and r["microbatch_size"] == 1
            and r["independent_process_repeat"] == 1
        ),
        None,
    )

    if target is None:
        return

    raw = json.loads(
        (ROOT / target["worker_relative_path"]).read_text()
    )

    events = raw["measurements"]["pipeline"][
        "rounds"
    ][0]["events"]

    all_times = [
        value
        for event in events
        for key, value in event.items()
        if key.endswith("_ns") and isinstance(value, int)
    ]

    start = min(all_times)
    end = max(all_times)
    span = max(1, end - start)

    def x(ns):
        return 180 + 850 * (ns - start) / span

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="1120" height="550" viewBox="0 0 1120 550">',
        '<rect width="1120" height="550" fill="white"/>',
        '<text x="35" y="38" font-family="Arial" '
        'font-size="22" font-weight="bold">'
        'T21 CPU-to-MPS cut 3 — measured pipeline timeline</text>',
        '<text x="35" y="63" font-family="Arial" '
        'font-size="13" fill="#555">'
        'Batch 8, microbatch 1, first timed pipeline round; '
        'CPU-side synchronized wall-clock intervals</text>',
    ]

    for event in sorted(events, key=lambda e: e["index"]):
        idx = event["index"]
        y = 103 + idx * 46

        parts.append(
            f'<text x="35" y="{y+15}" font-size="14" '
            f'font-family="Arial">Microbatch {idx+1}</text>'
        )

        segments = [
            ("stage1_start_ns", "stage1_end_ns", "#326ba8"),
            ("transfer_start_ns", "transfer_end_ns", "#dc9c39"),
            ("stage2_start_ns", "stage2_end_ns", "#288268"),
        ]

        for a, b, color in segments:
            x1 = x(event[a])
            x2 = x(event[b])
            width = max(1, x2-x1)

            parts.append(
                f'<rect x="{x1:.2f}" y="{y}" '
                f'width="{width:.2f}" height="22" '
                f'rx="3" fill="{color}"/>'
            )

    parts.extend([
        '<rect x="170" y="498" width="15" height="15" '
        'fill="#326ba8"/>',
        '<text x="191" y="511" font-size="13" '
        'font-family="Arial">CPU stage</text>',
        '<rect x="340" y="498" width="15" height="15" '
        'fill="#dc9c39"/>',
        '<text x="361" y="511" font-size="13" '
        'font-family="Arial">Transfer</text>',
        '<rect x="515" y="498" width="15" height="15" '
        'fill="#288268"/>',
        '<text x="536" y="511" font-size="13" '
        'font-family="Arial">MPS stage</text>',
        '<text x="790" y="511" font-size="13" '
        'font-family="Arial" fill="#555">'
        'Time →</text>',
        '</svg>',
    ])

    path = ROOT / "pipeline_timeline.svg"
    path.write_text("\n".join(parts), encoding="utf-8")

def find_task(conn, bits, mb, mode):
    matches = conn.execute(
        """
        SELECT t.task_id
        FROM tasks t
        JOIN placements p
          ON t.placement_id = p.placement_id
        WHERE p.model = 'resnet50'
          AND p.group_mapping = ?
          AND t.cpu_threads = 4
          AND t.workload_id = ?
          AND t.execution_mode = ?
          AND t.hardware_id = 'mac_m4'
        """,
        (bits, f"batch8_micro{mb}", mode),
    ).fetchall()

    assert len(matches) == 1, (
        "Expected exactly one catalog task:",
        bits, mb, mode, len(matches)
    )

    return matches[0][0]

def integrate_dataset(rows):
    destination = ROOT / "dataset_v3_with_pipeline.sqlite"

    if destination.exists():
        db = sqlite3.connect(
            f"file:{destination}?mode=ro",
            uri=True
        )
        try:
            assert db.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0] == "ok"
            assert db.execute(
                "SELECT COUNT(*) FROM pipeline_measurements"
            ).fetchone()[0] == 24
        finally:
            db.close()
        return destination

    temp = ROOT / (
        "dataset_v3_pipeline_"
        + uuid.uuid4().hex
        + ".tmp"
    )

    source = sqlite3.connect(
        f"file:{SOURCE_DB}?mode=ro",
        uri=True,
    )
    dest = sqlite3.connect(str(temp))

    try:
        source.backup(dest)

        dest.execute("""
            CREATE TABLE pipeline_measurements (
                measurement_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                mapping TEXT NOT NULL,
                microbatch_size INTEGER NOT NULL,
                independent_process_repeat INTEGER NOT NULL,
                execution_mode TEXT NOT NULL,
                throughput_images_s REAL NOT NULL,
                batch_wall_median_ms REAL NOT NULL,
                microbatch_latency_median_ms REAL NOT NULL,
                transfer_median_ms REAL NOT NULL,
                stage_overlap_median_ms REAL NOT NULL,
                positive_overlap_rounds INTEGER NOT NULL,
                result_relative_path TEXT NOT NULL,
                result_sha256 TEXT NOT NULL,
                evidence_status TEXT NOT NULL,
                FOREIGN KEY(task_id) REFERENCES tasks(task_id),
                UNIQUE(
                    mapping, microbatch_size,
                    independent_process_repeat,
                    execution_mode
                )
            )
        """)

        for row in rows:
            name = row["mapping"]
            mb = row["microbatch_size"]
            repeat = row["independent_process_repeat"]

            raw = json.loads(
                (
                    ROOT / row["worker_relative_path"]
                ).read_text()
            )

            for mode in ("sequential", "pipeline"):
                task_id = find_task(
                    dest,
                    MAPPINGS[name][3],
                    mb,
                    mode,
                )

                metrics = raw["measurements"][mode]
                mid = hashlib.sha256(
                    f"T21|{name}|{mb}|{repeat}|{mode}".encode()
                ).hexdigest()

                dest.execute(
                    """
                    INSERT INTO pipeline_measurements
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        "t21_" + mid,
                        task_id,
                        name,
                        mb,
                        repeat,
                        mode,
                        metrics["throughput_median_images_s"],
                        metrics["batch_wall_median_ms"],
                        metrics["microbatch_latency_median_ms"],
                        metrics["transfer_median_ms"],
                        metrics["stage_overlap_median_ms"],
                        metrics["rounds_with_positive_overlap"],
                        row["worker_relative_path"],
                        row["worker_sha256"],
                        "validated_exploratory",
                    ),
                )

        dest.commit()

        assert dest.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        assert dest.execute(
            "SELECT COUNT(*) FROM pipeline_measurements"
        ).fetchone()[0] == 24

        assert dest.execute(
            "SELECT COUNT(*) FROM energy_measurements"
        ).fetchone()[0] == 18

        assert dict(dest.execute(
            "SELECT state,COUNT(*) FROM tasks GROUP BY state"
        ).fetchall()) == {
            "completed": 6,
            "planned": 327330,
        }

        assert dest.execute(
            "PRAGMA foreign_key_check"
        ).fetchall() == []

    finally:
        dest.close()
        source.close()

    os.replace(temp, destination)
    return destination

def run():
    verify_source()

    plan_path = ROOT / "experiment_plan.json"

    plan = {
        "protocol": "T26A_RESNET_PIPELINE_ENERGY_V1",
        "model": "resnet50",
        "cpu_threads": 4,
        "batch_size": BATCH,
        "microbatch_sizes": list(MICROBATCHES),
        "repeats": list(REPEATS),
        "warmups": WARMUPS,
        "timed_rounds_per_mode": TIMED_ROUNDS,
        "cases": [list(x) for x in CASES],
        "source_db_sha256": sha(SOURCE_DB),
        "source_worker_sha256": sha(SOURCE_WORKER),
        "script_sha256": sha(Path(__file__)),
    }

    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, (
            "Campaign plan/source changed: refusing resume"
        )
    else:
        save(plan_path, plan)

    print("=== T21 REAL PIPELINE CAMPAIGN ===")
    print("MODEL: ResNet50 FP32")
    print("HARDWARE: Apple M4 CPU + MPS")
    print("BATCH SIZE: 8")
    print("MICROBATCH SIZES: 1, 2")
    print("PLACEMENTS: 3")
    print("INDEPENDENT PROCESSES: 12")
    print("TIMED ROUNDS PER MODE: 10")
    print("EXECUTION: sequential vs two-thread pipeline")
    print("ORIGINAL SQLITE: READ-ONLY")
    print("INTERRUPT: Ctrl+C")
    print("RESULT DIRECTORY:", ROOT)
    print()

    try:
        for index, (name, mb, repeat) in enumerate(CASES, 1):
            if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                print("DISK GUARD: STOPPED", flush=True)
                break

            print(f"\nCASE {index}/12", flush=True)
            run_case(name, mb, repeat)

            rows = gather()
            export_csv(rows)

    except KeyboardInterrupt:
        print(
            "\nT21 INTERRUPTED: completed results retained.",
            flush=True,
        )

    rows = gather()
    export_csv(rows)

    print("\n=== T21 CAMPAIGN STATUS ===")
    print("COMPLETED PROCESSES:", len(rows), "/12")
    print(
        "CASES WITH POSITIVE TIMELINE OVERLAP:",
        sum(
            r["pipeline_positive_overlap_rounds"] > 0
            for r in rows
        ),
        "/", len(rows),
    )

    if len(rows) != len(CASES):
        print("STATUS: INCOMPLETE — safe to resume")
        print(
            "RESUME:",
            sys.executable,
            "-u",
            Path(__file__).resolve(),
            "run",
        )
        return

    create_timeline(rows)
    dataset = integrate_dataset(rows)

    # Verify every saved success before declaring completion.
    assert len(gather()) == 12

    report = {
        "protocol": "T26A_RESNET_PIPELINE_ENERGY_V1",
        "completed_independent_processes": 12,
        "mode_evaluations": 24,
        "source_database_unmodified": True,
        "new_database": str(dataset),
        "hardware_kernel_overlap_proven": False,
        "pipeline_timeline_available": (
            ROOT / "pipeline_timeline.svg"
        ).exists(),
        "measured_power": False,
        "measured_energy": False,
        "cases_with_positive_stage_overlap": sum(
            r["pipeline_positive_overlap_rounds"] > 0
            for r in rows
        ),
    }

    save(ROOT / "final_report.json", report)

    files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file()
        and p.name != "checksums.json"
        and not p.name.endswith(".tmp")
    )

    save(ROOT / "checksums.json", {
        str(path.relative_to(ROOT)): sha(path)
        for path in files
    })

    print("\n=== T21 FINAL VALIDATION ===")
    print("INDEPENDENT PROCESSES: 12/12 PASS")
    print("SEQUENTIAL/PIPELINE EVALUATIONS: 24")
    print("NUMERICAL VALIDATION: PASS")
    print(
        "POSITIVE STAGE OVERLAP CASES:",
        report["cases_with_positive_stage_overlap"],
        "/12",
    )
    print("PIPELINE TIMELINE SVG: CREATED")
    print("SOURCE ENERGY MEASUREMENTS: 18 PRESERVED")
    print("ORIGINAL TASKS: 327336 PRESERVED")
    print("ORIGINAL COMPLETED TASKS: 6 PRESERVED")
    print("OLD SQLITE MODIFIED: NO")
    print("NEW SQLITE INTEGRITY: PASS")
    print("ARTIFACT CHECKSUM INVENTORY: CREATED")
    print("T21 REAL PIPELINE CAMPAIGN: PASS")
    print("NEW SQLITE:", dataset)
    print("RESULT DIRECTORY:", ROOT)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "run":
        run()

    elif len(sys.argv) == 6 and sys.argv[1] == "worker":
        worker(
            sys.argv[2],
            int(sys.argv[3]),
            int(sys.argv[4]),
            Path(sys.argv[5]).resolve(),
        )

    else:
        raise SystemExit(
            "Usage: python t21.py run"
        )
