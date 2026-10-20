import hashlib
import json
import math
import os
import re
import select
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None

BASE = Path("results/benchmark_v3").resolve()
DB = BASE / (
    "dataset_extensions/"
    "t19_3_20261017T183826Z_2438/"
    "dataset_v3_extended.sqlite"
)
T05 = BASE / (
    "validation/t05_20261015T173344Z_2438/"
    "metadata.json"
)
IMAGES = BASE / (
    "holdout_thread_confirmation/"
    "t12_20261015T192923Z_2438/"
    "holdout_images.json"
)
IMAGE_ROOT = Path(
    "data/imagenetv2/"
    "imagenetv2-matched-frequency-format-val"
).resolve()

DURATION_SECONDS = 18
POWER_SAMPLES = 10

MAPPINGS = {
    "full_cpu": "CCCCCC",
    "full_mps": "AAAAAA",
    "mps_cpu_cut5": "AAAAAC",
}

SEQUENCE = [
    ("full_cpu", 1),
    ("full_mps", 1),
    ("mps_cpu_cut5", 1),
    ("mps_cpu_cut5", 2),
    ("full_mps", 2),
    ("full_cpu", 2),
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
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

def worker(mapping_name, result_path):
    import torch
    from torch import nn
    from torchvision.models import resnet50, ResNet50_Weights
    from PIL import Image

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    assert torch.backends.mps.is_available()

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    weights = ResNet50_Weights.IMAGENET1K_V2

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    expected = json.loads(T05.read_text())[
        "models"
    ]["resnet50"]["weight_sha256"]

    assert checkpoint.is_file()
    assert sha(checkpoint) == expected

    model = resnet50(weights=None).eval()
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state, strict=True)
    del state

    image_info = json.loads(IMAGES.read_text())[0]
    image_path = IMAGE_ROOT / image_info["path"]

    assert sha(image_path) == image_info["sha256"]

    with Image.open(image_path) as image:
        x = weights.transforms()(
            image.convert("RGB")
        ).unsqueeze(0)

    stages = [
        nn.Sequential(
            model.conv1, model.bn1,
            model.relu, model.maxpool
        ),
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        nn.Sequential(
            model.avgpool,
            nn.Flatten(1),
            model.fc
        ),
    ]

    mapping = MAPPINGS[mapping_name]
    devices = [
        "cpu" if bit == "C" else "mps"
        for bit in mapping
    ]

    with torch.inference_mode():
        reference = model(x).detach().clone()

    for stage, device in zip(stages, devices):
        stage.to(device)

    uses_mps = "mps" in devices

    def forward():
        y = x

        for stage, device in zip(stages, devices):
            if y.device.type != device:
                y = y.to(device)
            y = stage(y)

        if y.device.type != "cpu":
            y = y.cpu()

        if uses_mps:
            torch.mps.synchronize()

        return y

    with torch.inference_mode():
        actual = forward()

        numerical_pass = bool(torch.allclose(
            reference, actual,
            atol=1e-3, rtol=1e-3
        ))
        agreement = (
            reference.argmax().item()
            == actual.argmax().item()
        )
        error = float(
            (reference - actual).abs().max().item()
        )

        assert numerical_pass and agreement

        for _ in range(5):
            result = forward()
            del result

    if uses_mps:
        torch.mps.synchronize()

    print("READY", flush=True)

    command = sys.stdin.readline().strip()
    assert command == "GO"

    started_ns = time.monotonic_ns()
    print("STARTED", flush=True)

    completions_ns = []
    durations_ns = []

    with torch.inference_mode():
        deadline = (
            started_ns
            + DURATION_SECONDS * 1_000_000_000
        )

        while time.monotonic_ns() < deadline:
            call_start = time.monotonic_ns()
            y = forward()
            call_end = time.monotonic_ns()

            del y

            durations_ns.append(call_end - call_start)
            completions_ns.append(call_end)

    ended_ns = time.monotonic_ns()

    assert len(completions_ns) > 0

    save(result_path, {
        "protocol": "T20_4_load_power_pilot",
        "model": "resnet50",
        "mapping": mapping_name,
        "device_sequence": devices,
        "cpu_threads": 4,
        "precision": "float32",
        "batch_size": 1,
        "image_sha256": image_info["sha256"],
        "weights_sha256": expected,
        "numerical_pass": numerical_pass,
        "top1_agreement": agreement,
        "max_abs_error": error,
        "active_start_monotonic_ns": started_ns,
        "active_end_monotonic_ns": ended_ns,
        "completion_timestamps_ns": completions_ns,
        "inference_durations_ns": durations_ns,
        "total_completed": len(completions_ns),
        "duration_seconds": (
            ended_ns - started_ns
        ) / 1e9,
        "torch_version": torch.__version__,
        "measurement_type": "sustained_inference",
    })

    print("FINISHED", flush=True)

def parse_power(raw):
    pattern = re.compile(
        r"^\s*(CPU Power|GPU Power|ANE Power|"
        r"Combined Power \(CPU \+ GPU \+ ANE\))"
        r":\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b"
    )

    names = {
        "CPU Power": "cpu",
        "GPU Power": "gpu",
        "ANE Power": "ane",
        "Combined Power (CPU + GPU + ANE)": "combined",
    }

    samples = []
    current = {}

    for line in raw.splitlines():
        match = pattern.match(line)
        if not match:
            continue

        name, number, unit = match.groups()
        watts = float(number) / (
            1000 if unit == "mW" else 1
        )

        key = names[name]
        current[key] = watts

        if key == "combined":
            if all(
                k in current
                for k in ("cpu", "gpu", "ane", "combined")
            ):
                samples.append({
                    k: current[k]
                    for k in (
                        "cpu", "gpu",
                        "ane", "combined"
                    )
                })
            current = {}

    return samples

def wait_for_line(proc, expected, seconds=120):
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Worker exited before {expected}, "
                f"returncode={proc.returncode}"
            )

        available, _, _ = select.select(
            [proc.stdout],
            [],
            [],
            min(1.0, max(
                0.01,
                deadline - time.monotonic()
            ))
        )

        if available:
            line = proc.stdout.readline().strip()

            if line == expected:
                return

            if not line:
                raise RuntimeError(
                    "Worker stdout closed unexpectedly"
                )

    raise TimeoutError(
        f"Worker did not signal {expected}"
    )

def run_one(mapping_name, repeat, pm):
    run_dir = OUT / f"{mapping_name}_repeat{repeat}"
    run_dir.mkdir(parents=True, exist_ok=False)

    result_path = run_dir / "worker_result.json"
    power_path = run_dir / "powermetrics_raw.txt"
    power_err = run_dir / "powermetrics_stderr.txt"
    worker_err = run_dir / "worker_stderr.txt"

    environment = {
        **os.environ,
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }

    with worker_err.open("w") as error_file:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                str(OUT),
                "worker",
                mapping_name,
                str(result_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=error_file,
            text=True,
            bufsize=1,
            env=environment,
        )

        try:
            wait_for_line(proc, "READY")

            proc.stdin.write("GO\n")
            proc.stdin.flush()

            wait_for_line(proc, "STARTED", seconds=15)

            sample_start_ns = time.monotonic_ns()

            with power_path.open("w") as power_out, \
                    power_err.open("w") as power_stderr:
                sampled = subprocess.run(
                    [
                        "sudo", "-n", pm,
                        "-i", "1000",
                        "-n", str(POWER_SAMPLES),
                        "-s", "cpu_power,gpu_power",
                    ],
                    stdout=power_out,
                    stderr=power_stderr,
                    timeout=25,
                    check=False
                )

            sample_end_ns = time.monotonic_ns()

            assert sampled.returncode == 0, (
                power_err.read_text(errors="replace")
            )

            proc.communicate(timeout=60)
            assert proc.returncode == 0

        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise

    worker_data = json.loads(result_path.read_text())
    samples = parse_power(
        power_path.read_text(errors="replace")
    )

    assert len(samples) >= 8, (
        f"Insufficient power samples: {len(samples)}"
    )

    # Exclude first/last sampling periods.
    selected = samples[1:-1]

    # Central wall-time interval, approximately aligned
    # with the retained power sampling periods.
    central_start_ns = sample_start_ns + 1_000_000_000
    central_end_ns = sample_end_ns - 1_000_000_000

    assert central_end_ns > central_start_ns
    assert worker_data[
        "active_start_monotonic_ns"
    ] <= sample_start_ns

    assert worker_data[
        "active_end_monotonic_ns"
    ] >= sample_end_ns

    completed = sum(
        central_start_ns <= t <= central_end_ns
        for t in worker_data["completion_timestamps_ns"]
    )

    interval_seconds = (
        central_end_ns - central_start_ns
    ) / 1e9

    assert completed >= 1

    throughput = completed / interval_seconds

    average_power = {
        component: statistics.mean(
            sample[component]
            for sample in selected
        )
        for component in (
            "cpu", "gpu", "ane", "combined"
        )
    }

    energy_per_image = (
        average_power["combined"] / throughput
    )

    assert math.isfinite(energy_per_image)
    assert energy_per_image > 0

    summary = {
        "protocol": "T20_4_load_power_pilot",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model": "resnet50",
        "mapping": mapping_name,
        "repeat": repeat,
        "power_scope":
            "powermetrics_estimated_cpu_gpu_ane",
        "power_is_whole_system": False,
        "power_samples_observed": len(samples),
        "power_samples_used": len(selected),
        "average_power_watts": average_power,
        "completed_in_central_window": completed,
        "central_window_seconds": interval_seconds,
        "throughput_images_per_second": throughput,
        "estimated_energy_joules_per_image":
            energy_per_image,
        "measurement_status":
            "exploratory_approximately_aligned",
        "idle_power_subtracted": False,
        "instrument_overhead_subtracted": False,
        "raw_power_sha256": sha(power_path),
        "worker_result_sha256": sha(result_path),
        "existing_database_modified": False,
    }

    save(run_dir / "summary.json", summary)

    print(
        f"{mapping_name:18s} repeat={repeat} "
        f"throughput={throughput:.2f} img/s "
        f"power={average_power['combined']:.3f} W "
        f"energy={energy_per_image:.4f} J/img "
        f"samples={len(selected)}",
        flush=True
    )

    return summary

def main():
    assert DB.is_file()
    assert T05.is_file()
    assert IMAGES.is_file()

    db = sqlite3.connect(
        f"file:{DB}?mode=ro", uri=True
    )
    try:
        assert db.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"
        assert db.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0] == 327336
        assert db.execute(
            "SELECT COUNT(*) FROM tasks "
            "WHERE state='completed'"
        ).fetchone()[0] == 6
    finally:
        db.close()

    assert shutil.disk_usage(OUT).free > 5 * 1024**3, (
        "Less than 5 GiB free disk space"
    )

    pm = shutil.which("powermetrics")
    assert pm

    print("=== T20.4 ACTIVE POWER PILOT ===")
    print("DATABASE READ-ONLY: PASS")
    print("WORKER PROCESSES: 6")
    print("MEASUREMENT MODE: sustained inference")
    print("POWER SOURCE: powermetrics CPU+GPU+ANE")
    print()

    summaries = []

    for name, repeat in SEQUENCE:
        summaries.append(
            run_one(name, repeat, pm)
        )

    save(OUT / "pilot_summary.json", {
        "protocol": "T20_4_load_power_pilot",
        "runs": summaries,
        "number_of_independent_worker_processes": 6,
        "statistical_status": "pilot_only",
        "power_alignment": (
            "Approximate centered-window alignment; "
            "not per-sample timestamp synchronized."
        ),
        "energy_comparison_scope": (
            "Within same Apple M4 powermetrics "
            "CPU+GPU+ANE boundary only"
        ),
        "new_sqlite_rows": 0,
    })

    files = [
        p for p in OUT.rglob("*")
        if p.is_file()
        and p.name != "checksums.json"
    ]

    checks = {
        str(p.relative_to(OUT)): sha(p)
        for p in sorted(files)
    }

    save(OUT / "checksums.json", checks)

    for relative, expected in checks.items():
        assert sha(OUT / relative) == expected

    print()
    print("INDEPENDENT WORKER RUNS: 6/6")
    print("POWER OUTPUT CAPTURE: PASS")
    print("NUMERICAL VALIDATION: PASS")
    print("ARTIFACT SHA256: PASS")
    print("EXISTING SQLITE MODIFIED: NO")
    print("T20.4 ENERGY+THROUGHPUT PILOT: PASS")
    print("RESULT DIRECTORY:", OUT)

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[2] == "worker":
        assert len(sys.argv) == 5
        worker(sys.argv[3], Path(sys.argv[4]))
    else:
        assert OUT is not None
        main()
