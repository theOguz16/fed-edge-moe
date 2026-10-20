import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"
CAMPAIGN = BASE / "production_campaigns/g2_cut_graph_mac"
OUT = CAMPAIGN / "vision_pilot_v1"
T05 = BASE / "validation/t05_20261015T173344Z_2438/metadata.json"
HOLDOUT = BASE / "holdout_thread_confirmation/t12_20261015T192923Z_2438/holdout_images.json"
IMAGES = ROOT / "data/imagenetv2/imagenetv2-matched-frequency-format-val"
MANIFESTS = ROOT / "configs/partition_manifests_v2"

CASES = [
    ("resnet50", 2, "fp32"),
    ("resnet50", 2, "fp16"),
    ("convnext_base", 20, "fp32"),
    ("convnext_base", 20, "fp16"),
    ("convnext_large", 20, "fp32"),
    ("convnext_large", 20, "fp16"),
]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def save(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, indent=2) + "\n")

def child(model_name, cut, precision, dest):
    import torch
    from torch import nn
    from PIL import Image
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        convnext_base, ConvNeXt_Base_Weights,
        convnext_large, ConvNeXt_Large_Weights,
    )
    from urllib.parse import urlparse

    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(2026)

    constructors = {
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2, "resnet50__stage6.json"),
        "convnext_base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, "convnext_base__fine41.json"),
        "convnext_large": (convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1, "convnext_large__fine41.json"),
    }

    constructor, weights, manifest_name = constructors[model_name]
    checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    assert checkpoint.is_file(), "Missing cached model weights"
    expected = json.loads(T05.read_text())["models"][model_name]["weight_sha256"]
    assert sha(checkpoint) == expected, "Checkpoint hash mismatch"

    model = constructor(weights=None).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state

    records = json.loads(HOLDOUT.read_text())
    assert len(records) >= 2
    tensors = []
    hashes = []

    for item in records[:2]:
        path = IMAGES / item["path"]
        assert sha(path) == item["sha256"]
        with Image.open(path) as image:
            tensors.append(weights.transforms()(image.convert("RGB")))
        hashes.append(item["sha256"])

    inputs = torch.stack(tensors)

    with torch.inference_mode():
        reference = model(inputs).detach().cpu()
    assert bool(torch.isfinite(reference).all())

    manifest_path = MANIFESTS / manifest_name
    manifest = json.loads(manifest_path.read_text())
    groups = manifest["groups"]

    if model_name == "resnet50":
        stages = [
            nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.Sequential(model.avgpool, nn.Flatten(1), model.fc),
        ]
        assert len(stages) == 6
    else:
        assert len(groups) == 41
        stages = [
            nn.Sequential(*[
                model.get_submodule(path) for path in group["modules"]
            ])
            for group in groups
        ]

    assert len(stages) == len(groups)
    assert 0 < cut < len(stages)

    dtype = torch.float32 if precision == "fp32" else torch.float16

    for index, stage in enumerate(stages):
        stage.to(
            device="cpu" if index < cut else "mps",
            dtype=dtype,
        )

    with torch.inference_mode():
        x = inputs.to(dtype=dtype)

        for index, stage in enumerate(stages):
            device = "cpu" if index < cut else "mps"
            x = x.to(device)
            x = stage(x)

        if x.device.type == "mps":
            torch.mps.synchronize()
        actual = x.float().cpu()

    finite = bool(torch.isfinite(actual).all())
    shape_match = actual.shape == reference.shape

    if finite and shape_match:
        max_abs = float((actual - reference).abs().max())
        top1_agreement = float(
            (actual.argmax(1) == reference.argmax(1))
            .float().mean()
        )
        allclose = bool(torch.allclose(
            actual, reference, rtol=1e-3, atol=1e-3
        ))
    else:
        max_abs = None
        top1_agreement = None
        allclose = False

    save(dest, {
        "protocol": "G2_VISION_NUMERICAL_PILOT_V1",
        "model": model_name,
        "precision": precision,
        "cut_after_stage": cut,
        "direction": "cpu_mps",
        "model_stage_count": len(stages),
        "manifest_sha256": sha(manifest_path),
        "checkpoint_sha256": sha(checkpoint),
        "input_sha256_list": hashes,
        "reference_dtype": "float32",
        "output_finite": finite,
        "output_shape_match": shape_match,
        "top1_agreement": top1_agreement,
        "max_abs_logit_error": max_abs,
        "strict_allclose_1e_3": allclose,
        "numerical_status": (
            "numerical_pass"
            if finite and shape_match and allclose
            and top1_agreement == 1.0
            else "numerical_mismatch"
        ),
        "throughput_measured": False,
        "power_measured": False,
        "energy_measured": False,
    })

def parent():
    assert T05.is_file()
    assert HOLDOUT.is_file()
    assert (CAMPAIGN / "manifest.json").is_file()
    assert not OUT.exists(), "Pilot already exists; refusing overwrite"
    OUT.mkdir(parents=True)

    summary = []

    for index, (model, cut, precision) in enumerate(CASES, 1):
        label = f"{model}_cut{cut}_{precision}"
        result = OUT / (label + ".json")
        stdout = OUT / (label + ".stdout.txt")
        stderr = OUT / (label + ".stderr.txt")

        print(f"[{index}/6] {label}", flush=True)

        with stdout.open("w") as out, stderr.open("w") as err:
            try:
                proc = subprocess.run(
                    [
                        sys.executable, "-u", str(Path(__file__).resolve()),
                        "child", model, str(cut), precision, str(result),
                    ],
                    env={
                        **os.environ,
                        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                    },
                    stdout=out,
                    stderr=err,
                    timeout=900,
                )
                code = proc.returncode
                error = ""
            except subprocess.TimeoutExpired:
                code = 124
                error = "Worker timed out"

        if result.is_file() and code == 0:
            record = json.loads(result.read_text())
            status = record["numerical_status"]
            agreement = record["top1_agreement"]
            difference = record["max_abs_logit_error"]
        else:
            status = "execution_failed"
            agreement = None
            difference = None
            error = error or stderr.read_text(errors="replace")[-1500:]

        summary.append({
            "model": model,
            "cut": cut,
            "precision": precision,
            "status": status,
            "top1_agreement": agreement,
            "max_abs_logit_error": difference,
            "exit_code": code,
            "error": error,
            "result_path": str(result) if result.is_file() else None,
            "result_sha256": sha(result) if result.is_file() else None,
        })

        print(
            "STATUS:", status,
            "TOP1:", agreement,
            "MAX_ABS:", difference,
            flush=True,
        )
        if error:
            print("ERROR:", error[-400:], flush=True)

    save(OUT / "summary.json", {
        "protocol": "G2_VISION_NUMERICAL_PILOT_V1",
        "cases": summary,
        "source_worker_sha256": sha(Path(__file__).resolve()),
        "notes": [
            "Two ImageNetV2 holdout images per configuration",
            "CPU->MPS cut only in this pilot",
            "Strict allclose and task-level top1 agreement are separate",
            "No speed or energy measurements",
            "Original T23-T26 SQLite and result files are untouched",
        ],
    })

    print("\nVISION PILOT: FINISHED")
    print("STATUSES:", {s: sum(r["status"] == s for r in summary)
                        for s in sorted({r["status"] for r in summary})})
    print("OUTPUT:", OUT)
    print("SQLITE MODIFIED: NO")
    print("BENCHMARK RERUN: NO")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        parent()
    else:
        assert sys.argv[1] == "child"
        try:
            child(
                sys.argv[2],
                int(sys.argv[3]),
                sys.argv[4],
                Path(sys.argv[5]),
            )
        except BaseException:
            traceback.print_exc()
            raise
