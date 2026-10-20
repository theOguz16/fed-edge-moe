import hashlib
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"
G4 = BASE / "production_campaigns/g4_campaign_mac_v1"
OUT = G4 / "vision_fp16_quality_v1"
IMAGES = ROOT / "data/imagenetv2/imagenetv2-matched-frequency-format-val"
MANIFESTS = ROOT / "configs/partition_manifests_v2"
METADATA = BASE / "validation/t05_20261015T173344Z_2438/metadata.json"

PROTOCOL = "G4_VISION_FP16_QUALITY_V1"
MODELS = ("resnet50", "convnext_base", "convnext_large")
DIRECTIONS = ("cpu_mps", "mps_cpu")
CUTS = {"resnet50": 2, "convnext_base": 20, "convnext_large": 20}
SAMPLE_COUNT = 64
SEED = 2026

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def save(path, data):
    assert not path.exists(), path
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def select_images():
    assert IMAGES.is_dir(), IMAGES
    paths = sorted(
        p for p in IMAGES.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".jpeg", ".jpg", ".png")
        and p.parent.name.isdigit()
    )
    assert len(paths) >= SAMPLE_COUNT, (
        "Insufficient labeled ImageNetV2 images", len(paths)
    )
    chosen = sorted(random.Random(SEED).sample(paths, SAMPLE_COUNT))
    rows = []
    for p in chosen:
        label = int(p.parent.name)
        assert 0 <= label < 1000, (p, label)
        rows.append({
            "path": str(p.relative_to(IMAGES)),
            "label": label,
            "sha256": sha(p),
        })
    return rows

def evaluate(name, direction, selection, selection_sha):
    import torch
    from torch import nn
    from PIL import Image
    from urllib.parse import urlparse
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        convnext_base, ConvNeXt_Base_Weights,
        convnext_large, ConvNeXt_Large_Weights,
    )

    assert torch.backends.mps.is_available()
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"
    torch.set_num_threads(4)
    torch.manual_seed(SEED)

    options = {
        "resnet50": (
            resnet50, ResNet50_Weights.IMAGENET1K_V2,
            "resnet50__stage6.json"
        ),
        "convnext_base": (
            convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1,
            "convnext_base__fine41.json"
        ),
        "convnext_large": (
            convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1,
            "convnext_large__fine41.json"
        ),
    }

    factory, weights, manifest_file = options[name]
    checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )
    assert checkpoint.is_file()

    expected = json.loads(
        METADATA.read_text()
    )["models"][name]["weight_sha256"]
    assert sha(checkpoint) == expected

    model = factory(weights=None).eval().cpu()
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    del state

    manifest_path = MANIFESTS / manifest_file
    groups = json.loads(manifest_path.read_text())["groups"]

    if name == "resnet50":
        stages = [
            nn.Sequential(model.conv1, model.bn1,
                          model.relu, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.Sequential(model.avgpool, nn.Flatten(1), model.fc),
        ]
    else:
        stages = [
            nn.Sequential(*[
                model.get_submodule(p) for p in group["modules"]
            ])
            for group in groups
        ]

    assert len(groups) == len(stages)
    cut = CUTS[name]
    first, second = (
        ("cpu", "mps")
        if direction == "cpu_mps"
        else ("mps", "cpu")
    )

    transform = weights.transforms()
    rows = []

    # Load each file once in this worker, hash-verified.
    inputs = []
    for item in selection:
        p = IMAGES / item["path"]
        assert sha(p) == item["sha256"]
        with Image.open(p) as image:
            inputs.append(transform(image.convert("RGB")))

    # Reference computed before modules move to different devices.
    reference = []
    with torch.inference_mode():
        for start in range(0, len(inputs), 2):
            x = torch.stack(inputs[start:start+2])
            reference.append(model(x).detach().float().cpu())
    reference = torch.cat(reference)
    assert bool(torch.isfinite(reference).all())

    for i, stage in enumerate(stages):
        stage.to(
            device=first if i < cut else second,
            dtype=torch.float16,
        )

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(inputs), 2):
            x = torch.stack(inputs[start:start+2])
            x = x.to(first, dtype=torch.float16)

            for stage in stages[:cut]:
                x = stage(x)

            x = x.to(second, copy=True)

            for stage in stages[cut:]:
                x = stage(x)

            if x.device.type == "mps":
                torch.mps.synchronize()

            outputs.append(x.float().cpu())

    actual = torch.cat(outputs)
    assert actual.shape == reference.shape

    finite_rows = torch.isfinite(actual).all(dim=1)
    agreement_rows = actual.argmax(-1) == reference.argmax(-1)
    strict_rows = torch.isclose(
        actual, reference, atol=1e-3, rtol=1e-3
    ).all(dim=1)

    valid = 0
    ref_correct = 0
    fp16_correct = 0
    changed_prediction = 0
    strict_count = 0

    for i, item in enumerate(selection):
        finite = bool(finite_rows[i])
        ref_pred = int(reference[i].argmax())
        pred = int(actual[i].argmax()) if finite else None
        label = item["label"]

        valid += int(finite)
        ref_correct += int(ref_pred == label)
        fp16_correct += int(pred == label) if finite else 0
        changed_prediction += int(pred != ref_pred)
        strict_count += int(bool(strict_rows[i])) if finite else 0

        rows.append({
            "image": item["path"],
            "image_sha256": item["sha256"],
            "label": label,
            "reference_top1": ref_pred,
            "candidate_top1": pred,
            "finite": finite,
            "top1_agreement": pred == ref_pred,
            "reference_correct": ref_pred == label,
            "candidate_correct": pred == label if finite else False,
            "strict_allclose_1e_3": (
                bool(strict_rows[i]) if finite else False
            ),
        })

    count = len(selection)
    return {
        "protocol": PROTOCOL,
        "model": name,
        "precision": "fp16",
        "reference": "full_cpu_fp32",
        "direction": direction,
        "cut_after_stage": cut,
        "stage_count": len(stages),
        "sample_count": count,
        "sample_seed": SEED,
        "sample_manifest_sha256": selection_sha,
        "checkpoint_sha256": sha(checkpoint),
        "partition_manifest_sha256": sha(manifest_path),
        "finite_count": valid,
        "strict_allclose_image_count": strict_count,
        "top1_agreement_count": count - changed_prediction,
        "reference_top1_accuracy": ref_correct / count,
        "fp16_top1_accuracy": fp16_correct / count,
        "top1_accuracy_delta": (
            fp16_correct - ref_correct
        ) / count,
        "changed_prediction_count": changed_prediction,
        "images": rows,
        "full_imagenetv2_accuracy_established": False,
        "pipeline_concurrency_measured": False,
        "power_measured": False,
        "source_worker": "G4_VISION_FP16_QUALITY_V1",
    }

def main():
    if len(sys.argv) == 5 and sys.argv[1] == "worker":
        name, direction = sys.argv[2:4]
        assert name in MODELS and direction in DIRECTIONS

        selection_path = OUT / "selection.json"
        selection = json.loads(selection_path.read_text())
        result = evaluate(
            name, direction, selection["images"],
            sha(selection_path),
        )
        save(Path(sys.argv[4]), result)
        return

    assert len(sys.argv) == 1, (
        "Usage: g4_vision_quality.py"
    )
    OUT.mkdir(parents=True, exist_ok=True)

    selection_file = OUT / "selection.json"
    if selection_file.exists():
        selection = json.loads(selection_file.read_text())
        assert selection["protocol"] == PROTOCOL
        assert selection["seed"] == SEED
        assert len(selection["images"]) == SAMPLE_COUNT
    else:
        selection = {
            "protocol": PROTOCOL,
            "seed": SEED,
            "images": select_images(),
        }
        save(selection_file, selection)

    print("G4.12 FP16 VISION QUALITY")
    print("IMAGES:", SAMPLE_COUNT)
    print("SAMPLE SHA256:", sha(selection_file), flush=True)

    for name in MODELS:
        for direction in DIRECTIONS:
            label = f"{name}_{direction}_fp16"
            folder = OUT / label
            folder.mkdir(exist_ok=True)

            result_path = folder / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                assert result["protocol"] == PROTOCOL
                assert result["model"] == name
                assert result["direction"] == direction
                assert result["sample_manifest_sha256"] == sha(selection_file)
                print("SKIP VERIFIED:", label, flush=True)
                continue

            attempt = 1
            while (folder / f"attempt_{attempt:03d}").exists():
                attempt += 1
            attempt_dir = folder / f"attempt_{attempt:03d}"
            attempt_dir.mkdir()

            candidate = attempt_dir / "result.json"
            env = {
                **os.environ,
                "PYTORCH_ENABLE_MPS_FALLBACK": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }

            print("RUN:", label, flush=True)

            with (attempt_dir / "stdout.txt").open("x") as out, \
                 (attempt_dir / "stderr.txt").open("x") as err:
                try:
                    proc = subprocess.run(
                        [
                            sys.executable, "-u",
                            str(Path(__file__).resolve()),
                            "worker", name, direction, str(candidate),
                        ],
                        env=env,
                        stdout=out,
                        stderr=err,
                        timeout=1800,
                    )
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    rc = 124

            if rc != 0 or not candidate.exists():
                print("FAILED:", label, "exit=", rc)
                print(
                    (attempt_dir / "stderr.txt")
                    .read_text(errors="replace")[-1600:]
                )
                raise SystemExit(1)

            result = json.loads(candidate.read_text())
            assert result["sample_count"] == SAMPLE_COUNT
            assert result["sample_manifest_sha256"] == sha(selection_file)
            os.link(candidate, result_path)

            print(
                "COMMITTED:", label,
                "FINITE:", result["finite_count"],
                "AGREEMENT:",
                result["top1_agreement_count"],
                "REF_ACC:",
                round(result["reference_top1_accuracy"], 4),
                "FP16_ACC:",
                round(result["fp16_top1_accuracy"], 4),
                "DELTA:",
                round(result["top1_accuracy_delta"], 4),
                flush=True,
            )

    print("\nG4.12 VISION QUALITY: PASS")
    print("RESULTS:", OUT)
    print("OLD SQLITE MODIFIED: NO")
    print("BENCHMARK PROTOCOL: FP16 QUALITY, NOT POWER")

if __name__ == "__main__":
    main()
