import hashlib
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
BASE = ROOT / "results/benchmark_v3"
G4 = BASE / "production_campaigns/g4_campaign_mac_v1"
OUT = G4 / "vision_fp16_allcuts_v1"
IMAGES = ROOT / "data/imagenetv2/imagenetv2-matched-frequency-format-val"
MANIFESTS = ROOT / "configs/partition_manifests_v2"
METADATA = BASE / "validation/t05_20261015T173344Z_2438/metadata.json"
GRAPH = G4 / "g4_graph_merged_mac_v1.sqlite"

PROTOCOL = "G4_VISION_FP16_ALLCUTS_V1"
COUNT = 512
SEED = 2026

MODELS = {
    "resnet50": {
        "cuts": 5,
        "manifest": "resnet50__stage6.json",
    },
    "convnext_base": {
        "cuts": 40,
        "manifest": "convnext_base__fine41.json",
    },
    "convnext_large": {
        "cuts": 40,
        "manifest": "convnext_large__fine41.json",
    },
}
DIRECTIONS = ("cpu_mps", "mps_cpu")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path, obj):
    assert not path.exists(), path
    temp = path.with_name(path.name + ".tmp")
    assert not temp.exists(), temp
    with temp.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def selection():
    path = OUT / "selection.json"
    if path.is_file():
        obj = json.loads(path.read_text())
        assert obj["protocol"] == PROTOCOL
        assert obj["seed"] == SEED
        assert len(obj["images"]) == COUNT
        for item in obj["images"]:
            p = IMAGES / item["path"]
            assert p.is_file() and sha(p) == item["sha256"]
        return obj

    assert IMAGES.is_dir(), IMAGES
    files = sorted(
        p for p in IMAGES.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and p.parent.name.isdigit()
    )
    assert len(files) >= COUNT, len(files)

    chosen = sorted(random.Random(SEED).sample(files, COUNT))
    obj = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "dataset": "imagenetv2-matched-frequency",
        "sample_count": COUNT,
        "images": [
            {
                "path": str(p.relative_to(IMAGES)),
                "label": int(p.parent.name),
                "sha256": sha(p),
            }
            for p in chosen
        ],
    }
    assert all(0 <= x["label"] < 1000 for x in obj["images"])
    atomic_json(path, obj)
    return obj

def model_components(name):
    import torch
    from torch import nn
    from torchvision.models import (
        resnet50, ResNet50_Weights,
        convnext_base, ConvNeXt_Base_Weights,
        convnext_large, ConvNeXt_Large_Weights,
    )
    from urllib.parse import urlparse

    choices = {
        "resnet50": (
            resnet50, ResNet50_Weights.IMAGENET1K_V2,
        ),
        "convnext_base": (
            convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1,
        ),
        "convnext_large": (
            convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1,
        ),
    }
    factory, weights = choices[name]

    checkpoint = (
        Path(torch.hub.get_dir()) / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )
    assert checkpoint.is_file(), checkpoint

    expected = json.loads(METADATA.read_text())[
        "models"
    ][name]["weight_sha256"]
    assert sha(checkpoint) == expected, name

    model = factory(weights=None).eval().cpu()
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    del state

    manifest = MANIFESTS / MODELS[name]["manifest"]
    assert manifest.is_file()
    groups = json.loads(manifest.read_text())["groups"]

    if name == "resnet50":
        stages = [
            nn.Sequential(
                model.conv1, model.bn1,
                model.relu, model.maxpool,
            ),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            nn.Sequential(
                model.avgpool, nn.Flatten(1), model.fc
            ),
        ]
    else:
        stages = [
            nn.Sequential(
                *[model.get_submodule(p) for p in group["modules"]]
            )
            for group in groups
        ]

    assert len(stages) == len(groups)
    assert len(stages) == MODELS[name]["cuts"] + 1

    provenance = {
        "checkpoint_sha256": sha(checkpoint),
        "partition_manifest_sha256": sha(manifest),
        "stage_count": len(stages),
    }
    return model, stages, weights.transforms(), provenance

def inputs_for(transform, selected):
    import torch
    from PIL import Image

    inputs = []
    for item in selected:
        path = IMAGES / item["path"]
        assert sha(path) == item["sha256"]
        with Image.open(path) as image:
            inputs.append(transform(image.convert("RGB")))
    assert len(inputs) == COUNT
    return inputs

def compute_reference(name):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    selected = selection()["images"]
    model, stages, transform, provenance = model_components(name)
    inputs = inputs_for(transform, selected)

    logits = []
    with torch.inference_mode():
        for i in range(0, COUNT, 2):
            batch = torch.stack(inputs[i:i+2])
            y = model(batch).detach().cpu().float()
            assert y.shape == (len(batch), 1000)
            assert bool(torch.isfinite(y).all())
            logits.append(y)

    reference = torch.cat(logits, dim=0)
    assert reference.shape == (COUNT, 1000)

    record = {
        "protocol": PROTOCOL,
        "model": name,
        "selection_sha256": sha(OUT / "selection.json"),
        "model_provenance": provenance,
        "logits": reference,
    }
    target = OUT / "references" / f"{name}.pt"
    temp = target.with_suffix(".pt.tmp")
    assert not target.exists()
    assert not temp.exists()
    torch.save(record, temp)
    os.replace(temp, target)

    print(
        "REFERENCE COMPLETE", name,
        "shape=", tuple(reference.shape),
        "sha256=", sha(target),
        flush=True,
    )

def verify_reference(name):
    import torch

    path = OUT / "references" / f"{name}.pt"
    assert path.is_file()
    data = torch.load(
        path, map_location="cpu", weights_only=True
    )
    assert data["protocol"] == PROTOCOL
    assert data["model"] == name
    assert data["selection_sha256"] == sha(
        OUT / "selection.json"
    )
    tensor = data["logits"]
    assert tensor.shape == (COUNT, 1000)
    assert bool(torch.isfinite(tensor).all())
    return data, sha(path)

def evaluate(name, cut, direction):
    import torch

    assert torch.backends.mps.is_available()
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"

    torch.set_num_threads(4)
    torch.manual_seed(SEED)

    selected = selection()["images"]
    ref_obj, reference_sha = verify_reference(name)
    reference = ref_obj["logits"]

    model, stages, transform, provenance = model_components(name)
    assert provenance == ref_obj["model_provenance"]

    first, second = (
        ("cpu", "mps")
        if direction == "cpu_mps"
        else ("mps", "cpu")
    )

    for i, stage in enumerate(stages):
        stage.to(
            device=first if i < cut else second,
            dtype=torch.float16,
        )

    inputs = inputs_for(transform, selected)

    predictions = []
    activation_shape = None
    activation_bytes = None
    full_logits = []

    with torch.inference_mode():
        for start in range(0, COUNT, 2):
            x = torch.stack(inputs[start:start+2])
            x = x.to(first, dtype=torch.float16)

            for stage in stages[:cut]:
                x = stage(x)

            current_shape = list(x.shape)
            current_bytes = x.numel() * x.element_size()

            if activation_shape is None:
                activation_shape = current_shape
                activation_bytes = current_bytes
            else:
                assert current_shape == activation_shape
                assert current_bytes == activation_bytes

            x = x.to(second, copy=True)

            for stage in stages[cut:]:
                x = stage(x)

            if x.device.type == "mps":
                torch.mps.synchronize()

            y = x.float().cpu()
            assert y.shape == (len(y), 1000)
            full_logits.append(y)

    actual = torch.cat(full_logits, dim=0)
    assert actual.shape == reference.shape

    finite = torch.isfinite(actual).all(dim=1)
    strict = torch.isclose(
        actual, reference, atol=1e-3, rtol=1e-3
    ).all(dim=1)

    reference_predictions = reference.argmax(dim=1)
    candidate_predictions = actual.argmax(dim=1)

    finite_count = 0
    agreement_count = 0
    reference_correct = 0
    candidate_correct = 0
    strict_count = 0
    both_correct = 0
    max_abs_error = 0.0
    details = []

    for i, item in enumerate(selected):
        is_finite = bool(finite[i])
        label = item["label"]

        ref_pred = int(reference_predictions[i])
        candidate = (
            int(candidate_predictions[i]) if is_finite else None
        )
        agree = is_finite and candidate == ref_pred

        ref_ok = ref_pred == label
        candidate_ok = is_finite and candidate == label

        strict_ok = is_finite and bool(strict[i])
        error = (
            float((actual[i] - reference[i]).abs().max())
            if is_finite else None
        )

        finite_count += int(is_finite)
        agreement_count += int(agree)
        reference_correct += int(ref_ok)
        candidate_correct += int(candidate_ok)
        strict_count += int(strict_ok)
        both_correct += int(ref_ok and candidate_ok)

        if error is not None:
            max_abs_error = max(max_abs_error, error)

        details.append({
            "image": item["path"],
            "image_sha256": item["sha256"],
            "label": label,
            "finite": is_finite,
            "fp32_top1": ref_pred,
            "fp16_top1": candidate,
            "top1_agreement": agree,
            "fp32_correct": ref_ok,
            "fp16_correct": candidate_ok,
            "strict_allclose_1e_3": strict_ok,
            "max_abs_logit_error": error,
        })

    return {
        "protocol": PROTOCOL,
        "model": name,
        "direction": direction,
        "cut_after_stage": cut,
        "stage_count": len(stages),
        "first_device": first,
        "second_device": second,
        "precision": "fp16",
        "reference_precision": "fp32",
        "reference_device": "cpu",
        "sample_count": COUNT,
        "selection_sha256": sha(OUT / "selection.json"),
        "reference_file_sha256": reference_sha,
        "model_provenance": provenance,
        "activation_shape_per_microbatch": activation_shape,
        "logical_activation_bytes_per_microbatch": activation_bytes,
        "finite_count": finite_count,
        "top1_agreement_count": agreement_count,
        "strict_allclose_image_count": strict_count,
        "fp32_correct_count": reference_correct,
        "fp16_correct_count": candidate_correct,
        "both_correct_count": both_correct,
        "fp32_top1_accuracy": reference_correct / COUNT,
        "fp16_top1_accuracy": candidate_correct / COUNT,
        "accuracy_delta": (
            candidate_correct - reference_correct
        ) / COUNT,
        "max_abs_logit_error": max_abs_error,
        "images": details,
        "quality_acceptable": None,
        "quality_acceptance_threshold": None,
        "full_imagenetv2_accuracy_established": False,
        "power_measured": False,
        "throughput_measured": False,
        "gpu_kernel_overlap_proven": False,
    }

def checked_completed(path, name, cut, direction, selection_sha):
    if not path.exists():
        return False

    obj = json.loads(path.read_text())
    assert obj["protocol"] == PROTOCOL
    assert obj["model"] == name
    assert obj["direction"] == direction
    assert obj["cut_after_stage"] == cut
    assert obj["sample_count"] == COUNT
    assert obj["selection_sha256"] == selection_sha
    assert len(obj["images"]) == COUNT

    expected_reference_sha = sha(
        OUT / "references" / f"{name}.pt"
    )
    assert obj["reference_file_sha256"] == expected_reference_sha

    for i, row in enumerate(obj["images"]):
        assert row["image"] == selection()["images"][i]["path"]

    return True

def check_graph():
    assert GRAPH.is_file(), GRAPH
    with sqlite3.connect(
        f"file:{GRAPH}?mode=ro", uri=True
    ) as con:
        assert con.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0] == "ok"

        for name, spec in MODELS.items():
            cuts = con.execute("""
                SELECT cut_after_stage
                FROM cut_boundaries
                WHERE model_id=?
                ORDER BY cut_after_stage
            """, (name,)).fetchall()
            actual = [x[0] for x in cuts]
            expected = list(range(1, spec["cuts"] + 1))
            assert actual == expected, (name, actual, expected)

def run_child(args, folder, timeout):
    attempt = 1
    while (folder / f"attempt_{attempt:03d}").exists():
        attempt += 1

    attempt_path = folder / f"attempt_{attempt:03d}"
    attempt_path.mkdir(parents=True)

    result = attempt_path / "result.json"
    env = dict(os.environ)
    env.update({
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })

    with (attempt_path / "stdout.txt").open("x") as stdout, \
         (attempt_path / "stderr.txt").open("x") as stderr:
        try:
            process = subprocess.run(
                [
                    sys.executable, "-u",
                    str(Path(__file__).resolve()),
                    *args, str(result),
                ],
                stdout=stdout,
                stderr=stderr,
                env=env,
                timeout=timeout,
            )
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = 124

    if code != 0 or not result.is_file():
        print(
            "FAILED:", args, "exit=", code,
            "logs=", attempt_path,
            flush=True,
        )
        print(
            (attempt_path / "stderr.txt")
            .read_text(errors="replace")[-2500:],
            flush=True,
        )
        raise SystemExit(1)

    return result

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "reference":
        assert len(sys.argv) == 4
        name = sys.argv[2]
        assert name in MODELS
        compute_reference(name)
        Path(sys.argv[3]).write_text("reference complete\n")
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "case":
        assert len(sys.argv) == 6
        name = sys.argv[2]
        cut = int(sys.argv[3])
        direction = sys.argv[4]
        assert name in MODELS
        assert 1 <= cut <= MODELS[name]["cuts"]
        assert direction in DIRECTIONS
        result = evaluate(name, cut, direction)
        atomic_json(Path(sys.argv[5]), result)
        return

    assert len(sys.argv) == 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "references").mkdir(exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)

    check_graph()
    selected = selection()
    selection_sha = sha(OUT / "selection.json")

    print("=== G4.19 VISION ALL CUTS ===", flush=True)
    print("PROTOCOL:", PROTOCOL, flush=True)
    print("IMAGES:", COUNT, flush=True)
    print("SELECTION SHA256:", selection_sha, flush=True)
    print("TOTAL CASES:", 170, flush=True)
    print("SOURCE GRAPH:", GRAPH, flush=True)

    completed = 0

    for name, spec in MODELS.items():
        ref = OUT / "references" / f"{name}.pt"

        if ref.exists():
            verify_reference(name)
            print("REFERENCE VERIFIED:", name, flush=True)
        else:
            folder = OUT / "references" / f"{name}_attempts"
            folder.mkdir(exist_ok=True)
            run_child(["reference", name], folder, timeout=7200)
            verify_reference(name)

        for cut in range(1, spec["cuts"] + 1):
            for direction in DIRECTIONS:
                label = f"{name}_cut{cut:02d}_{direction}"
                folder = OUT / "runs" / label
                folder.mkdir(exist_ok=True)

                final = folder / "result.json"
                if checked_completed(
                    final, name, cut, direction, selection_sha
                ):
                    completed += 1
                    print(
                        f"[{completed}/170] SKIP VERIFIED:",
                        label, flush=True
                    )
                    continue

                print(
                    f"[{completed+1}/170] RUN:",
                    label, flush=True
                )

                candidate = run_child(
                    ["case", name, str(cut), direction],
                    folder,
                    timeout=7200,
                )

                obj = json.loads(candidate.read_text())
                assert obj["protocol"] == PROTOCOL
                assert obj["model"] == name
                assert obj["cut_after_stage"] == cut
                assert obj["direction"] == direction
                assert obj["sample_count"] == COUNT
                assert obj["selection_sha256"] == selection_sha

                # Hard-link the verified attempt file; do not rewrite it.
                os.link(candidate, final)
                assert checked_completed(
                    final, name, cut, direction, selection_sha
                )

                completed += 1
                print(
                    f"[{completed}/170] COMMITTED:",
                    label,
                    "FINITE:", obj["finite_count"],
                    "AGREEMENT:", obj["top1_agreement_count"],
                    "FP32_ACC:", round(
                        obj["fp32_top1_accuracy"], 6
                    ),
                    "FP16_ACC:", round(
                        obj["fp16_top1_accuracy"], 6
                    ),
                    "DELTA:", round(
                        obj["accuracy_delta"], 6
                    ),
                    "RESULT_SHA256:", sha(final),
                    flush=True,
                )

    assert completed == 170
    print("\n=== G4.19 FINAL ===")
    print("COMPLETED:", completed)
    print("TOTAL:", 170)
    print("QUALITY CAMPAIGN: COMPLETE")
    print("RESULTS:", OUT)
    print("OLD SQLITE MODIFIED: NO")
    print("G4.18 GRAPH SQLITE MODIFIED: NO")
    print("FP16 STRICT STATUSES RECLASSIFIED: NO")

if __name__ == "__main__":
    main()
