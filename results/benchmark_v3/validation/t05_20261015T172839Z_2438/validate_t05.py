import csv
import hashlib
import json
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import torch
import torchvision
import PIL
from PIL import Image
from torch import nn
from torchvision.models import (
    resnet50, convnext_base, convnext_large,
    ResNet50_Weights, ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)

outdir = Path(sys.argv[1])
root = Path("data/imagenetv2/imagenetv2-matched-frequency-format-val")
assert root.is_dir(), "ImageNetV2 folder not found"
assert torch.backends.mps.is_available(), "MPS unavailable"

torch.set_num_threads(4)
torch.manual_seed(2026)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def digest(obj):
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()

def git(*args):
    return subprocess.check_output(
        ["git", *args], text=True
    ).strip()

models = [
    ("resnet50", resnet50, ResNet50_Weights.IMAGENET1K_V2,
     "resnet50__stage6.json",
     "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"),
    ("convnext_base", convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1,
     "convnext_base__fine41.json",
     "6075fbad31c688f9ed013922a59daf0fc2f4dc12e6e43bc01fa840c7bd1ca2cd"),
    ("convnext_large", convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1,
     "convnext_large__fine41.json",
     "ea097f822cdb2cab40905cec3949f4bdf6d6bf1d546c3ec5e43f0b1c6275e144"),
]

files = sorted(
    p for p in root.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)
assert len(files) >= 16, "Not enough images"

step = len(files) // 16
selected = [files[i * step] for i in range(16)]
images_info = [
    {"path": str(p.relative_to(root)), "sha256": sha(p)}
    for p in selected
]
image_set_hash = digest(images_info)

session_id = uuid.uuid4().hex

meta = {
    "protocol": "T05_V3_pretrained_numerical_validation",
    "session_id": session_id,
    "process_independent_repeats": 1,
    "input_set_sha256": image_set_hash,
    "images": images_info,
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
    "python": sys.version.split()[0],
    "macos": platform.mac_ver()[0],
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "pillow": PIL.__version__,
    "mps_available": True,
    "mps_fallback": "disabled",
    "cpu_threads": 4,
    "precision": "float32",
    "batch_size": 1,
    "atol": 1e-3,
    "rtol": 1e-3,
    "models": {},
}

def make_groups(m, name):
    if name == "resnet50":
        return [
            nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool),
            m.layer1, m.layer2, m.layer3, m.layer4,
            nn.Sequential(m.avgpool, nn.Flatten(1), m.fc),
        ]
    return [
        m.features[0],
        m.features[1],
        nn.Sequential(m.features[2], m.features[3]),
        nn.Sequential(m.features[4], m.features[5]),
        nn.Sequential(m.features[6], m.features[7]),
        nn.Sequential(m.avgpool, m.classifier),
    ]

def forward(groups, devices, x):
    y = x
    for group, device in zip(groups, devices):
        y = group(y.to(device))
    if y.device.type == "mps":
        torch.mps.synchronize()
    return y.cpu()

def compare(ref, actual):
    if actual.shape != ref.shape:
        return False, None, None, -1, False
    if not bool(torch.isfinite(actual).all()):
        return False, None, None, -1, False

    diff = (ref - actual).abs()
    limits = 1e-3 + 1e-3 * ref.abs()
    violations = int((diff > limits).sum().item())
    ok = violations == 0
    top1 = ref.argmax().item() == actual.argmax().item()
    return ok, diff.max().item(), diff.mean().item(), violations, top1

summary_path = outdir / "validation_summary.csv"
samples_path = outdir / "validation_samples.csv"

all_pass = True
comparison_count = 0

with summary_path.open("w", newline="") as sf, \
     samples_path.open("w", newline="") as rf:

    sw = csv.writer(sf)
    rw = csv.writer(rf)

    sw.writerow([
        "session_id", "run_id", "config_id", "model",
        "mapping", "cut_after", "samples", "passed",
        "top1_matches", "max_abs_error", "status", "error"
    ])

    rw.writerow([
        "run_id", "config_id", "model", "mapping",
        "cut_after", "sample_index", "image_path",
        "image_sha256", "max_abs_error", "mean_abs_error",
        "tolerance_violations", "top1_agreement",
        "numerical_pass"
    ])

    for name, factory, weights, manifest_file, expected_sha in models:
        print(f"\nMODEL: {name}", flush=True)

        checkpoint = (
            Path(torch.hub.get_dir()) / "checkpoints" /
            Path(urlparse(weights.url).path).name
        )
        assert checkpoint.is_file(), f"Missing weights: {checkpoint}"
        weight_sha = sha(checkpoint)
        assert weight_sha == expected_sha, f"Weight SHA mismatch: {name}"

        manifest = Path("configs/partition_manifests_v2") / manifest_file
        assert manifest.is_file(), f"Missing manifest: {manifest}"
        manifest_sha = sha(manifest)

        model = factory(weights=None).eval().cpu()
        state = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        del state

        groups = make_groups(model, name)
        transforms = weights.transforms()

        inputs = []
        for path in selected:
            with Image.open(path) as im:
                inputs.append(
                    transforms(im.convert("RGB")).unsqueeze(0)
                )

        with torch.inference_mode():
            refs = [model(x).detach().clone() for x in inputs]

        assert all(
            tuple(y.shape) == (1, 1000)
            and bool(torch.isfinite(y).all())
            for y in refs
        ), "Invalid CPU reference"

        meta["models"][name] = {
            "weight_enum": weights.name,
            "weight_sha256": weight_sha,
            "source_manifest": str(manifest),
            "source_manifest_sha256": manifest_sha,
            "stage_grouping": "six_canonical_stages",
            "preprocessing": str(transforms),
        }

        layouts = [
            ("FULL_CPU", None, ["cpu"] * 6),
            ("FULL_MPS", None, ["mps"] * 6),
        ]

        for cut in range(1, 6):
            layouts.append((
                "CPU_TO_MPS", cut,
                ["cpu"] * cut + ["mps"] * (6-cut)
            ))
            layouts.append((
                "MPS_TO_CPU", cut,
                ["mps"] * cut + ["cpu"] * (6-cut)
            ))

        for mapping, cut, devices in layouts:
            config = {
                "model": name,
                "weight_sha256": weight_sha,
                "manifest_sha256": manifest_sha,
                "grouping": "six_canonical_stages",
                "hardware": "mac_m4",
                "input_set": image_set_hash,
                "precision": "fp32",
                "cpu_threads": 4 if "cpu" in devices else None,
                "mapping": mapping,
                "devices": devices,
                "cut_after": cut,
                "mode": "sequential_numerical_validation",
                "torch": torch.__version__,
                "atol": 1e-3,
                "rtol": 1e-3,
            }
            config_id = "cfg_" + digest(config)
            run_id = "run_" + uuid.uuid4().hex

            passed = 0
            top1_matches = 0
            max_error = 0.0
            error_message = ""

            try:
                for group, device in zip(groups, devices):
                    group.to(device)

                with torch.inference_mode():
                    for index, (x, ref) in enumerate(zip(inputs, refs)):
                        actual = forward(groups, devices, x)
                        ok, err, mean_err, violations, top1 = compare(
                            ref, actual
                        )

                        passed += int(ok)
                        top1_matches += int(top1)

                        if err is not None:
                            max_error = max(max_error, err)

                        rw.writerow([
                            run_id, config_id, name, mapping,
                            cut, index, images_info[index]["path"],
                            images_info[index]["sha256"],
                            err, mean_err, violations,
                            int(top1), int(ok)
                        ])

                status = "PASS" if passed == 16 else "FAIL"

            except Exception as exc:
                status = "ERROR"
                error_message = repr(exc)

            if status != "PASS":
                all_pass = False

            sw.writerow([
                session_id, run_id, config_id, name,
                mapping, cut, 16, passed, top1_matches,
                max_error, status, error_message
            ])

            sf.flush()
            rf.flush()

            comparison_count += passed

            print(
                f"  {mapping} cut={cut}: "
                f"{passed}/16 {status} "
                f"max_error={max_error:.8f}",
                flush=True
            )

        model.cpu()
        del model, groups, refs, inputs
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

metadata_path = outdir / "metadata.json"
metadata_path.write_text(
    json.dumps(meta, indent=2, default=str)
)

artifacts = [
    outdir / "validate_t05.py",
    summary_path,
    samples_path,
    metadata_path,
]

checksums = {
    p.name: sha(p) for p in artifacts
}

(outdir / "checksums.json").write_text(
    json.dumps(checksums, indent=2)
)

print("\n==============================")
print("T05.6 FINAL STATUS:", "PASS" if all_pass else "FAIL")
print("Successful image comparisons:", comparison_count, "/ 576")
print("Result directory:", outdir)
print("==============================")

if not all_pass:
    sys.exit(1)
