import gc
import json
import os
import re
import sys
from pathlib import Path

MANIFESTS = Path("configs/partition_manifests_v2")
OUTPUT = Path("results/benchmark_inventory_v1/mac27_validation.json")

results = []

def record(model, profile, status, detail):
    results.append({
        "model": model,
        "profile": profile,
        "status": status,
        "detail": detail,
    })
    print(f"{model:17s} {profile:21s} {status:17s} {detail}", flush=True)

def load(model, profile):
    p = MANIFESTS / f"{model}__{profile}.json"
    return json.loads(p.read_text())

def cache_config(repo):
    roots = []

    if os.getenv("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.getenv("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")

    roots.append(Path.home() / ".cache/huggingface/hub")

    folder = "models--" + repo.replace("/", "--")

    for root in dict.fromkeys(roots):
        for p in sorted((root / folder / "snapshots").glob("*/config.json")):
            try:
                return json.loads(p.read_text())
            except (OSError, ValueError):
                pass
    return None

def expected_vision_paths(model, key):
    if key == "resnet50":
        paths = ["conv1", "bn1", "relu", "maxpool"]
        for stage in range(1, 5):
            paths.extend(
                f"layer{stage}.{i}"
                for i in range(len(getattr(model, f"layer{stage}")))
            )
        return paths + ["avgpool", "fc"]

    f = model.features
    paths = []
    for i in range(8):
        if i in (1, 3, 5, 7):
            paths.extend(
                f"features.{i}.{j}"
                for j in range(len(f[i]))
            )
        else:
            paths.append(f"features.{i}")
    return paths + ["avgpool", "classifier"]

def expanded_vision_path(model, key, path):
    if key == "resnet50" and path in (
        "layer1", "layer2", "layer3", "layer4"
    ):
        return [
            f"{path}.{i}"
            for i in range(len(model.get_submodule(path)))
        ]

    if key != "resnet50" and path in (
        "features.1", "features.3",
        "features.5", "features.7"
    ):
        return [
            f"{path}.{i}"
            for i in range(len(model.get_submodule(path)))
        ]

    return [path]

def validate_vision(model, key, manifest, x, reference):
    import torch

    groups = manifest["groups"]
    assert len(groups) == manifest["group_count"]

    module_paths = [
        path
        for group in groups
        for path in group["modules"]
    ]

    # Validate every named module exists.
    for path in module_paths:
        model.get_submodule(path)

    # Validate complete, ordered coverage of the architecture.
    expanded = [
        item
        for path in module_paths
        for item in expanded_vision_path(model, key, path)
    ]
    expected = expected_vision_paths(model, key)

    if expanded != expected:
        mismatch = next(
            (
                (i, a, b)
                for i, (a, b) in enumerate(zip(expanded, expected))
                if a != b
            ),
            None
        )
        raise RuntimeError(
            f"Architecture coverage mismatch: "
            f"got={len(expanded)} expected={len(expected)} "
            f"first_mismatch={mismatch}"
        )

    # Validate grouped execution gives the same output.
    z = x
    with torch.inference_mode():
        for group in groups:
            for path in group["modules"]:
                module = model.get_submodule(path)

                # ResNet50's forward inserts flatten before fc.
                if key == "resnet50" and path == "fc":
                    z = torch.flatten(z, 1)

                z = module(z)

    if z.shape != reference.shape:
        raise RuntimeError(
            f"Output shape mismatch: {z.shape} vs {reference.shape}"
        )

    diff = (z - reference).abs().max().item()
    passed = torch.allclose(
        z, reference,
        atol=1e-5,
        rtol=1e-5
    )

    if not passed:
        raise RuntimeError(f"Numerical mismatch: {diff:.3e}")

    return diff

def run_vision():
    import torch
    from torchvision import models

    torch.set_num_threads(4)
    torch.manual_seed(42)

    definitions = {
        "resnet50": ["stage6", "block18"],
        "convnext_base": ["compatible35", "fine41"],
        "convnext_large": ["compatible35", "fine41"],
    }

    for key, profiles in definitions.items():
        model = getattr(models, key)(weights=None).eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.inference_mode():
            reference = model(x)

        for profile in profiles:
            try:
                manifest = load(key, profile)
                diff = validate_vision(
                    model, key, manifest, x, reference
                )
                record(
                    key, profile, "NUMERICAL_PASS",
                    f"groups={manifest['group_count']} "
                    f"max_diff={diff:.3e}"
                )
            except Exception as exc:
                record(key, profile, "FAILED", str(exc))

        del model, x, reference
        gc.collect()

def run_text_metadata():
    definitions = {
        "qwen25_15b": (
            "Qwen/Qwen2.5-1.5B-Instruct",
            "qwen2",
        ),
        "qwen3_17b": (
            "Qwen/Qwen3-1.7B",
            "qwen3",
        ),
    }

    for key, (repo, expected_type) in definitions.items():
        cfg = cache_config(repo)

        if cfg is None:
            record(key, "config", "PENDING", "local config missing")
            continue

        layers = int(cfg["num_hidden_layers"])

        for chunk in (1, 2, 4):
            profile = f"blocks_per_group_{chunk}"

            try:
                manifest = load(key, profile)
                groups = manifest["groups"]

                paths = [
                    p for g in groups
                    for p in g["modules"]
                ]

                indices = [
                    int(match.group(1))
                    for p in paths
                    if (match := re.fullmatch(
                        r"model\.layers\.(\d+)", p
                    ))
                ]

                assert cfg["model_type"] == expected_type
                assert len(groups) == manifest["group_count"]
                assert indices == list(range(layers))
                assert paths[0] == "model.embed_tokens"
                assert paths[-2:] == ["model.norm", "lm_head"]

                record(
                    key, profile, "METADATA_PASS",
                    f"decoder_layers={layers} "
                    f"groups={len(groups)} "
                    "runtime_NOT_checked"
                )
            except Exception as exc:
                record(key, profile, "FAILED", str(exc))

record(
    "distilgpt2", "all", "PENDING",
    "local config unavailable; runtime not checked"
)

run_text_metadata()
run_vision()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps({
    "validator": "mac27-v1",
    "numerical_test_input": "random_fp32_1x3x224x224_cpu",
    "results": results,
}, indent=2) + "\n")

print("\n=== MAC-27 SUMMARY ===")
for status in sorted(set(x["status"] for x in results)):
    print(status, sum(x["status"] == status for x in results))

print("Saved:", OUTPUT)

if any(x["status"] == "FAILED" for x in results):
    sys.exit(1)
