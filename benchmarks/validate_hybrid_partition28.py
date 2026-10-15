import argparse
import hashlib
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--case", default="resnet50:stage6")
p.add_argument("--mapping", default="")
p.add_argument(
    "--out",
    default="results/cut_research/mac_hybrid_validation28"
)
a = p.parse_args()

import torch
from torchvision import models

torch.manual_seed(42)
torch.set_num_threads(4)

model_name, profile = a.case.split(":", 1)
manifest_path = (
    Path("configs/partition_manifests_v2")
    / f"{model_name}__{profile}.json"
)

manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
groups = manifest["groups"]
n = len(groups)

assert manifest["group_count"] == n
assert model_name in (
    "resnet50", "convnext_base", "convnext_large"
)

model = getattr(models, model_name)(
    weights=None
).eval()

x = torch.randn(1, 3, 224, 224)

with torch.inference_mode():
    reference = model(x).clone()

def default_mappings(n):
    half = n // 2
    first = n // 3
    second = (2 * n) // 3

    return {
        "C": "C" * n,
        "M": "M" * n,
        "CM": "C" * half + "M" * (n - half),
        "MC": "M" * half + "C" * (n - half),
        "MCM": (
            "M" * first
            + "C" * (second - first)
            + "M" * (n - second)
        ),
        "CMC": (
            "C" * first
            + "M" * (second - first)
            + "C" * (n - second)
        )
    }

mappings = (
    {"custom": a.mapping}
    if a.mapping
    else default_mappings(n)
)

for label, mapping in mappings.items():
    if len(mapping) != n or set(mapping) - {"C", "M"}:
        raise ValueError(
            f"Invalid mapping {label}: "
            f"expected {n} characters from C/M"
        )

if any("M" in mapping for mapping in mappings.values()):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable on this machine")

def run_mapping(mapping):
    # Assign every architectural group to its requested device.
    for group, code in zip(groups, mapping):
        target = "mps" if code == "M" else "cpu"

        for path in group["modules"]:
            model.get_submodule(path).to(target).eval()

    transfers = 0
    z = x

    with torch.inference_mode():
        for group, code in zip(groups, mapping):
            target = "mps" if code == "M" else "cpu"

            if z.device.type != target:
                z = z.to(target)
                transfers += 1

            for path in group["modules"]:
                module = model.get_submodule(path)

                # torchvision ResNet50 flattens before fc.
                if model_name == "resnet50" and path == "fc":
                    z = torch.flatten(z, 1)

                z = module(z)

        if z.device.type != "cpu":
            z = z.to("cpu")
            transfers += 1

        if "M" in mapping:
            torch.mps.synchronize()

    return z, transfers

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rows = []

print(f"=== MAC-28 HYBRID VALIDATION ===")
print(f"Model: {model_name}")
print(f"Profile: {profile}")
print(f"Groups: {n}")

for label, mapping in mappings.items():
    try:
        actual, transfers = run_mapping(mapping)

        shape_ok = actual.shape == reference.shape

        max_diff = (
            (actual - reference).abs().max().item()
            if shape_ok else None
        )

        numerical_pass = (
            shape_ok and
            torch.allclose(
                actual,
                reference,
                rtol=1e-3,
                atol=1e-3
            )
        )

        status = "PASS" if numerical_pass else "FAIL"

        record = {
            "model": model_name,
            "profile": profile,
            "mapping_name": label,
            "mapping": mapping,
            "status": status,
            "max_abs_diff": max_diff,
            "transfer_count": transfers,
            "error": None,
        }

    except Exception as exc:
        record = {
            "model": model_name,
            "profile": profile,
            "mapping_name": label,
            "mapping": mapping,
            "status": "FAIL",
            "max_abs_diff": None,
            "transfer_count": None,
            "error": repr(exc),
        }

    rows.append(record)

    print(
        f"{label:7s} {record['status']:5s} "
        f"transfers={str(record['transfer_count']):>2s} "
        f"max_diff={record['max_abs_diff']}"
    )

    if record["error"]:
        print("  ERROR:", record["error"])

result = {
    "validator": "mac28-v1",
    "model": model_name,
    "profile": profile,
    "manifest_sha256": hashlib.sha256(
        manifest_bytes
    ).hexdigest(),
    "weights": "random_seed42",
    "input_shape": list(x.shape),
    "reference_device": "cpu",
    "rtol": 1e-3,
    "atol": 1e-3,
    "torch_version": torch.__version__,
    "results": rows,
}

dest = out / f"{model_name}__{profile}.json"
dest.write_text(json.dumps(result, indent=2) + "\n")

passed = sum(r["status"] == "PASS" for r in rows)

print(f"\nPassed: {passed}/{len(rows)}")
print("Saved:", dest)

if passed != len(rows):
    sys.exit(1)
