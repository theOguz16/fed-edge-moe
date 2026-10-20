import collections
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

OUT = Path(sys.argv[1]).resolve()
BASE = Path("results/benchmark_v3").resolve()

DB = BASE / (
    "pipeline_recovery/"
    "t21_1_20261017T201021Z_2438/"
    "dataset_v3_with_pipeline.sqlite"
)

T05 = BASE / (
    "validation/t05_20261015T173344Z_2438/"
    "metadata.json"
)

IMAGE_ROOT = Path(
    "data/imagenetv2/"
    "imagenetv2-matched-frequency-format-val"
).resolve()

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()

def save(path, data):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

assert DB.is_file(), "T21.1 database missing"
assert T05.is_file(), "T05 metadata missing"

db = sqlite3.connect(
    f"file:{DB}?mode=ro",
    uri=True
)

try:
    assert db.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0] == "ok"

    task_states = dict(db.execute(
        "SELECT state,COUNT(*) FROM tasks GROUP BY state"
    ).fetchall())

    counts = {
        "tasks": db.execute(
            "SELECT COUNT(*) FROM tasks"
        ).fetchone()[0],
        "energy": db.execute(
            "SELECT COUNT(*) FROM energy_measurements"
        ).fetchone()[0],
        "pipeline": db.execute(
            "SELECT COUNT(*) FROM pipeline_measurements"
        ).fetchone()[0],
        "research_models": db.execute(
            "SELECT COUNT(*) FROM research_models"
        ).fetchone()[0],
    }

    assert counts == {
        "tasks": 327336,
        "energy": 18,
        "pipeline": 24,
        "research_models": 6,
    }
    assert task_states == {
        "completed": 6,
        "planned": 327330,
    }
finally:
    db.close()

packages = {}
for pkg in (
    "torch", "torchvision", "transformers",
    "tokenizers", "safetensors",
    "huggingface_hub", "accelerate",
    "sentencepiece", "datasets", "psutil"
):
    try:
        packages[pkg] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        packages[pkg] = None

import torch
from torchvision.models import (
    ResNet50_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)

weight_enums = {
    "resnet50": ResNet50_Weights.IMAGENET1K_V2,
    "convnext_base": ConvNeXt_Base_Weights.IMAGENET1K_V1,
    "convnext_large": ConvNeXt_Large_Weights.IMAGENET1K_V1,
}

old_meta = json.loads(T05.read_text())
old_models = old_meta.get("models", {})

vision = []

for model, weights in weight_enums.items():
    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlparse(weights.url).path).name
    )

    exists = checkpoint.is_file()
    actual_sha = sha(checkpoint) if exists else None

    expected_sha = (
        old_models.get(model, {})
        .get("weight_sha256")
    )

    verified = (
        actual_sha == expected_sha
        if actual_sha and expected_sha
        else None
    )

    vision.append({
        "model": model,
        "checkpoint": str(checkpoint),
        "checkpoint_exists": exists,
        "checkpoint_bytes":
            checkpoint.stat().st_size if exists else None,
        "checkpoint_sha256": actual_sha,
        "t05_expected_sha256": expected_sha,
        "matches_t05": verified,
        "declared_weight_variant": str(weights),
        "checkpoint_variant_confirmed":
            verified is True,
    })

cache_roots = []

if os.environ.get("HF_HUB_CACHE"):
    cache_roots.append(
        Path(os.environ["HF_HUB_CACHE"]).expanduser()
    )

if os.environ.get("HF_HOME"):
    cache_roots.append(
        Path(os.environ["HF_HOME"]).expanduser() / "hub"
    )

cache_roots.append(
    Path.home() / ".cache/huggingface/hub"
)

cache_roots = list(dict.fromkeys(
    path.resolve() for path in cache_roots
))

text_targets = {
    "distilgpt2": ["distilgpt2"],
    "qwen2_5_1_5b": ["qwen2.5-1.5b"],
    "qwen3_1_7b": ["qwen3-1.7b"],
}

text_models = []

for model, needles in text_targets.items():
    candidates = []

    for root in cache_roots:
        if not root.is_dir():
            continue

        for repository in root.iterdir():
            if not repository.is_dir():
                continue

            name = repository.name.lower()

            if not name.startswith("models--"):
                continue

            if not any(
                needle in name for needle in needles
            ):
                continue

            snapshots = repository / "snapshots"
            if not snapshots.is_dir():
                continue

            for snapshot in snapshots.iterdir():
                if not snapshot.is_dir():
                    continue

                files = [
                    p for p in snapshot.iterdir()
                    if p.is_file()
                ]

                config = snapshot / "config.json"

                weights = [
                    p for p in files
                    if p.name.endswith((
                        ".safetensors",
                        ".bin",
                        ".safetensors.index.json",
                        ".bin.index.json",
                    ))
                ]

                tokenizer_files = [
                    p for p in files
                    if p.name in (
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "vocab.json",
                        "merges.txt",
                        "tokenizer.model",
                        "spiece.model",
                    )
                ]

                candidates.append({
                    "repository_cache_name":
                        repository.name,
                    "snapshot_revision":
                        snapshot.name,
                    "snapshot_path": str(snapshot),
                    "config_present":
                        config.is_file(),
                    "config_sha256":
                        sha(config)
                        if config.is_file()
                        else None,
                    "weight_filenames":
                        sorted(p.name for p in weights),
                    "tokenizer_filenames":
                        sorted(
                            p.name for p in tokenizer_files
                        ),
                    "candidate_complete":
                        bool(
                            config.is_file()
                            and weights
                            and tokenizer_files
                        ),
                    "full_weight_hash_verified": False,
                })

    text_models.append({
        "model": model,
        "candidate_snapshots": candidates,
        "candidate_count": len(candidates),
        "checkpoint_identity_finalized": False,
        "inference_validated": False,
    })

image_counts = collections.Counter()
image_count = 0
image_manifest = OUT / "imagenetv2_manifest.jsonl"
image_temp = OUT / "imagenetv2_manifest.jsonl.tmp"

if IMAGE_ROOT.is_dir():
    with image_temp.open("w", encoding="utf-8") as f:
        for path in sorted(IMAGE_ROOT.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in (
                ".jpg", ".jpeg", ".png"
            ):
                continue

            relative = path.relative_to(IMAGE_ROOT)
            class_name = path.parent.name

            label = (
                int(class_name)
                if class_name.isdigit()
                and 0 <= int(class_name) <= 999
                else None
            )

            if label is not None:
                image_counts[label] += 1

            record = {
                "path": str(relative),
                "label": label,
                "sha256": sha(path),
                "bytes": path.stat().st_size,
            }

            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )
            image_count += 1

        f.flush()
        os.fsync(f.fileno())

    os.replace(image_temp, image_manifest)

dataset_ready = (
    image_count == 10000
    and len(image_counts) == 1000
    and all(
        image_counts[i] == 10
        for i in range(1000)
    )
)

protocol = {
    "name": "FedEdgeMoE Dataset V3 model-quality protocol",
    "version": "T22_v1",
    "models": {
        "vision": [
            "resnet50",
            "convnext_base",
            "convnext_large",
        ],
        "text": [
            "distilgpt2",
            "qwen2_5_1_5b",
            "qwen3_1_7b",
        ],
    },
    "precision_candidates": [
        "fp32", "fp16", "bf16"
    ],
    "precision_support_requires_model_validation": True,
    "vision": {
        "dataset": "ImageNetV2 matched-frequency",
        "required_images": 10000,
        "metrics": [
            "top1_accuracy",
            "top5_accuracy",
            "latency",
            "throughput",
            "memory",
            "power",
            "energy",
        ],
    },
    "text": {
        "evaluation_workloads":
            "Fixed evaluation corpora and prompt sets pending",
        "metrics": [
            "perplexity",
            "negative_log_likelihood",
            "fp32_to_lower_precision_kl",
            "token_agreement",
            "selfcheck_nli_consistency",
            "prefill_latency",
            "time_to_first_token",
            "decode_tokens_per_second",
            "memory",
            "power",
            "energy",
        ],
        "quality_metrics_not_interchangeable": True,
    },
    "pipeline": {
        "modes": ["sequential", "pipeline"],
        "require_matched_inputs": True,
        "require_stage_timeline": True,
        "gpu_kernel_overlap_not_inferred_from_cpu_timeline":
            True,
    },
    "evidence": {
        "prior_paper":
            "reference_only_until_raw_sources_audited",
        "v3_measurements":
            "separate_provenance_required",
    },
}

save(OUT / "quality_protocol.json", protocol)

report = {
    "protocol": "T22_model_integration_manifest_v1",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "source_database": str(DB),
    "source_database_integrity": "pass",
    "source_counts": counts,
    "source_task_states": task_states,
    "python_packages": packages,
    "mps_available": torch.backends.mps.is_available(),
    "vision_checkpoints": vision,
    "text_checkpoints": text_models,
    "huggingface_cache_roots": [
        str(x) for x in cache_roots
    ],
    "imagenetv2": {
        "root": str(IMAGE_ROOT),
        "images": image_count,
        "labeled_classes": len(image_counts),
        "expected_10000_image_protocol_ready":
            dataset_ready,
        "manifest_sha256":
            sha(image_manifest)
            if image_manifest.exists()
            else None,
    },
    "new_inference_runs": 0,
    "existing_sqlite_modified": False,
    "automatic_downloads": False,
}

save(OUT / "model_inventory.json", report)

artifacts = [
    p for p in OUT.iterdir()
    if p.is_file()
    and p.name != "checksums.json"
]
save(OUT / "checksums.json", {
    p.name: sha(p) for p in artifacts
})

print("=== T22 MODEL & QUALITY MANIFEST ===")
print("SOURCE SQLITE: PASS")
print("TASKS:", counts["tasks"])
print("ENERGY RECORDS:", counts["energy"])
print("PIPELINE RECORDS:", counts["pipeline"])
print("IMAGENETV2 IMAGES:", image_count)
print("IMAGENETV2 READY:", dataset_ready)
print()
print("VISION CHECKPOINTS:")
for item in vision:
    print(
        f"  {item['model']}: "
        f"found={item['checkpoint_exists']} "
        f"t05_sha_match={item['matches_t05']}"
    )

print("\nTEXT CHECKPOINT CANDIDATES:")
for item in text_models:
    complete = sum(
        int(c["candidate_complete"])
        for c in item["candidate_snapshots"]
    )
    print(
        f"  {item['model']}: "
        f"snapshots={item['candidate_count']} "
        f"possibly_complete={complete}"
    )
    for candidate in item["candidate_snapshots"][:4]:
        print(
            "   ",
            candidate["repository_cache_name"],
            candidate["snapshot_revision"],
        )

print("\nTRANSFORMERS:", packages["transformers"])
print("SAFETENSORS:", packages["safetensors"])
print("MPS:", report["mps_available"])
print("AUTOMATIC DOWNLOADS: NO")
print("NEW INFERENCE RUNS: 0")
print("EXISTING SQLITE MODIFIED: NO")
print("T22 MODEL MANIFEST: PASS")
print("RESULT DIRECTORY:", OUT)
