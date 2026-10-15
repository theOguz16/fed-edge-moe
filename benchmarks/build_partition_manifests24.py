import csv
import json
from pathlib import Path

OUT = Path("configs/partition_manifests_v2")
OUT.mkdir(parents=True, exist_ok=True)

records = []

def group(stage, name, *modules):
    return {
        "stage": stage,
        "name": name,
        "modules": list(modules),
    }

def emit(model, granularity, groups, notes):
    n = len(groups)
    assert n >= 2

    indexed = [
        {"group_id": f"G{i}", **g}
        for i, g in enumerate(groups, 1)
    ]

    cuts = [
        {
            "cut_after": f"G{i}",
            "before": indexed[i-1]["name"],
            "after": indexed[i]["name"],
            "boundary_type": (
                "stage_boundary"
                if indexed[i-1]["stage"] != indexed[i]["stage"]
                else "block_boundary"
            ),
        }
        for i in range(1, n)
    ]

    # Two choices of starting device, with 0-2 inner switches.
    placements = 2 * (
        1 + (n-1) + ((n-1)*(n-2))//2
    )

    obj = {
        "schema_version": "2.0-draft",
        "model_id": model,
        "granularity": granularity,
        "execution_status": "manifest_only_not_validated",
        "group_count": n,
        "cut_boundary_count": len(cuts),
        "theoretical_placements_max_2_switches": placements,
        "groups": indexed,
        "valid_cut_boundaries": cuts,
        "notes": notes,
    }

    dest = OUT / f"{model}__{granularity}.json"
    dest.write_text(json.dumps(obj, indent=2) + "\n")

    records.append({
        "model": model,
        "granularity": granularity,
        "groups": n,
        "cut_boundaries": n - 1,
        "placements_max_2_switches": placements,
        "status": "manifest_ready",
    })

def resnet50():
    coarse = [group("stem", "stem",
                    "conv1", "bn1", "relu", "maxpool")]

    for stage in range(1, 5):
        coarse.append(group(
            f"layer{stage}", f"layer{stage}",
            f"layer{stage}"
        ))

    coarse.append(group("head", "head", "avgpool", "fc"))

    fine = [coarse[0]]
    for stage, count in [(1,3), (2,4), (3,6), (4,3)]:
        for block in range(count):
            fine.append(group(
                f"layer{stage}",
                f"layer{stage}_block{block+1}",
                f"layer{stage}.{block}"
            ))
    fine.append(coarse[-1])

    notes = [
        "Bottleneck residual branch remains inside its block.",
        "Head requires flatten between avgpool and fc.",
        "Every proposed execution must pass numerical validation.",
    ]
    emit("resnet50", "stage6", coarse, notes)
    emit("resnet50", "block18", fine, notes)

def convnext(model):
    canonical = [
        group("stem_stage1", "stem_and_stage1",
              "features.0", "features.1"),
        group("stage2", "downsample2_and_stage2",
              "features.2", "features.3"),
        group("stage3", "downsample3", "features.4"),
    ]

    for i in range(27):
        canonical.append(group(
            "stage3", f"stage3_block{i+1}",
            f"features.5.{i}"
        ))

    canonical.append(
        group("stage4", "downsample4", "features.6")
    )

    for i in range(3):
        canonical.append(group(
            "stage4", f"stage4_block{i+1}",
            f"features.7.{i}"
        ))

    canonical.append(
        group("head", "head", "avgpool", "classifier")
    )

    assert len(canonical) == 35

    fine = [group("stem", "stem", "features.0")]

    for i in range(3):
        fine.append(group(
            "stage1", f"stage1_block{i+1}",
            f"features.1.{i}"
        ))

    fine.append(
        group("stage2", "downsample2", "features.2")
    )

    for i in range(3):
        fine.append(group(
            "stage2", f"stage2_block{i+1}",
            f"features.3.{i}"
        ))

    fine.extend(canonical[2:])
    assert len(fine) == 41

    notes = [
        "Both variants share stage depths 3/3/27/3.",
        "Group35 preserves previous ConvNeXt-Large results.",
        "Group41 additionally separates Stage1 and Stage2.",
        "Module shapes and execution must be validated per model.",
    ]

    emit(model, "compatible35", canonical, notes)
    emit(model, "fine41", fine, notes)


def cached_transformer_layers(model_ref):
    import os

    roots = []

    if os.environ.get("HF_HUB_CACHE"):
        roots.append(
            Path(os.environ["HF_HUB_CACHE"]).expanduser()
        )

    if os.environ.get("HF_HOME"):
        roots.append(
            Path(os.environ["HF_HOME"]).expanduser() / "hub"
        )

    roots.append(Path.home() / ".cache/huggingface/hub")

    folder_name = "models--" + model_ref.replace("/", "--")

    for root in dict.fromkeys(roots):
        snapshots = root / folder_name / "snapshots"

        for config_path in sorted(
            snapshots.glob("*/config.json")
        ):
            try:
                cfg = json.loads(config_path.read_text())
                count = cfg.get(
                    "num_hidden_layers",
                    cfg.get("n_layer")
                )

                if count is not None and int(count) > 0:
                    return int(count), None

            except (OSError, ValueError, TypeError) as exc:
                print(
                    "Invalid local config:",
                    config_path, exc
                )

    return None, (
        "Local config.json not found; "
        "no download attempted"
    )

def text_model(model, ref, architecture):
    layers, error = cached_transformer_layers(ref)

    if layers is None:
        records.append({
            "model": model,
            "granularity": "pending_local_config",
            "groups": "",
            "cut_boundaries": "",
            "placements_max_2_switches": "",
            "status": f"config_unavailable: {error}",
        })
        return

    if architecture == "gpt2":
        embedding = group(
            "embedding", "token_position_embeddings",
            "transformer.wte", "transformer.wpe",
            "transformer.drop"
        )
        layer_path = "transformer.h"
        final_norm = group(
            "final_norm", "final_norm", "transformer.ln_f"
        )
    else:
        embedding = group(
            "embedding", "token_embeddings",
            "model.embed_tokens"
        )
        layer_path = "model.layers"
        final_norm = group(
            "final_norm", "final_norm", "model.norm"
        )

    head = group("lm_head", "lm_head", "lm_head")

    notes = [
        f"Layer count read from local config: {layers}.",
        "Decoder block boundaries only, not inside attention.",
        "Attention masks, positions, RoPE and KV cache "
        "require architecture-specific forward logic.",
        "Manifest alone does not establish executable partitioning.",
    ]

    for chunk in ([1, 3] if architecture == "gpt2"
                  else [1, 2, 4]):
        groups = [embedding]

        for start in range(0, layers, chunk):
            end = min(start + chunk, layers)
            modules = [
                f"{layer_path}.{i}"
                for i in range(start, end)
            ]
            groups.append(group(
                "decoder",
                f"blocks_{start+1}_to_{end}",
                *modules
            ))

        groups.extend([final_norm, head])

        emit(
            model,
            f"blocks_per_group_{chunk}",
            groups,
            notes,
        )

resnet50()
convnext("convnext_base")
convnext("convnext_large")

text_model(
    "distilgpt2",
    "distilbert/distilgpt2",
    "gpt2",
)
text_model(
    "qwen25_15b",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen",
)
text_model(
    "qwen3_17b",
    "Qwen/Qwen3-1.7B",
    "qwen",
)

with (OUT / "manifest_index.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)

print("=== MAC-24 PARTITION MANIFESTS ===")
for r in records:
    print(
        f"{r['model']:17s} "
        f"{r['granularity']:24s} "
        f"groups={str(r['groups']):>3s} "
        f"cuts={str(r['cut_boundaries']):>3s} "
        f"status={r['status']}"
    )

print("\nSaved:", OUT)
