import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2MoeForCausalLM


BASE_MODEL = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

E0_V006 = Path(
    "checkpoints/m20/global_v006/L0_E0.safetensors"
)

OUT_DIR = Path(
    "checkpoints/m21/base_v006"
)

RESULT = Path(
    "results/m21b_multi_expert_base.json"
)


TARGETS = [
    {
        "client": "mac_A",
        "domain": 0,
        "layer": 0,
        "expert": 3,
        "affinity_pct": 32.97,
    },
    {
        "client": "msi_B",
        "domain": 1,
        "layer": 1,
        "expert": 2,
        "affinity_pct": 27.45,
    },
]


def extract_expert(model, layer_idx, expert_idx):
    experts = (
        model.model.layers[layer_idx]
        .mlp.experts
    )

    gate_up = (
        experts.gate_up_proj[
            expert_idx
        ]
        .detach()
        .cpu()
        .clone()
    )

    split = gate_up.shape[0] // 2

    gate = gate_up[:split].clone()
    up = gate_up[split:].clone()

    down = (
        experts.down_proj[
            expert_idx
        ]
        .detach()
        .cpu()
        .clone()
    )

    return {
        "gate_proj": gate,
        "up_proj": up,
        "down_proj": down,
    }


def apply_expert(
    model,
    layer_idx,
    expert_idx,
    shard,
):
    experts = (
        model.model.layers[layer_idx]
        .mlp.experts
    )

    with torch.no_grad():
        experts.gate_up_proj[
            expert_idx
        ].copy_(
            torch.cat(
                [
                    shard["gate_proj"],
                    shard["up_proj"],
                ],
                dim=0,
            )
        )

        experts.down_proj[
            expert_idx
        ].copy_(
            shard["down_proj"]
        )


def max_diff(a, b):
    return max(
        float(
            (
                a[k] - b[k]
            )
            .abs()
            .max()
        )
        for k in a
    )


def main():
    print()
    print("FedEdgeMoE - M21B")
    print("Prepare Multi-Expert Base")
    print()

    model = (
        Qwen2MoeForCausalLM
        .from_pretrained(BASE_MODEL)
    )

    # Global state = immutable V000 model
    # + accepted L0-E0 V006 overlay.
    e0 = load_file(
        str(E0_V006)
    )

    apply_expert(
        model,
        layer_idx=0,
        expert_idx=0,
        shard=e0,
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "global_version": "V006",
        "targets": [],
    }

    for t in TARGETS:
        layer_idx = t["layer"]
        expert_idx = t["expert"]

        shard = extract_expert(
            model,
            layer_idx,
            expert_idx,
        )

        path = (
            OUT_DIR
            / (
                f"L{layer_idx}_"
                f"E{expert_idx}.safetensors"
            )
        )

        save_file(
            shard,
            str(path),
        )

        reloaded = load_file(
            str(path)
        )

        diff = max_diff(
            shard,
            reloaded,
        )

        params = sum(
            x.numel()
            for x in shard.values()
        )

        bytes_ = path.stat().st_size

        item = {
            **t,
            "parameters": params,
            "bytes": bytes_,
            "reload_max_diff": diff,
            "path": str(path),
        }

        report["targets"].append(
            item
        )

        print(
            f"{t['client']} / D{t['domain']}"
        )
        print(
            f"  target: L{layer_idx}-E{expert_idx}"
        )
        print(
            f"  affinity: {t['affinity_pct']:.2f}%"
        )
        print(
            f"  params: {params:,}"
        )
        print(
            f"  bytes: {bytes_:,}"
        )
        print(
            f"  reload max diff: {diff:.3e}"
        )
        print()

    RESULT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    ok = all(
        x["reload_max_diff"] == 0.0
        for x in report["targets"]
    )

    print(
        "MULTI-EXPERT BASE READY:",
        ok,
    )

    print(
        "Saved:",
        RESULT,
    )


if __name__ == "__main__":
    main()
