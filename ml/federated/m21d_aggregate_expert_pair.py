import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2MoeForCausalLM

from ml.training.synthetic_v2 import build_split_tensor


BASE_MODEL = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

GLOBAL_E0_V006 = Path(
    "checkpoints/m20/global_v006/L0_E0.safetensors"
)

ROUND_DIR = Path(
    "checkpoints/m21/round_v006_multi_expert"
)

OUT_DIR = Path(
    "checkpoints/m21/global_v007"
)

RESULT = Path(
    "results/m21d_L0_E3_federated_aggregation.json"
)

TARGET_LAYER = 0
TARGET_EXPERT = 3
LAMBDA = 2.0


def apply_shard(model, layer_idx, expert_idx, shard):
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


def extract_shard(model, layer_idx, expert_idx):
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

    return {
        "gate_proj": gate_up[:split].clone(),
        "up_proj": gate_up[split:].clone(),
        "down_proj":
            experts.down_proj[
                expert_idx
            ]
            .detach()
            .cpu()
            .clone(),
    }


def effective_delta(state, prefix):
    A = state[f"{prefix}.lora_A"]
    B = state[f"{prefix}.lora_B"]

    # alpha / rank = 4 / 4 = 1
    return B @ A


def adapter_deltas(state):
    return {
        name: effective_delta(
            state,
            name,
        )
        for name in [
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    }


@torch.no_grad()
def val_loss(model, domain):
    model.eval()

    seq = build_split_tensor(
        domain,
        "validation",
        16,
    )[:64]

    labels = seq.clone()
    labels[:, :2] = -100

    out = model(
        input_ids=seq,
        labels=labels,
    )

    return float(
        out.loss.cpu()
    )


def main():
    print()
    print("FedEdgeMoE - M21D")
    print("Physical Expert-Wise Aggregation")
    print()

    model = (
        Qwen2MoeForCausalLM
        .from_pretrained(BASE_MODEL)
    )

    # Reconstruct global V006.
    e0 = load_file(
        str(GLOBAL_E0_V006)
    )

    apply_shard(
        model,
        0,
        0,
        e0,
    )

    base_target = extract_shard(
        model,
        TARGET_LAYER,
        TARGET_EXPERT,
    )

    mac = load_file(
        str(
            ROUND_DIR
            / "mac_A_L0_E3_lora.safetensors"
        )
    )

    msi = load_file(
        str(
            ROUND_DIR
            / "msi_B_L0_E3_lora.safetensors"
        )
    )

    da = adapter_deltas(mac)
    db = adapter_deltas(msi)

    aggregate = {
        k: (da[k] + db[k]) / 2.0
        for k in da
    }

    before = [
        val_loss(model, d)
        for d in range(4)
    ]

    proposed = {
        k:
            base_target[k]
            + aggregate[k]
        for k in base_target
    }

    apply_shard(
        model,
        TARGET_LAYER,
        TARGET_EXPERT,
        proposed,
    )

    after = [
        val_loss(model, d)
        for d in range(4)
    ]

    improvement = [
        before[i] - after[i]
        for i in range(4)
    ]

    target_gain = (
        improvement[0]
        + improvement[1]
    ) / 2.0

    retention_damage = (
        max(0.0, -improvement[2])
        + max(0.0, -improvement[3])
    ) / 2.0

    utility = (
        target_gain
        - LAMBDA * retention_damage
    )

    decision = (
        "ACCEPT"
        if utility > 0
        else "REJECT"
    )

    if decision == "ACCEPT":
        OUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_file(
            proposed,
            str(
                OUT_DIR
                / "L0_E3.safetensors"
            ),
        )

    print("Target expert: L0-E3")
    print()

    for d in range(4):
        print(
            f"D{d}: "
            f"{before[d]:.6f} -> "
            f"{after[d]:.6f} "
            f"(improve {improvement[d]:+.6f})"
        )

    print()

    print(
        "Target gain:",
        f"{target_gain:+.6f}",
    )

    print(
        "Retention damage:",
        f"{retention_damage:.6f}",
    )

    print(
        "Utility:",
        f"{utility:+.6f}",
    )

    print(
        "Decision:",
        decision,
    )

    print(
        "Global version:",
        (
            "V006 -> V007"
            if decision == "ACCEPT"
            else "V006 unchanged"
        ),
    )

    report = {
        "base_version": 6,
        "proposed_version": 7,
        "target_layer": TARGET_LAYER,
        "target_expert": TARGET_EXPERT,
        "clients": [
            "mac_A",
            "msi_B",
        ],
        "before": before,
        "after": after,
        "improvement": improvement,
        "target_gain": target_gain,
        "retention_damage":
            retention_damage,
        "utility": utility,
        "decision": decision,
    }

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

    print()
    print("Saved report:", RESULT)


if __name__ == "__main__":
    main()
