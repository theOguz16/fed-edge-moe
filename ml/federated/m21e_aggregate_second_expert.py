import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2MoeForCausalLM

from ml.training.synthetic_v2 import build_split_tensor


BASE_MODEL = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

E0_V006 = Path(
    "checkpoints/m20/global_v006/L0_E0.safetensors"
)

E3_V007 = Path(
    "checkpoints/m21/global_v007/L0_E3.safetensors"
)

ROUND_DIR = Path(
    "checkpoints/m21/round_v007_multi_expert"
)

OUT_DIR = Path(
    "checkpoints/m21/global_v008"
)

RESULT = Path(
    "results/m21e_L1_E2_federated_aggregation.json"
)

TARGET_LAYER = 1
TARGET_EXPERT = 2
LAMBDA = 2.0


def apply_shard(model, layer_idx, expert_idx, shard):
    experts = model.model.layers[layer_idx].mlp.experts

    with torch.no_grad():
        experts.gate_up_proj[expert_idx].copy_(
            torch.cat(
                [
                    shard["gate_proj"],
                    shard["up_proj"],
                ],
                dim=0,
            )
        )

        experts.down_proj[expert_idx].copy_(
            shard["down_proj"]
        )


def extract_shard(model, layer_idx, expert_idx):
    experts = model.model.layers[layer_idx].mlp.experts

    gate_up = (
        experts.gate_up_proj[expert_idx]
        .detach()
        .cpu()
        .clone()
    )

    split = gate_up.shape[0] // 2

    return {
        "gate_proj":
            gate_up[:split].clone(),

        "up_proj":
            gate_up[split:].clone(),

        "down_proj":
            experts.down_proj[expert_idx]
            .detach()
            .cpu()
            .clone(),
    }


def effective_delta(state, prefix):
    A = state[f"{prefix}.lora_A"]
    B = state[f"{prefix}.lora_B"]

    return B @ A


def adapter_deltas(state):
    return {
        name:
            effective_delta(state, name)
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

    return float(out.loss.cpu())


def main():
    print()
    print("FedEdgeMoE - M21E")
    print("Second Expert Federated Aggregation")
    print()

    model = (
        Qwen2MoeForCausalLM
        .from_pretrained(BASE_MODEL)
    )

    # Reconstruct accepted global V007.
    apply_shard(
        model,
        0,
        0,
        load_file(str(E0_V006)),
    )

    apply_shard(
        model,
        0,
        3,
        load_file(str(E3_V007)),
    )

    base_target = extract_shard(
        model,
        TARGET_LAYER,
        TARGET_EXPERT,
    )

    mac = load_file(
        str(
            ROUND_DIR
            / "mac_A_L1_E2_lora.safetensors"
        )
    )

    msi = load_file(
        str(
            ROUND_DIR
            / "msi_B_L1_E2_lora.safetensors"
        )
    )

    da = adapter_deltas(mac)
    db = adapter_deltas(msi)

    aggregate = {
        key:
            (da[key] + db[key]) / 2.0
        for key in da
    }

    before = [
        val_loss(model, d)
        for d in range(4)
    ]

    proposed = {
        key:
            base_target[key]
            + aggregate[key]
        for key in base_target
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
                / "L1_E2.safetensors"
            ),
        )

        registry = {
            "version": 8,
            "inherits": 7,
            "experts": [
                {
                    "layer": 0,
                    "expert": 0,
                    "source":
                        "m20/global_v006/L0_E0.safetensors",
                },
                {
                    "layer": 0,
                    "expert": 3,
                    "source":
                        "m21/global_v007/L0_E3.safetensors",
                },
                {
                    "layer": 1,
                    "expert": 2,
                    "source":
                        "m21/global_v008/L1_E2.safetensors",
                },
            ],
        }

        (
            OUT_DIR
            / "registry.json"
        ).write_text(
            json.dumps(
                registry,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("Target expert: L1-E2")
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
            "V007 -> V008"
            if decision == "ACCEPT"
            else "V007 unchanged"
        ),
    )

    RESULT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT.write_text(
        json.dumps(
            {
                "base_version": 7,
                "proposed_version": 8,
                "target_layer": 1,
                "target_expert": 2,
                "before": before,
                "after": after,
                "improvement":
                    improvement,
                "target_gain":
                    target_gain,
                "retention_damage":
                    retention_damage,
                "utility":
                    utility,
                "decision":
                    decision,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
