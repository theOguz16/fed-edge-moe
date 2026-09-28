import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2MoeForCausalLM

from ml.training.synthetic_v2 import build_split_tensor


BASE_MODEL = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

BASE_SHARD = Path(
    "checkpoints/m19f/global_v001/L0_E0.safetensors"
)

ROUND_DIR = Path(
    "checkpoints/m20/round_v001_to_v002"
)

CLIENT_A = ROUND_DIR / "mac_A_E0_lora.safetensors"
CLIENT_B = ROUND_DIR / "msi_B_E0_lora.safetensors"

OUTPUT = Path(
    "checkpoints/m20/global_v002/L0_E0.safetensors"
)

REPORT = Path(
    "results/m20b_round_v001_to_v002.json"
)

LAMBDA = 2.0


def effective_delta(state, prefix):
    A = state[f"{prefix}.lora_A"]
    B = state[f"{prefix}.lora_B"]

    # alpha/rank = 4/4 = 1
    return B @ A


def extract_deltas(state):
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


def apply_shard(model, shard):
    experts = (
        model.model.layers[0]
        .mlp.experts
    )

    with torch.no_grad():
        experts.gate_up_proj[0].copy_(
            torch.cat(
                [
                    shard["gate_proj"],
                    shard["up_proj"],
                ],
                dim=0,
            )
        )

        experts.down_proj[0].copy_(
            shard["down_proj"]
        )


@torch.no_grad()
def validation_loss(model, domain):
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
    state_a = load_file(
        str(CLIENT_A)
    )

    state_b = load_file(
        str(CLIENT_B)
    )

    delta_a = extract_deltas(
        state_a
    )

    delta_b = extract_deltas(
        state_b
    )

    aggregate = {
        key: (
            delta_a[key]
            + delta_b[key]
        ) / 2.0
        for key in delta_a
    }

    model = (
        Qwen2MoeForCausalLM
        .from_pretrained(BASE_MODEL)
    )

    v001 = load_file(
        str(BASE_SHARD)
    )

    apply_shard(
        model,
        v001,
    )

    before = [
        validation_loss(
            model,
            domain,
        )
        for domain in range(4)
    ]

    proposed = {
        key:
            v001[key]
            + aggregate[key]
        for key in aggregate
    }

    apply_shard(
        model,
        proposed,
    )

    after = [
        validation_loss(
            model,
            domain,
        )
        for domain in range(4)
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
        max(
            0.0,
            -improvement[2],
        )
        +
        max(
            0.0,
            -improvement[3],
        )
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
        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_file(
            {
                key:
                    value.cpu()
                for key, value
                in proposed.items()
            },
            str(OUTPUT),
        )

    print()
    print("FedEdgeMoE - M20B")
    print("Physical Multi-Round Aggregation")
    print()

    for domain in range(4):
        print(
            f"D{domain}: "
            f"{before[domain]:.6f} "
            f"-> {after[domain]:.6f} "
            f"(improve "
            f"{improvement[domain]:+.6f})"
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
        "Version:",
        (
            "V001 -> V002"
            if decision == "ACCEPT"
            else "V001 unchanged"
        ),
    )

    print()

    print(
        "ROUND COMMITTED:",
        decision == "ACCEPT",
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "before":
                    before,
                "after":
                    after,
                "improvement":
                    improvement,
                "target_gain":
                    target_gain,
                "retention_damage":
                    retention_damage,
                "lambda":
                    LAMBDA,
                "utility":
                    utility,
                "decision":
                    decision,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
