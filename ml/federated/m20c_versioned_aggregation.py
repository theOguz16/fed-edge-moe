import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2MoeForCausalLM

from ml.training.synthetic_v2 import build_split_tensor


BASE_MODEL = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

LAMBDA = 2.0


def shard_path(version):
    if version == 1:
        legacy = Path(
            "checkpoints/m19f/global_v001/"
            "L0_E0.safetensors"
        )

        if legacy.exists():
            return legacy

    return Path(
        f"checkpoints/m20/"
        f"global_v{version:03d}/"
        "L0_E0.safetensors"
    )


def effective_delta(state, prefix):
    A = state[f"{prefix}.lora_A"]
    B = state[f"{prefix}.lora_B"]

    return B @ A


def deltas(state):
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
def val_loss(model, domain):
    seq = build_split_tensor(
        domain,
        "validation",
        16,
    )[:64]

    labels = seq.clone()
    labels[:, :2] = -100

    output = model(
        input_ids=seq,
        labels=labels,
    )

    return float(
        output.loss.cpu()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-version",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    base_version = args.base_version
    new_version = base_version + 1

    round_dir = Path(
        f"checkpoints/m20/"
        f"round_v{base_version:03d}_"
        f"to_v{new_version:03d}"
    )

    client_a = load_file(
        str(
            round_dir
            / "mac_A_E0_lora.safetensors"
        )
    )

    client_b = load_file(
        str(
            round_dir
            / "msi_B_E0_lora.safetensors"
        )
    )

    da = deltas(client_a)
    db = deltas(client_b)

    aggregate = {
        key: (
            da[key] + db[key]
        ) / 2.0
        for key in da
    }

    base_shard = load_file(
        str(
            shard_path(
                base_version
            )
        )
    )

    model = (
        Qwen2MoeForCausalLM
        .from_pretrained(BASE_MODEL)
    )

    apply_shard(
        model,
        base_shard,
    )

    before = [
        val_loss(model, d)
        for d in range(4)
    ]

    proposed = {
        key:
            base_shard[key]
            + aggregate[key]
        for key in aggregate
    }

    apply_shard(
        model,
        proposed,
    )

    after = [
        val_loss(model, d)
        for d in range(4)
    ]

    gain = [
        before[i] - after[i]
        for i in range(4)
    ]

    target_gain = (
        gain[0] + gain[1]
    ) / 2.0

    retention_damage = (
        max(0.0, -gain[2])
        + max(0.0, -gain[3])
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
        output = Path(
            f"checkpoints/m20/"
            f"global_v{new_version:03d}/"
            "L0_E0.safetensors"
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_file(
            {
                k: v.cpu()
                for k, v in proposed.items()
            },
            str(output),
        )

    print()
    print("FedEdgeMoE - M20C")
    print("Versioned Round Aggregation")
    print()

    for d in range(4):
        print(
            f"D{d}: "
            f"{before[d]:.6f} -> "
            f"{after[d]:.6f} "
            f"(improve {gain[d]:+.6f})"
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
            f"V{base_version:03d} "
            f"-> V{new_version:03d}"
            if decision == "ACCEPT"
            else
            f"V{base_version:03d} unchanged"
        ),
    )

    report = Path(
        f"results/m20_round_"
        f"v{base_version:03d}_"
        f"to_v{new_version:03d}.json"
    )

    report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        report,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "base_version":
                    base_version,
                "new_version":
                    new_version,
                "before":
                    before,
                "after":
                    after,
                "gain":
                    gain,
                "target_gain":
                    target_gain,
                "retention_damage":
                    retention_damage,
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
