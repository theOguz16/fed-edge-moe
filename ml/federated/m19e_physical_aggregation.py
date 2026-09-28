import json
from pathlib import Path

import torch
from safetensors.torch import (
    load_file,
    save_file,
)
from transformers import Qwen2MoeForCausalLM

from ml.training.synthetic_v2 import (
    build_split_tensor,
)


CHECKPOINT = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

CLIENT_A = Path(
    "checkpoints/m19d/clientA_E0_lora.safetensors"
)

CLIENT_B = Path(
    "checkpoints/m19c/clientB_E0_lora_msi.safetensors"
)

OUTPUT = Path(
    "checkpoints/m19e/global_L0_E0_physical.safetensors"
)

REPORT = Path(
    "results/m19e_physical_aggregation.json"
)


def effective_delta(state, prefix):
    A = state[f"{prefix}.lora_A"]
    B = state[f"{prefix}.lora_B"]

    # alpha/rank = 4/4 = 1
    return B @ A


def client_deltas(state):
    return {
        "gate_proj":
            effective_delta(
                state,
                "gate_proj",
            ),
        "up_proj":
            effective_delta(
                state,
                "up_proj",
            ),
        "down_proj":
            effective_delta(
                state,
                "down_proj",
            ),
    }


@torch.no_grad()
def validation_loss(
    model,
    domain,
):
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
    state_a = load_file(
        str(CLIENT_A)
    )

    state_b = load_file(
        str(CLIENT_B)
    )

    delta_a = client_deltas(
        state_a
    )

    delta_b = client_deltas(
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
        .from_pretrained(
            CHECKPOINT
        )
    )

    baseline = [
        validation_loss(
            model,
            domain,
        )
        for domain in range(4)
    ]

    experts = (
        model.model.layers[0]
        .mlp.experts
    )

    other_before = (
        experts.gate_up_proj[1]
        .detach()
        .clone()
    )

    router = (
        model.model.layers[0]
        .mlp.gate
    )

    router_before = (
        router.weight
        .detach()
        .clone()
    )

    gate, up = (
        experts.gate_up_proj[0]
        .detach()
        .clone()
        .chunk(
            2,
            dim=0,
        )
    )

    down = (
        experts.down_proj[0]
        .detach()
        .clone()
    )

    new_gate = (
        gate
        + aggregate["gate_proj"]
    )

    new_up = (
        up
        + aggregate["up_proj"]
    )

    new_down = (
        down
        + aggregate["down_proj"]
    )

    with torch.no_grad():
        experts.gate_up_proj[0].copy_(
            torch.cat(
                [
                    new_gate,
                    new_up,
                ],
                dim=0,
            )
        )

        experts.down_proj[0].copy_(
            new_down
        )

    final = [
        validation_loss(
            model,
            domain,
        )
        for domain in range(4)
    ]

    other_change = (
        experts.gate_up_proj[1]
        - other_before
    ).abs().max().item()

    router_change = (
        router.weight
        - router_before
    ).abs().max().item()

    mean_before = (
        sum(baseline) / 4
    )

    mean_after = (
        sum(final) / 4
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        {
            "gate_proj":
                new_gate.cpu(),
            "up_proj":
                new_up.cpu(),
            "down_proj":
                new_down.cpu(),
        },
        str(OUTPUT),
    )

    print()
    print("FedEdgeMoE - M19E")
    print("Physical Federated Aggregation")
    print()

    for domain in range(4):
        improvement = (
            baseline[domain]
            - final[domain]
        )

        print(
            f"D{domain}: "
            f"{baseline[domain]:.6f} "
            f"-> {final[domain]:.6f} "
            f"(improve {improvement:+.6f})"
        )

    print()

    print(
        "Mean loss:",
        f"{mean_before:.6f}",
        "->",
        f"{mean_after:.6f}",
        f"(improve {mean_before-mean_after:+.6f})",
    )

    print(
        "Other expert change:",
        other_change,
    )

    print(
        "Router change:",
        router_change,
    )

    physical_ok = (
        mean_after < mean_before
        and other_change == 0.0
        and router_change == 0.0
    )

    print()
    print(
        "PHYSICAL FEDERATION OK:",
        physical_ok,
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
                "baseline": baseline,
                "final": final,
                "mean_before":
                    mean_before,
                "mean_after":
                    mean_after,
                "other_expert_change":
                    other_change,
                "router_change":
                    router_change,
                "physical_ok":
                    physical_ok,
            },
            f,
            indent=2,
        )

    print(
        "Saved expert:",
        OUTPUT,
    )

    print(
        "Report:",
        REPORT,
    )


if __name__ == "__main__":
    main()
