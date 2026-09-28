import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ml.federated.m18c_qwen2moe_federated_round import (
    build_model,
    train_client,
    average_deltas,
    SEED,
)


GATE_REPORT = Path(
    "results/m18d_qwen2_retention_gate.json"
)

OUTPUT = Path(
    "checkpoints/m18e/"
    "qwen2_L0_E0_accepted.safetensors"
)


def main():
    with open(
        GATE_REPORT,
        "r",
        encoding="utf-8",
    ) as f:
        gate = json.load(f)

    decision = gate["decision"]

    print()
    print("FedEdgeMoE - M18E")
    print("Transactional Global Commit")
    print()

    print(
        "Gate decision:",
        decision,
    )

    if decision != "ACCEPT":
        print(
            "Checkpoint written: False"
        )
        print(
            "TRANSACTION ABORTED"
        )
        return

    torch.manual_seed(SEED)

    global_model = build_model()

    global_state = {
        key:
            value.detach()
            .cpu()
            .clone()
        for key, value
        in global_model.state_dict().items()
    }

    client_a = train_client(
        global_state,
        domain=0,
    )

    client_b = train_client(
        global_state,
        domain=1,
    )

    aggregate = average_deltas(
        client_a,
        client_b,
    )

    experts = (
        global_model
        .model.layers[0]
        .mlp.experts
    )

    gate_up = (
        experts.gate_up_proj[0]
        .detach()
        .clone()
    )

    gate_weight, up_weight = (
        gate_up.chunk(
            2,
            dim=0,
        )
    )

    down_weight = (
        experts.down_proj[0]
        .detach()
        .clone()
    )

    new_gate = (
        gate_weight
        + aggregate["gate_proj"]
    )

    new_up = (
        up_weight
        + aggregate["up_proj"]
    )

    new_down = (
        down_weight
        + aggregate["down_proj"]
    )

    shard = {
        "gate_proj":
            new_gate.cpu(),
        "up_proj":
            new_up.cpu(),
        "down_proj":
            new_down.cpu(),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        shard,
        str(OUTPUT),
    )

    size = OUTPUT.stat().st_size

    print(
        "Checkpoint written:",
        True,
    )

    print(
        "Expert:",
        "Layer 0 / Expert 0",
    )

    print(
        "Parameters:",
        f"{sum(x.numel() for x in shard.values()):,}",
    )

    print(
        "Checkpoint bytes:",
        f"{size:,}",
    )

    print(
        "Committed utility:",
        f"{gate['utility']:+.6f}",
    )

    print()
    print(
        "TRANSACTION COMMIT OK:",
        OUTPUT.exists(),
    )

    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
