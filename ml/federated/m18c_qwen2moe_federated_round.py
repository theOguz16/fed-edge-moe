import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.federated.m18b_qwen2moe_full_forward_lora import (
    HybridExperts,
    build_model,
)
from ml.adapters.expert_adapter import (
    MiniMoEExpertAdapter,
)
from ml.adapters.generic_lora import (
    inject_lora,
)
from ml.adapters.qwen2moe_tensor_expert import (
    Qwen2MoeTensorExpertAdapter,
)
from ml.training.synthetic_v2 import (
    build_split_tensor,
)


SEED = 20260927
LORA_SEED = 424242

TARGET_EXPERT = 0

RANK = 4
ALPHA = 4.0
LR = 1e-3

STEPS = 80
BATCH_SIZE = 32

REPORT = Path(
    "results/m18c_qwen2moe_federated_round.json"
)


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

    output = model(
        input_ids=seq,
        labels=labels,
    )

    return float(
        output.loss.cpu()
    )


def effective_delta(layer):
    return (
        layer.scaling
        * (
            layer.lora_B
            @ layer.lora_A
        )
    ).detach().cpu().clone()


def train_client(
    global_state,
    domain,
):
    torch.manual_seed(SEED)

    model = build_model()

    model.load_state_dict(
        global_state
    )

    for p in model.parameters():
        p.requires_grad = False

    sparse_block = (
        model.model.layers[0].mlp
    )

    original_experts = (
        sparse_block.experts
    )

    bridge = Qwen2MoeTensorExpertAdapter(
        original_experts,
        expert_index=TARGET_EXPERT,
    )

    standalone = bridge.materialize()

    adapter = MiniMoEExpertAdapter(
        standalone
    )

    # Same initial LoRA on every client.
    torch.manual_seed(LORA_SEED)

    inject_lora(
        adapter,
        rank=RANK,
        alpha=ALPHA,
    )

    hybrid = HybridExperts(
        original_experts,
        standalone,
        target_expert=TARGET_EXPERT,
    )

    sparse_block.experts = hybrid

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    trainable_params = sum(
        p.numel()
        for p in trainable
    )

    before = validation_loss(
        model,
        domain,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=0.0,
    )

    pool = build_split_tensor(
        domain,
        "train",
        16,
    )

    # Domain-specific batch randomness.
    torch.manual_seed(
        SEED + 100 + domain
    )

    first_loss = None
    last_loss = None

    model.train()

    for _ in range(STEPS):
        idx = torch.randint(
            0,
            pool.shape[0],
            (BATCH_SIZE,),
        )

        seq = pool[idx]

        labels = seq.clone()
        labels[:, :2] = -100

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            input_ids=seq,
            labels=labels,
        )

        loss = output.loss

        loss.backward()
        optimizer.step()

        value = float(
            loss.detach()
        )

        if first_loss is None:
            first_loss = value

        last_loss = value

    after = validation_loss(
        model,
        domain,
    )

    deltas = {
        "gate_proj":
            effective_delta(
                standalone.gate_proj
            ),

        "up_proj":
            effective_delta(
                standalone.up_proj
            ),

        "down_proj":
            effective_delta(
                standalone.down_proj
            ),
    }

    delta_norm = torch.sqrt(
        sum(
            value.float()
            .pow(2)
            .sum()
            for value
            in deltas.values()
        )
    ).item()

    return {
        "domain": domain,
        "before": before,
        "after": after,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "routed_tokens":
            hybrid.routed_tokens,
        "trainable_params":
            trainable_params,
        "delta_norm":
            delta_norm,
        "deltas":
            deltas,
    }


def average_deltas(
    client_a,
    client_b,
):
    return {
        key: (
            client_a["deltas"][key]
            + client_b["deltas"][key]
        ) / 2.0
        for key in client_a["deltas"]
    }


def main():
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

    baseline = [
        validation_loss(
            global_model,
            domain,
        )
        for domain in range(4)
    ]

    experts = (
        global_model
        .model.layers[0]
        .mlp.experts
    )

    other_before = (
        experts.gate_up_proj[1]
        .detach()
        .clone()
    )

    router = (
        global_model
        .model.layers[0]
        .mlp.gate
    )

    router_before = (
        router.weight
        .detach()
        .clone()
    )

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

    gate, up = (
        experts.gate_up_proj[0]
        .detach()
        .chunk(
            2,
            dim=0,
        )
    )

    down = (
        experts.down_proj[0]
        .detach()
    )

    with torch.no_grad():
        experts.gate_up_proj[0].copy_(
            torch.cat(
                [
                    gate
                    + aggregate[
                        "gate_proj"
                    ],
                    up
                    + aggregate[
                        "up_proj"
                    ],
                ],
                dim=0,
            )
        )

        experts.down_proj[0].copy_(
            down
            + aggregate[
                "down_proj"
            ]
        )

    final = [
        validation_loss(
            global_model,
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

    raw_upload_per_client = (
        client_a["trainable_params"]
        * 4
    )

    mechanics_ok = (
        client_a["delta_norm"] > 0
        and client_b["delta_norm"] > 0
        and other_change == 0.0
        and router_change == 0.0
    )

    global_improved = (
        mean_after < mean_before
    )

    print()
    print("FedEdgeMoE - M18C")
    print("Tiny Qwen2 Federated LoRA Round")
    print()

    print(
        "Client A / D0:",
        f"{client_a['before']:.6f}",
        "->",
        f"{client_a['after']:.6f}",
    )

    print(
        "Client B / D1:",
        f"{client_b['before']:.6f}",
        "->",
        f"{client_b['after']:.6f}",
    )

    print()

    print(
        "Client A delta norm:",
        f"{client_a['delta_norm']:.6f}",
    )

    print(
        "Client B delta norm:",
        f"{client_b['delta_norm']:.6f}",
    )

    print(
        "LoRA upload/client:",
        f"{raw_upload_per_client:,}",
        "bytes fp32",
    )

    print()

    print("Global validation:")

    for domain in range(4):
        improvement = (
            baseline[domain]
            - final[domain]
        )

        print(
            f"  D{domain}: "
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

    print()

    print(
        "FEDERATED ROUND MECHANICS OK:",
        mechanics_ok,
    )

    print(
        "GLOBAL MEAN IMPROVED:",
        global_improved,
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
                "client_a": {
                    key: value
                    for key, value
                    in client_a.items()
                    if key != "deltas"
                },
                "client_b": {
                    key: value
                    for key, value
                    in client_b.items()
                    if key != "deltas"
                },
                "baseline":
                    baseline,
                "final":
                    final,
                "mean_before":
                    mean_before,
                "mean_after":
                    mean_after,
                "upload_per_client_bytes":
                    raw_upload_per_client,
                "other_expert_change":
                    other_change,
                "router_change":
                    router_change,
                "mechanics_ok":
                    mechanics_ok,
                "global_improved":
                    global_improved,
            },
            f,
            indent=2,
        )

    print(
        "Report:",
        REPORT,
    )


if __name__ == "__main__":
    main()
