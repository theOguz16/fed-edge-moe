import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ml.model.transformer import MiniMoELM
from ml.model.expert import ExpertMLP
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot
from ml.distributed.expert_dispatch import (
    prepare_expert_job,
    merge_expert_result,
)


CHECKPOINT = (
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

EXPERT_SHARD = (
    "checkpoints/m13a/v0100_L0_E7/"
    "experts/layer_00/"
    "expert_07_v0100.safetensors"
)

REPORT = Path(
    "results/m13g_split_backprop_equivalence.json"
)

TARGET_LAYER = 0
TARGET_EXPERT = 7
LR = 5e-4


def build_model():
    return MiniMoELM(
        vocab_size=64,
        max_seq_len=32,
        d_model=128,
        num_heads=4,
        num_layers=3,
        num_experts=8,
        top_k=2,
        expert_hidden_dim=256,
        router_aux_loss_weight=0.01,
    )


def get_batch():
    pool = build_split_tensor(
        domain=1,
        split="train",
        seq_len=16,
    )

    sequences = pool[:32]

    x = sequences[:, :-1]

    y = sequences[:, 1:].clone()

    # V2 recurrence için ilk target ignore.
    y[:, 0] = -100

    return x, y


def clone_state(module):
    return {
        key: value.detach().clone()
        for key, value
        in module.state_dict().items()
    }


def max_state_diff(a, b):
    return max(
        (
            a[key] - b[key]
        ).abs().max().item()
        for key in a
    )


def monolithic_step(
    model,
    x,
    y,
):
    for p in model.parameters():
        p.requires_grad = False

    expert = (
        model.blocks[TARGET_LAYER]
        .moe.experts[TARGET_EXPERT]
    )

    for p in expert.parameters():
        p.requires_grad = True

    before = clone_state(expert)

    optimizer = torch.optim.AdamW(
        expert.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    output = model(
        input_ids=x,
    )

    loss = F.cross_entropy(
        output.logits.reshape(
            -1,
            output.logits.shape[-1],
        ),
        y.reshape(-1),
    )

    loss.backward()

    grads = {
        name: p.grad.detach().clone()
        for name, p
        in expert.named_parameters()
    }

    optimizer.step()

    after = clone_state(expert)

    return (
        float(loss.detach()),
        grads,
        before,
        after,
        output.logits.detach(),
    )


def split_step(
    server,
    edge_expert,
    x,
    y,
):
    for p in server.parameters():
        p.requires_grad = False

    for p in edge_expert.parameters():
        p.requires_grad = True

    before = clone_state(
        edge_expert
    )

    optimizer = torch.optim.AdamW(
        edge_expert.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    batch_size, seq_len = (
        x.shape
    )

    positions = torch.arange(
        seq_len
    )

    h = (
        server.token_embedding(x)
        + server.position_embedding(
            positions
        )[None, :, :]
    )

    target_routes = 0
    boundary_output = None
    edge_output = None

    for layer_id, block in enumerate(
        server.blocks
    ):
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        attn_input = (
            block.attn_norm(h)
        )

        attn_output, _ = (
            block.attention(
                attn_input,
                attn_input,
                attn_input,
                attn_mask=causal_mask,
                need_weights=False,
            )
        )

        h = h + attn_output

        moe_input = (
            block.moe_norm(h)
        )

        # Sadece layer 0 target expert'i
        # fiziksel boundary gibi ayırıyoruz.
        if layer_id != TARGET_LAYER:
            moe_output = block.moe(
                moe_input
            )

            h = (
                h
                + moe_output.hidden_states
            )

            continue

        route = block.moe.router(
            moe_input
        )

        flat_output = torch.zeros_like(
            moe_input.reshape(
                -1,
                moe_input.shape[-1],
            )
        )

        for expert_id, expert in enumerate(
            block.moe.experts
        ):
            job = prepare_expert_job(
                moe_input,
                route,
                expert_id,
            )

            if (
                job.token_indices.numel()
                == 0
            ):
                continue

            if expert_id == TARGET_EXPERT:
                target_routes = int(
                    job.token_indices.numel()
                )

                # SERVER -> EDGE boundary
                edge_input = (
                    job.expert_input
                    .detach()
                )

                edge_output = (
                    edge_expert(
                        edge_input
                    )
                )

                # EDGE -> SERVER boundary
                boundary_output = (
                    edge_output
                    .detach()
                    .requires_grad_(True)
                )

                expert_result = (
                    boundary_output
                )

            else:
                expert_result = expert(
                    job.expert_input
                )

            flat_output = (
                merge_expert_result(
                    flat_output,
                    job,
                    expert_result,
                )
            )

        h = h + flat_output.reshape_as(
            moe_input
        )

    h = server.final_norm(h)

    logits = server.lm_head(h)

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            logits.shape[-1],
        ),
        y.reshape(-1),
    )

    # SERVER backward:
    # gradient expert-output boundary'ye gelir.
    loss.backward()

    gradient_to_edge = (
        boundary_output.grad
        .detach()
        .clone()
    )

    # SERVER -> EDGE gradient transfer.
    edge_output.backward(
        gradient_to_edge
    )

    grads = {
        name: p.grad.detach().clone()
        for name, p
        in edge_expert.named_parameters()
    }

    optimizer.step()

    after = clone_state(
        edge_expert
    )

    return (
        float(loss.detach()),
        grads,
        before,
        after,
        logits.detach(),
        target_routes,
        gradient_to_edge,
    )


def main():
    torch.manual_seed(20260927)
    torch.set_num_threads(1)

    x, y = get_batch()

    monolithic = build_model()
    split_server = build_model()

    load_global_snapshot(
        monolithic,
        CHECKPOINT,
    )

    load_global_snapshot(
        split_server,
        CHECKPOINT,
    )

    monolithic.eval()
    split_server.eval()

    edge_expert = ExpertMLP(
        d_model=128,
        hidden_dim=256,
    )

    edge_expert.load_state_dict(
        load_file(EXPERT_SHARD)
    )

    (
        mono_loss,
        mono_grads,
        mono_before,
        mono_after,
        mono_logits,
    ) = monolithic_step(
        monolithic,
        x,
        y,
    )

    (
        split_loss,
        split_grads,
        split_before,
        split_after,
        split_logits,
        target_routes,
        gradient_to_edge,
    ) = split_step(
        split_server,
        edge_expert,
        x,
        y,
    )

    initial_diff = max_state_diff(
        mono_before,
        split_before,
    )

    grad_diff = max(
        (
            mono_grads[key]
            - split_grads[key]
        ).abs().max().item()
        for key in mono_grads
    )

    update_diff = max_state_diff(
        mono_after,
        split_after,
    )

    logits_diff = (
        mono_logits
        - split_logits
    ).abs().max().item()

    loss_diff = abs(
        mono_loss
        - split_loss
    )

    gradient_norm = (
        gradient_to_edge.norm().item()
    )

    equivalent = (
        initial_diff < 1e-8
        and logits_diff < 1e-6
        and loss_diff < 1e-6
        and grad_diff < 1e-6
        and update_diff < 1e-6
        and target_routes > 0
    )

    print()
    print("FedEdgeMoE - M13G")
    print("Split Backprop Equivalence")
    print()

    print(
        "Target:",
        f"L{TARGET_LAYER}-E{TARGET_EXPERT}",
    )

    print(
        "Routed tokens:",
        target_routes,
    )

    print(
        "Monolithic loss:",
        f"{mono_loss:.8f}",
    )

    print(
        "Split loss:",
        f"{split_loss:.8f}",
    )

    print(
        "Loss difference:",
        loss_diff,
    )

    print(
        "Logits max diff:",
        logits_diff,
    )

    print(
        "Gradient-to-edge norm:",
        gradient_norm,
    )

    print(
        "Expert gradient max diff:",
        grad_diff,
    )

    print(
        "Expert update max diff:",
        update_diff,
    )

    print()
    print(
        "SPLIT BACKPROP EQUIVALENT:",
        equivalent,
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
                "target":
                    "L0-E7",
                "routed_tokens":
                    target_routes,
                "monolithic_loss":
                    mono_loss,
                "split_loss":
                    split_loss,
                "loss_diff":
                    loss_diff,
                "logits_max_diff":
                    logits_diff,
                "gradient_to_edge_norm":
                    gradient_norm,
                "expert_gradient_max_diff":
                    grad_diff,
                "expert_update_max_diff":
                    update_diff,
                "equivalent":
                    equivalent,
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
