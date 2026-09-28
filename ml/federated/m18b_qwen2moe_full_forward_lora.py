import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
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
TARGET_EXPERT = 0
RANK = 4
ALPHA = 4.0
LR = 1e-3
STEPS = 60
BATCH_SIZE = 32

REPORT = Path(
    "results/m18b_qwen2moe_full_forward_lora.json"
)


def build_model():
    config = Qwen2MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,

        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
        decoder_sparse_step=1,

        use_cache=False,
    )

    return Qwen2MoeForCausalLM(config)


class HybridExperts(nn.Module):
    """
    Experts 1..N use original Qwen tensor weights.
    Expert 0 uses our materialized LoRA expert.
    """

    def __init__(
        self,
        base_experts,
        trainable_expert,
        target_expert=0,
    ):
        super().__init__()

        self.base_experts = base_experts
        self.trainable_expert = trainable_expert

        self.target_expert = target_expert
        self.num_experts = (
            base_experts.num_experts
        )

        self.act_fn = (
            base_experts.act_fn
        )

        self.routed_tokens = 0

    def forward(
        self,
        hidden_states,
        top_k_index,
        top_k_weights,
    ):
        final_hidden_states = (
            torch.zeros_like(
                hidden_states
            )
        )

        expert_mask = F.one_hot(
            top_k_index,
            num_classes=(
                self.num_experts + 1
            ),
        )

        expert_mask = (
            expert_mask.permute(
                2, 1, 0
            )
        )

        expert_hit = (
            expert_mask
            .sum(dim=(-1, -2))
            .gt(0)
            .nonzero()
        )

        for item in expert_hit:
            expert_idx = int(
                item.item()
            )

            if (
                expert_idx
                == self.num_experts
            ):
                continue

            top_k_pos, token_idx = (
                torch.where(
                    expert_mask[
                        expert_idx
                    ]
                )
            )

            current_state = (
                hidden_states[
                    token_idx
                ]
            )

            if (
                expert_idx
                == self.target_expert
            ):
                self.routed_tokens += (
                    token_idx.numel()
                )

                current = (
                    self.trainable_expert(
                        current_state
                    )
                )

            else:
                gate_up = (
                    self.base_experts
                    .gate_up_proj[
                        expert_idx
                    ]
                )

                gate, up = (
                    F.linear(
                        current_state,
                        gate_up,
                    ).chunk(
                        2,
                        dim=-1,
                    )
                )

                current = (
                    self.act_fn(gate)
                    * up
                )

                current = F.linear(
                    current,
                    self.base_experts
                    .down_proj[
                        expert_idx
                    ],
                )

            current = (
                current
                * top_k_weights[
                    token_idx,
                    top_k_pos,
                    None,
                ]
            )

            final_hidden_states.index_add_(
                0,
                token_idx,
                current.to(
                    final_hidden_states.dtype
                ),
            )

        return final_hidden_states


def adapter_change(expert):
    values = []

    for name, parameter in (
        expert.named_parameters()
    ):
        if "lora_B" in name:
            values.append(
                parameter.detach()
                .abs()
                .max()
                .item()
            )

    return max(values)


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

    # recurrence depends on initial context;
    # do not score the seed positions.
    labels[:, :2] = -100

    output = model(
        input_ids=seq,
        labels=labels,
    )

    return float(
        output.loss.cpu()
    )


def main():
    torch.manual_seed(SEED)

    # CPU deliberately:
    # tiny model + deterministic integration test.
    device = torch.device("cpu")

    model = build_model()

    for parameter in model.parameters():
        parameter.requires_grad = False

    sparse_block = (
        model.model.layers[0].mlp
    )

    original_experts = (
        sparse_block.experts
    )

    qwen_bridge = (
        Qwen2MoeTensorExpertAdapter(
            original_experts,
            expert_index=TARGET_EXPERT,
        )
    )

    standalone = (
        qwen_bridge.materialize()
    )

    generic_adapter = (
        MiniMoEExpertAdapter(
            standalone
        )
    )

    inject_lora(
        generic_adapter,
        rank=RANK,
        alpha=ALPHA,
    )

    hybrid = HybridExperts(
        original_experts,
        standalone,
        target_expert=TARGET_EXPERT,
    )

    sparse_block.experts = hybrid

    model = model.to(device)

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    trainable_params = sum(
        p.numel()
        for p in trainable
    )

    original_e0_before = (
        original_experts
        .gate_up_proj[0]
        .detach()
        .clone()
    )

    original_e1_before = (
        original_experts
        .gate_up_proj[1]
        .detach()
        .clone()
    )

    router_before = (
        sparse_block.gate.weight
        .detach()
        .clone()
    )

    val_before = validation_loss(
        model,
        domain=0,
    )

    pool = build_split_tensor(
        0,
        "train",
        16,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=0.0,
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

    val_after = validation_loss(
        model,
        domain=0,
    )

    lora_change = adapter_change(
        standalone
    )

    e0_base_change = (
        original_experts
        .gate_up_proj[0]
        - original_e0_before
    ).abs().max().item()

    e1_change = (
        original_experts
        .gate_up_proj[1]
        - original_e1_before
    ).abs().max().item()

    router_change = (
        sparse_block.gate.weight
        - router_before
    ).abs().max().item()

    print()
    print("FedEdgeMoE - M18B")
    print("Full Qwen2-MoE LoRA Gradient")
    print()

    print(
        "Trainable params:",
        f"{trainable_params:,}",
    )

    print(
        "Expert 0 routed tokens:",
        hybrid.routed_tokens,
    )

    print(
        "First train loss:",
        f"{first_loss:.6f}",
    )

    print(
        "Last train loss:",
        f"{last_loss:.6f}",
    )

    print(
        "D0 validation loss:",
        f"{val_before:.6f}",
        "->",
        f"{val_after:.6f}",
    )

    print(
        "LoRA max change:",
        lora_change,
    )

    print(
        "Original E0 base change:",
        e0_base_change,
    )

    print(
        "Other expert change:",
        e1_change,
    )

    print(
        "Router change:",
        router_change,
    )

    ok = (
        trainable_params == 1152
        and hybrid.routed_tokens > 0
        and lora_change > 0.0
        and e0_base_change == 0.0
        and e1_change == 0.0
        and router_change == 0.0
    )

    print()
    print(
        "FULL-FORWARD LORA OK:",
        ok,
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
                "trainable_params":
                    trainable_params,
                "routed_tokens":
                    hybrid.routed_tokens,
                "first_train_loss":
                    first_loss,
                "last_train_loss":
                    last_loss,
                "validation_before":
                    val_before,
                "validation_after":
                    val_after,
                "lora_change":
                    lora_change,
                "e0_base_change":
                    e0_base_change,
                "e1_change":
                    e1_change,
                "router_change":
                    router_change,
                "ok":
                    ok,
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
