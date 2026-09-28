import torch
import torch.nn.functional as F

from transformers import (
    Qwen2MoeConfig,
    Qwen2MoeModel,
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


RANK = 4
ALPHA = 4.0
LR = 1e-3


def build_model():
    config = Qwen2MoeConfig(
        vocab_size=128,
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

    return Qwen2MoeModel(config)


def effective_weight(layer):
    return (
        layer.base.weight
        + layer.scaling
        * (
            layer.lora_B
            @ layer.lora_A
        )
    )


def main():
    torch.manual_seed(20260927)

    model = build_model()

    experts = (
        model.layers[0]
        .mlp.experts
    )

    expert1_before = (
        experts.gate_up_proj[1]
        .detach()
        .clone()
    )

    bridge = Qwen2MoeTensorExpertAdapter(
        experts,
        expert_index=0,
    )

    standalone = bridge.materialize()

    base_before = {
        name: p.detach().clone()
        for name, p
        in standalone.named_parameters()
    }

    adapter = MiniMoEExpertAdapter(
        standalone
    )

    inject_lora(
        adapter,
        rank=RANK,
        alpha=ALPHA,
    )

    trainable = [
        p
        for p in standalone.parameters()
        if p.requires_grad
    ]

    trainable_params = sum(
        p.numel()
        for p in trainable
    )

    x = torch.randn(
        64,
        64,
    )

    target = torch.randn(
        64,
        64,
    )

    with torch.no_grad():
        initial_output = standalone(x)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=0.0,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    output = standalone(x)

    loss = F.mse_loss(
        output,
        target,
    )

    loss.backward()
    optimizer.step()

    lora_change = max(
        standalone.gate_proj.lora_B
        .detach()
        .abs()
        .max()
        .item(),

        standalone.up_proj.lora_B
        .detach()
        .abs()
        .max()
        .item(),

        standalone.down_proj.lora_B
        .detach()
        .abs()
        .max()
        .item(),
    )

    base_after = {
        "gate_proj.weight":
            standalone.gate_proj
            .base.weight
            .detach()
            .clone(),

        "up_proj.weight":
            standalone.up_proj
            .base.weight
            .detach()
            .clone(),

        "down_proj.weight":
            standalone.down_proj
            .base.weight
            .detach()
            .clone(),
    }

    base_change = max(
        (
            base_before[name]
            - base_after[name]
        )
        .abs()
        .max()
        .item()
        for name in base_after
    )

    with torch.no_grad():
        trained_output = standalone(x)

    merged_gate = effective_weight(
        standalone.gate_proj
    )

    merged_up = effective_weight(
        standalone.up_proj
    )

    merged_down = effective_weight(
        standalone.down_proj
    )

    with torch.no_grad():
        merged_output = F.linear(
            F.silu(
                F.linear(
                    x,
                    merged_gate,
                )
            )
            * F.linear(
                x,
                merged_up,
            ),
            merged_down,
        )

    merge_output_diff = (
        trained_output
        - merged_output
    ).abs().max().item()

    with torch.no_grad():
        experts.gate_up_proj[0].copy_(
            torch.cat(
                [
                    merged_gate,
                    merged_up,
                ],
                dim=0,
            )
        )

        experts.down_proj[0].copy_(
            merged_down
        )

    committed_gate = (
        experts.gate_up_proj[0]
        .chunk(2, dim=0)[0]
    )

    commit_diff = (
        committed_gate
        - merged_gate
    ).abs().max().item()

    expert1_change = (
        experts.gate_up_proj[1]
        - expert1_before
    ).abs().max().item()

    output_change = (
        trained_output
        - initial_output
    ).abs().max().item()

    print()
    print("FedEdgeMoE - M17H")
    print("Qwen2-MoE Generic LoRA Pipeline")
    print()

    print(
        "LoRA rank:",
        RANK,
    )

    print(
        "Trainable params:",
        f"{trainable_params:,}",
    )

    print(
        "Training loss:",
        f"{loss.item():.6f}",
    )

    print(
        "Base weight change:",
        base_change,
    )

    print(
        "LoRA change:",
        lora_change,
    )

    print(
        "Expert output change:",
        output_change,
    )

    print(
        "Merged output diff:",
        merge_output_diff,
    )

    print(
        "Commit diff:",
        commit_diff,
    )

    print(
        "Other expert change:",
        expert1_change,
    )

    ok = (
        trainable_params == 1152
        and base_change == 0.0
        and lora_change > 0.0
        and output_change > 0.0
        and merge_output_diff < 1e-6
        and commit_diff == 0.0
        and expert1_change == 0.0
    )

    print()
    print(
        "QWEN2 LORA PIPELINE OK:",
        ok,
    )


if __name__ == "__main__":
    main()
