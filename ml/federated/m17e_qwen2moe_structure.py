from transformers import (
    Qwen2MoeConfig,
    Qwen2MoeModel,
)


def main():
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

    model = Qwen2MoeModel(config)

    block = model.layers[0].mlp
    experts = block.experts

    print()
    print("FedEdgeMoE - M17E")
    print("Qwen2-MoE Structure Inspection")
    print()

    print(
        "MLP block type:",
        type(block).__name__,
    )

    print(
        "Experts container:",
        type(experts).__name__,
    )

    print(
        "Number of experts:",
        experts.num_experts,
    )

    print(
        "gate_up_proj:",
        tuple(
            experts.gate_up_proj.shape
        ),
    )

    print(
        "down_proj:",
        tuple(
            experts.down_proj.shape
        ),
    )

    e0_gate_up = (
        experts.gate_up_proj[0]
    )

    e0_down = (
        experts.down_proj[0]
    )

    gate, up = e0_gate_up.chunk(
        2,
        dim=0,
    )

    print()
    print("Expert 0 slices:")

    print(
        "gate:",
        tuple(gate.shape),
    )

    print(
        "up:",
        tuple(up.shape),
    )

    print(
        "down:",
        tuple(e0_down.shape),
    )

    expert_params = (
        gate.numel()
        + up.numel()
        + e0_down.numel()
    )

    print(
        "Expert 0 params:",
        f"{expert_params:,}",
    )

    ok = (
        experts.num_experts == 4
        and tuple(gate.shape)
        == (32, 64)
        and tuple(up.shape)
        == (32, 64)
        and tuple(e0_down.shape)
        == (64, 32)
    )

    print()
    print(
        "QWEN2 MOE STRUCTURE OK:",
        ok,
    )


if __name__ == "__main__":
    main()
