from pathlib import Path

import torch
from safetensors.torch import (
    save_file,
    load_file,
)

from transformers import (
    Qwen2MoeConfig,
    Qwen2MoeModel,
)


OUTPUT = Path(
    "checkpoints/m17f/"
    "qwen2moe_L0_E0.safetensors"
)


def build_model(seed):
    torch.manual_seed(seed)

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


def extract_expert(model):
    experts = (
        model.layers[0]
        .mlp.experts
    )

    gate_up = (
        experts.gate_up_proj[0]
        .detach()
        .cpu()
        .clone()
    )

    gate, up = gate_up.chunk(
        2,
        dim=0,
    )

    down = (
        experts.down_proj[0]
        .detach()
        .cpu()
        .clone()
    )

    return {
        "gate_proj": gate,
        "up_proj": up,
        "down_proj": down,
    }


def load_expert(model, shard):
    experts = (
        model.layers[0]
        .mlp.experts
    )

    gate_up = torch.cat(
        [
            shard["gate_proj"],
            shard["up_proj"],
        ],
        dim=0,
    )

    with torch.no_grad():
        experts.gate_up_proj[0].copy_(
            gate_up
        )

        experts.down_proj[0].copy_(
            shard["down_proj"]
        )


def max_diff(a, b):
    return max(
        (
            a[key] - b[key]
        ).abs().max().item()
        for key in a
    )


def main():
    source = build_model(
        seed=20260927
    )

    source_shard = extract_expert(
        source
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        source_shard,
        str(OUTPUT),
    )

    loaded_shard = load_file(
        str(OUTPUT)
    )

    fresh = build_model(
        seed=12345
    )

    before_other = (
        fresh.layers[0]
        .mlp.experts
        .gate_up_proj[1]
        .detach()
        .cpu()
        .clone()
    )

    load_expert(
        fresh,
        loaded_shard,
    )

    reconstructed = extract_expert(
        fresh
    )

    after_other = (
        fresh.layers[0]
        .mlp.experts
        .gate_up_proj[1]
        .detach()
        .cpu()
        .clone()
    )

    expert_diff = max_diff(
        source_shard,
        reconstructed,
    )

    other_diff = (
        before_other
        - after_other
    ).abs().max().item()

    params = sum(
        tensor.numel()
        for tensor
        in source_shard.values()
    )

    size_bytes = (
        OUTPUT.stat().st_size
    )

    print()
    print("FedEdgeMoE - M17F")
    print("Qwen2-MoE Expert Shard")
    print()

    print(
        "Shard:",
        "Layer 0 / Expert 0",
    )

    print(
        "Parameters:",
        f"{params:,}",
    )

    print(
        "Shard bytes:",
        f"{size_bytes:,}",
    )

    print(
        "Reload max diff:",
        expert_diff,
    )

    print(
        "Other expert change:",
        other_diff,
    )

    ok = (
        params == 6144
        and expert_diff == 0.0
        and other_diff == 0.0
    )

    print()
    print(
        "QWEN2 EXPERT SHARD EXACT:",
        ok,
    )

    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
