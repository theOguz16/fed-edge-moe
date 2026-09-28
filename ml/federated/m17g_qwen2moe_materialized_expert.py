import torch
import torch.nn.functional as F

from transformers import (
    Qwen2MoeConfig,
    Qwen2MoeModel,
)

from ml.adapters.qwen2moe_tensor_expert import (
    Qwen2MoeTensorExpertAdapter,
)


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


def main():
    torch.manual_seed(20260927)

    model = build_model()

    experts = (
        model.layers[0]
        .mlp.experts
    )

    bridge = Qwen2MoeTensorExpertAdapter(
        experts,
        expert_index=0,
    )

    standalone = bridge.materialize()

    params = sum(
        p.numel()
        for p in standalone.parameters()
    )

    x = torch.randn(
        23,
        64,
    )

    gate_up = experts.gate_up_proj[0]
    gate, up = gate_up.chunk(
        2,
        dim=0,
    )
    down = experts.down_proj[0]

    act_fn = getattr(
        experts,
        "act_fn",
        F.silu,
    )

    with torch.no_grad():
        reference = F.linear(
            act_fn(F.linear(x, gate))
            * F.linear(x, up),
            down,
        )

        materialized = standalone(x)

    output_diff = (
        reference - materialized
    ).abs().max().item()

    expert1_before = (
        experts.gate_up_proj[1]
        .detach()
        .clone()
    )

    with torch.no_grad():
        standalone.gate_proj.weight.add_(
            0.001
        )

    bridge.commit(standalone)

    committed_gate = (
        experts.gate_up_proj[0]
        .chunk(2, dim=0)[0]
    )

    commit_diff = (
        committed_gate
        - standalone.gate_proj.weight
    ).abs().max().item()

    expert1_change = (
        experts.gate_up_proj[1]
        - expert1_before
    ).abs().max().item()

    print()
    print("FedEdgeMoE - M17G")
    print("Qwen2 Materialized Expert Bridge")
    print()

    print(
        "Standalone params:",
        f"{params:,}",
    )
    print(
        "Initial output diff:",
        output_diff,
    )
    print(
        "Commit max diff:",
        commit_diff,
    )
    print(
        "Other expert change:",
        expert1_change,
    )

    ok = (
        params == 6144
        and output_diff < 1e-6
        and commit_diff == 0.0
        and expert1_change == 0.0
    )

    print()
    print(
        "QWEN2 MATERIALIZATION OK:",
        ok,
    )


if __name__ == "__main__":
    main()
