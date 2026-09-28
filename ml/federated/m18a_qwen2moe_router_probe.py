import torch

from transformers import (
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
)

from ml.training.synthetic_v2 import (
    build_split_tensor,
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

        output_router_logits=True,
        use_cache=False,
    )

    return Qwen2MoeForCausalLM(
        config
    )


@torch.no_grad()
def probe_domain(
    model,
    domain,
):
    seq = build_split_tensor(
        domain,
        "validation",
        16,
    )[:64]

    x = seq[:, :-1]

    out = model(
        input_ids=x,
        output_router_logits=True,
    )

    counts = []

    for layer_id, logits in enumerate(
        out.router_logits
    ):
        flat = logits.reshape(
            -1,
            logits.shape[-1],
        )

        topk = torch.topk(
            flat,
            k=2,
            dim=-1,
        ).indices

        layer_counts = torch.zeros(
            4,
            dtype=torch.long,
        )

        for expert_id in range(4):
            layer_counts[expert_id] = (
                topk == expert_id
            ).sum()

        counts.append(
            layer_counts
        )

    return counts


def main():
    torch.manual_seed(20260927)

    model = build_model()
    model.eval()

    print()
    print("FedEdgeMoE - M18A")
    print("Tiny Qwen2-MoE Router Probe")
    print()

    total_e0 = 0

    for domain in [0, 1]:
        counts = probe_domain(
            model,
            domain,
        )

        print(f"D{domain}")

        for layer_id, c in enumerate(
            counts
        ):
            total = int(c.sum())

            values = [
                int(x)
                for x in c
            ]

            e0_ratio = (
                values[0]
                / total
                * 100
            )

            total_e0 += values[0]

            print(
                f"  Layer {layer_id}: "
                f"E0={values[0]} "
                f"E1={values[1]} "
                f"E2={values[2]} "
                f"E3={values[3]} "
                f"| E0={e0_ratio:.1f}%"
            )

        print()

    print(
        "Expert 0 routed tokens:",
        total_e0,
    )

    print(
        "ROUTER PROBE OK:",
        total_e0 > 0,
    )


if __name__ == "__main__":
    main()
