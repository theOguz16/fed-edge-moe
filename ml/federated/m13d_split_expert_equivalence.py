import torch

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v2 import (
    generate_balanced_batch,
)
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
)
from ml.distributed.expert_dispatch import (
    prepare_expert_job,
    run_expert_job,
    merge_expert_result,
)


CHECKPOINT = (
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

TARGET_LAYER = 0
TARGET_EXPERT = 7


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


def build_layer0_moe_input(
    model,
    input_ids,
):
    _, seq_len = input_ids.shape

    positions = torch.arange(
        seq_len,
        device=input_ids.device,
    )

    x = (
        model.token_embedding(
            input_ids
        )
        + model.position_embedding(
            positions
        )[None, :, :]
    )

    block = model.blocks[
        TARGET_LAYER
    ]

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=input_ids.device,
        ),
        diagonal=1,
    )

    attn_input = (
        block.attn_norm(x)
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

    x = x + attn_output

    return block.moe_norm(x)


def main():
    torch.manual_seed(20260927)

    model = build_model()

    load_global_snapshot(
        model,
        CHECKPOINT,
    )

    model.eval()

    input_ids, _ = (
        generate_balanced_batch(
            split="validation",
            batch_size=32,
            seq_len=16,
            device=torch.device("cpu"),
        )
    )

    moe_input = (
        build_layer0_moe_input(
            model,
            input_ids,
        )
    )

    block = model.blocks[
        TARGET_LAYER
    ]

    with torch.no_grad():
        reference = block.moe(
            moe_input
        )

        route = block.moe.router(
            moe_input
        )

        flat_output = torch.zeros_like(
            moe_input.reshape(
                -1,
                moe_input.shape[-1],
            )
        )

        target_routes = 0

        for expert_id, expert in enumerate(
            block.moe.experts
        ):
            job = prepare_expert_job(
                moe_input,
                route,
                expert_id,
            )

            if expert_id == TARGET_EXPERT:
                target_routes = (
                    job.token_indices.numel()
                )

            if (
                job.token_indices.numel()
                == 0
            ):
                continue

            expert_output = (
                run_expert_job(
                    expert,
                    job,
                )
            )

            flat_output = (
                merge_expert_result(
                    flat_output,
                    job,
                    expert_output,
                )
            )

        distributed = (
            flat_output.reshape_as(
                moe_input
            )
        )

    max_diff = (
        reference.hidden_states
        - distributed
    ).abs().max().item()

    exact = max_diff < 1e-6

    print()
    print("FedEdgeMoE - M13D")
    print("Split Expert Execution")
    print()

    print(
        "Target:",
        f"L{TARGET_LAYER}-E{TARGET_EXPERT}",
    )

    print(
        "Target routed tokens:",
        target_routes,
    )

    print(
        "Monolithic shape:",
        tuple(
            reference
            .hidden_states
            .shape
        ),
    )

    print(
        "Split shape:",
        tuple(
            distributed.shape
        ),
    )

    print(
        "Max difference:",
        max_diff,
    )

    print(
        "SPLIT EXECUTION EQUIVALENT:",
        exact,
    )


if __name__ == "__main__":
    main()
