import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import (
    load_file,
    save_file,
)

from ml.model.transformer import MiniMoELM
from ml.model.expert import ExpertMLP
from ml.training.synthetic_v2 import (
    generate_balanced_batch,
)
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
)
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

WORK = Path(
    "checkpoints/m13e_transport"
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
        seq_len
    )

    x = (
        model.token_embedding(input_ids)
        + model.position_embedding(
            positions
        )[None, :, :]
    )

    block = model.blocks[0]

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
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


def edge_execute():
    job = load_file(
        str(WORK / "job.safetensors")
    )

    expert = ExpertMLP(
        d_model=128,
        hidden_dim=256,
    )

    expert.load_state_dict(
        load_file(EXPERT_SHARD)
    )

    expert.eval()

    with torch.no_grad():
        output = expert(
            job["expert_input"]
        )

    save_file(
        {
            "expert_output":
                output.contiguous(),
        },
        str(
            WORK
            / "result.safetensors"
        ),
    )


def main():
    if WORK.exists():
        shutil.rmtree(WORK)

    WORK.mkdir(
        parents=True
    )

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

    block = model.blocks[0]

    with torch.no_grad():
        route = block.moe.router(
            moe_input
        )

        job = prepare_expert_job(
            moe_input,
            route,
            TARGET_EXPERT,
        )

        reference = (
            block.moe.experts[
                TARGET_EXPERT
            ](
                job.expert_input
            )
        )

    save_file(
        {
            "expert_input":
                job.expert_input.contiguous(),
        },
        str(
            WORK
            / "job.safetensors"
        ),
    )

    with open(
        WORK / "job.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "expert_id":
                    TARGET_EXPERT,
                "routed_tokens":
                    int(
                        job.token_indices.numel()
                    ),
            },
            f,
            indent=2,
        )

    edge_execute()

    result = load_file(
        str(
            WORK
            / "result.safetensors"
        )
    )["expert_output"]

    max_diff = (
        reference
        - result
    ).abs().max().item()

    job_bytes = (
        WORK
        / "job.safetensors"
    ).stat().st_size

    result_bytes = (
        WORK
        / "result.safetensors"
    ).stat().st_size

    print()
    print("FedEdgeMoE - M13E")
    print("Edge Shard Execution Boundary")
    print()

    print(
        "Edge loaded full model:",
        False,
    )

    print(
        "Edge loaded expert:",
        f"L{TARGET_LAYER}-E{TARGET_EXPERT}",
    )

    print(
        "Routed tokens:",
        job.token_indices.numel(),
    )

    print(
        "Job payload:",
        f"{job_bytes / 1000:.1f} KB",
    )

    print(
        "Result payload:",
        f"{result_bytes / 1000:.1f} KB",
    )

    print(
        "Max difference:",
        max_diff,
    )

    print(
        "EDGE EXECUTION EXACT:",
        max_diff < 1e-6,
    )


if __name__ == "__main__":
    main()
