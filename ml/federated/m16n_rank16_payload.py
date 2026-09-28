from pathlib import Path

from safetensors.torch import save_file

from ml.model.transformer import MiniMoELM
from ml.model.lora_expert import LoRAExpertMLP
from ml.federated.checkpoint_manager import load_global_snapshot


CHECKPOINT = (
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

OUTPUT = Path(
    "checkpoints/m16n/"
    "L0_E7_lora_rank16.safetensors"
)

FULL_EXPERT_BYTES = 393_472

RANK = 16
ALPHA = 16.0


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


def main():
    model = build_model()

    load_global_snapshot(
        model,
        CHECKPOINT,
    )

    base_expert = (
        model.blocks[0]
        .moe.experts[7]
    )

    lora = LoRAExpertMLP(
        base_expert,
        rank=RANK,
        alpha=ALPHA,
    )

    adapter = {
        name: tensor.detach().cpu()
        for name, tensor
        in lora.adapter_state_dict().items()
    }

    adapter_params = sum(
        tensor.numel()
        for tensor in adapter.values()
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        adapter,
        str(OUTPUT),
    )

    adapter_bytes = (
        OUTPUT.stat().st_size
    )

    reduction = (
        1
        - adapter_bytes
        / FULL_EXPERT_BYTES
    )

    compression = (
        FULL_EXPERT_BYTES
        / adapter_bytes
    )

    two_client_bytes = (
        adapter_bytes * 2
    )

    print()
    print("FedEdgeMoE - M16N")
    print("Rank-16 LoRA Payload")
    print()

    print(
        "Adapter params:",
        f"{adapter_params:,}",
    )

    print(
        "Adapter bytes:",
        f"{adapter_bytes:,}",
    )

    print(
        "Full expert bytes:",
        f"{FULL_EXPERT_BYTES:,}",
    )

    print(
        "Payload reduction:",
        f"{reduction*100:.1f}%",
    )

    print(
        "Compression factor:",
        f"{compression:.2f}x",
    )

    print(
        "Two-client upload:",
        f"{two_client_bytes/1000:.1f} KB",
    )

    print()
    print(
        "RANK16 PAYLOAD OK:",
        (
            adapter_params == 18_432
            and adapter_bytes
            < FULL_EXPERT_BYTES
        ),
    )


if __name__ == "__main__":
    main()
