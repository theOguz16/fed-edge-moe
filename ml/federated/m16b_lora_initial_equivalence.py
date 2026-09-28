import torch
from safetensors.torch import load_file

from ml.model.expert import ExpertMLP
from ml.model.lora_expert import (
    LoRAExpertMLP,
)


SHARD = (
    "checkpoints/m13a/v0100_L0_E7/"
    "experts/layer_00/"
    "expert_07_v0100.safetensors"
)


def main():
    torch.manual_seed(20260927)

    base = ExpertMLP(
        d_model=128,
        hidden_dim=256,
    )

    base.load_state_dict(
        load_file(SHARD)
    )

    base.eval()

    lora = LoRAExpertMLP(
        base,
        rank=8,
        alpha=8.0,
    )

    lora.eval()

    x = torch.randn(
        64,
        128,
    )

    with torch.no_grad():
        base_output = base(x)
        lora_output = lora(x)

    max_diff = (
        base_output
        - lora_output
    ).abs().max().item()

    total_params = sum(
        p.numel()
        for p in lora.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in lora.parameters()
        if p.requires_grad
    )

    adapter = (
        lora.adapter_state_dict()
    )

    adapter_params = sum(
        t.numel()
        for t in adapter.values()
    )

    base_trainable = any(
        p.requires_grad
        for name, p
        in lora.named_parameters()
        if ".base." in name
    )

    print()
    print("FedEdgeMoE - M16B")
    print("LoRA Initial Equivalence")
    print()

    print(
        "Total wrapped params:",
        f"{total_params:,}",
    )

    print(
        "Trainable params:",
        f"{trainable_params:,}",
    )

    print(
        "Adapter params:",
        f"{adapter_params:,}",
    )

    print(
        "Base weights trainable:",
        base_trainable,
    )

    print(
        "Initial max difference:",
        max_diff,
    )

    print()
    print(
        "LORA INITIAL EQUIVALENT:",
        (
            max_diff == 0.0
            and trainable_params == 9216
            and adapter_params == 9216
            and not base_trainable
        ),
    )


if __name__ == "__main__":
    main()
