import torch

from ml.model.expert import ExpertMLP
from ml.adapters.expert_adapter import (
    MiniMoEExpertAdapter,
)
from ml.adapters.generic_lora import (
    inject_lora,
    lora_state_dict,
)


def main():
    torch.manual_seed(20260927)

    expert = ExpertMLP(
        d_model=128,
        hidden_dim=256,
    )

    x = torch.randn(
        32,
        128,
    )

    with torch.no_grad():
        before = expert(x)

    adapter = MiniMoEExpertAdapter(
        expert
    )

    wrapped = inject_lora(
        adapter,
        rank=16,
        alpha=16.0,
    )

    with torch.no_grad():
        after = expert(x)

    diff = (
        before - after
    ).abs().max().item()

    trainable = sum(
        p.numel()
        for p in expert.parameters()
        if p.requires_grad
    )

    lora_state = lora_state_dict(
        expert
    )

    lora_params = sum(
        t.numel()
        for t in lora_state.values()
    )

    print()
    print("FedEdgeMoE - M17B")
    print("Generic LoRA Injection")
    print()

    print(
        "Wrapped projections:",
        ", ".join(wrapped.keys()),
    )

    print(
        "Trainable params:",
        f"{trainable:,}",
    )

    print(
        "LoRA params:",
        f"{lora_params:,}",
    )

    print(
        "Initial output diff:",
        diff,
    )

    ok = (
        len(wrapped) == 3
        and trainable == 18_432
        and lora_params == 18_432
        and diff == 0.0
    )

    print()
    print(
        "GENERIC LORA OK:",
        ok,
    )


if __name__ == "__main__":
    main()
