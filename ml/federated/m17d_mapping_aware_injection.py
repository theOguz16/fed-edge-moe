import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.adapters.expert_adapter import (
    MappedExpertAdapter,
)
from ml.adapters.generic_lora import (
    inject_lora,
)


class ExternalExpert(nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = nn.Linear(
            128, 256, bias=False
        )

        self.w3 = nn.Linear(
            128, 256, bias=False
        )

        self.w2 = nn.Linear(
            256, 128, bias=False
        )

    def forward(self, x):
        return self.w2(
            F.silu(self.w1(x))
            * self.w3(x)
        )


def main():
    torch.manual_seed(20260927)

    expert = ExternalExpert()

    x = torch.randn(
        32,
        128,
    )

    with torch.no_grad():
        before = expert(x)

    adapter = MappedExpertAdapter(
        expert,
        {
            "gate_proj": "w1",
            "up_proj": "w3",
            "down_proj": "w2",
        },
    )

    inject_lora(
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

    logical_attrs_exist = any(
        hasattr(expert, name)
        for name in [
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    print()
    print("FedEdgeMoE - M17D")
    print("Mapping-Aware LoRA Injection")
    print()

    print(
        "External w1 wrapped:",
        type(expert.w1).__name__,
    )

    print(
        "External w3 wrapped:",
        type(expert.w3).__name__,
    )

    print(
        "External w2 wrapped:",
        type(expert.w2).__name__,
    )

    print(
        "Fake logical attrs created:",
        logical_attrs_exist,
    )

    print(
        "Trainable params:",
        f"{trainable:,}",
    )

    print(
        "Initial output diff:",
        diff,
    )

    ok = (
        trainable == 18_432
        and diff == 0.0
        and not logical_attrs_exist
    )

    print()
    print(
        "MAPPING-AWARE INJECTION OK:",
        ok,
    )


if __name__ == "__main__":
    main()
