import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.adapters.expert_adapter import (
    MappedExpertAdapter,
)
from ml.adapters.generic_lora import (
    inject_lora,
)


class ExternalStyleExpert(nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = nn.Linear(
            128,
            256,
            bias=False,
        )

        self.w3 = nn.Linear(
            128,
            256,
            bias=False,
        )

        self.w2 = nn.Linear(
            256,
            128,
            bias=False,
        )

    def forward(self, x):
        hidden = (
            F.silu(self.w1(x))
            * self.w3(x)
        )

        return self.w2(hidden)


def main():
    torch.manual_seed(20260927)

    expert = ExternalStyleExpert()

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

    layers = (
        adapter.projection_layers()
    )

    wrapped = inject_lora(
        adapter,
        rank=16,
        alpha=16.0,
    )

    # inject_lora logical isimleri setattr ile kullandığı için
    # external attribute'lara burada bağlamamız gerekiyor.
    expert.w1 = wrapped["gate_proj"]
    expert.w3 = wrapped["up_proj"]
    expert.w2 = wrapped["down_proj"]

    # inject_lora'nun eklediği geçici logical attrs kaldırılıyor.
    del expert.gate_proj
    del expert.up_proj
    del expert.down_proj

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

    print()
    print("FedEdgeMoE - M17C")
    print("Mapped External Expert")
    print()

    print(
        "Logical projections:",
        ", ".join(layers.keys()),
    )

    print(
        "External projections:",
        "w1, w3, w2",
    )

    print(
        "Trainable params:",
        f"{trainable:,}",
    )

    print(
        "Initial output diff:",
        diff,
    )

    print()

    print(
        "MAPPED EXPERT OK:",
        (
            trainable == 18_432
            and diff == 0.0
        ),
    )


if __name__ == "__main__":
    main()
