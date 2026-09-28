import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
    ):
        super().__init__()

        self.base = base
        self.rank = rank
        self.scaling = alpha / rank

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
            )
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=5 ** 0.5,
        )

    def forward(self, x):
        base_out = self.base(x)

        lora_out = F.linear(
            F.linear(
                x,
                self.lora_A,
            ),
            self.lora_B,
        )

        return (
            base_out
            + self.scaling * lora_out
        )


class LoRAExpertMLP(nn.Module):
    def __init__(
        self,
        base_expert,
        rank=8,
        alpha=8.0,
    ):
        super().__init__()

        self.gate_proj = LoRALinear(
            base_expert.gate_proj,
            rank,
            alpha,
        )

        self.up_proj = LoRALinear(
            base_expert.up_proj,
            rank,
            alpha,
        )

        self.down_proj = LoRALinear(
            base_expert.down_proj,
            rank,
            alpha,
        )

    def forward(self, x):
        gated = (
            F.silu(
                self.gate_proj(x)
            )
            * self.up_proj(x)
        )

        return self.down_proj(
            gated
        )

    def adapter_state_dict(self):
        return {
            name: tensor
            for name, tensor
            in self.state_dict().items()
            if "lora_" in name
        }
