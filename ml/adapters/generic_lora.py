import torch
import torch.nn as nn
import torch.nn.functional as F


class GenericLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
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


def inject_lora(
    expert_adapter,
    rank=16,
    alpha=16.0,
):
    layers = (
        expert_adapter
        .projection_layers()
    )

    wrapped = {}

    for name, layer in layers.items():
        lora_layer = GenericLoRALinear(
            layer,
            rank=rank,
            alpha=alpha,
        )

        expert_adapter.replace_projection(
            name,
            lora_layer,
        )

        wrapped[name] = lora_layer

    return wrapped

def lora_state_dict(expert):
    return {
        name: tensor
        for name, tensor
        in expert.state_dict().items()
        if "lora_" in name
    }
