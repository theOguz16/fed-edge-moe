import torch
import torch.nn as nn
import torch.nn.functional as F


class MaterializedQwen2MoeExpert(nn.Module):
    def __init__(
        self,
        hidden_size,
        moe_hidden_size,
        act_fn,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_size,
            moe_hidden_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            moe_hidden_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            moe_hidden_size,
            hidden_size,
            bias=False,
        )

        self.act_fn = act_fn

    def forward(self, x):
        hidden = (
            self.act_fn(self.gate_proj(x))
            * self.up_proj(x)
        )
        return self.down_proj(hidden)


class Qwen2MoeTensorExpertAdapter:
    def __init__(
        self,
        experts,
        expert_index,
    ):
        self.experts = experts
        self.expert_index = expert_index

        self.act_fn = getattr(
            experts,
            "act_fn",
            F.silu,
        )

    def materialize(self):
        gate_up = (
            self.experts
            .gate_up_proj[self.expert_index]
            .detach()
        )

        gate, up = gate_up.chunk(
            2,
            dim=0,
        )

        down = (
            self.experts
            .down_proj[self.expert_index]
            .detach()
        )

        moe_hidden = gate.shape[0]
        hidden = gate.shape[1]

        module = MaterializedQwen2MoeExpert(
            hidden_size=hidden,
            moe_hidden_size=moe_hidden,
            act_fn=self.act_fn,
        )

        with torch.no_grad():
            module.gate_proj.weight.copy_(gate)
            module.up_proj.weight.copy_(up)
            module.down_proj.weight.copy_(down)

        return module

    def commit(self, module):
        gate_up = torch.cat(
            [
                module.gate_proj.weight,
                module.up_proj.weight,
            ],
            dim=0,
        )

        with torch.no_grad():
            self.experts.gate_up_proj[
                self.expert_index
            ].copy_(gate_up)

            self.experts.down_proj[
                self.expert_index
            ].copy_(
                module.down_proj.weight
            )
