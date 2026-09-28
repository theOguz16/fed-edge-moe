# MoE (Mixture of Experts), bir büyük dil modelinin tamamını çalıştırmak yerine sadece ilgili görevi en iyi yapacak alt kısımlarını (expert) devreye sokan bir yapay zeka mimarisidir
# Self-attention, bir token'ın kendinden önceki token'lara bakıp bağlam oluşturmasını sağlar.

from dataclasses import dataclass

import torch
import torch.nn as nn

from ml.model.expert import ExpertMLP
from ml.model.router import TopKRouter


@dataclass
class MoEOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    expert_counts: torch.Tensor


class SparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
        )

        self.experts = nn.ModuleList([
            ExpertMLP(
                d_model=d_model,
                hidden_dim=expert_hidden_dim,
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> MoEOutput:
        batch_size, seq_len, d_model = x.shape

        route = self.router(x)

        flat_x = x.reshape(-1, d_model)

        flat_indices = route.topk_indices.reshape(
            -1,
            self.top_k,
        )

        flat_weights = route.topk_weights.reshape(
            -1,
            self.top_k,
        )

        output = torch.zeros_like(flat_x)

        for expert_id, expert in enumerate(self.experts):

            selected_mask = (
                flat_indices == expert_id
            )

            token_indices, slot_indices = selected_mask.nonzero(
                as_tuple=True
            )

            if token_indices.numel() == 0:
                continue

            expert_input = flat_x[token_indices]

            expert_output = expert(expert_input)

            weights = flat_weights[
                token_indices,
                slot_indices,
            ].unsqueeze(-1)

            weighted_output = expert_output * weights

            output.index_add_(
                0,
                token_indices,
                weighted_output,
            )

        output = output.reshape(
            batch_size,
            seq_len,
            d_model,
        )

        return MoEOutput(
            hidden_states=output,
            aux_loss=route.aux_loss,
            expert_counts=route.expert_counts,
        )
