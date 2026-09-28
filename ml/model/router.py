# Router her token için expert’lere skor veriyor
# Logits(Router’ın ham skorları.), softmax(Bu ham skorları toplamı 1 olan olasılık-benzeri değerlere çevirir) ile normalize ediliyor ve top-k seçiliyor. Seçilen expert’ler için token’lar normalize ediliyor ve expert’lerin yükü hesaplanıyor. Yük ve önem çarpımı ile aux loss hesaplanıyor.
# Top-k seçimi, token’ların hangi expert’lere yönlendirileceğini belirler. Bu sayede her token sadece en uygun expert’lere yönlendirilir ve modelin verimliliği artırılır. Bizde topk = 2

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    probabilities: torch.Tensor
    expert_counts: torch.Tensor
    aux_loss: torch.Tensor


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()

        if top_k > num_experts:
            raise ValueError("top_k, num_experts'tan buyuk olamaz")

        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        # x shape:
        # [batch, sequence, d_model]

        logits = self.gate(x)

        probabilities = F.softmax(logits, dim=-1)

        topk_weights, topk_indices = torch.topk(
            probabilities,
            k=self.top_k,
            dim=-1,
        )

        # Seçilen expert skorlarını kendi aralarında normalize et.
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1,
            keepdim=True,
        )

        selected = F.one_hot(
            topk_indices,
            num_classes=self.num_experts,
        ).float()

        expert_counts = selected.sum(dim=(0, 1, 2))

        total_routes = (
            x.shape[0]
            * x.shape[1]
            * self.top_k
        )

        load = expert_counts / total_routes

        importance = probabilities.mean(dim=(0, 1))

        aux_loss = self.num_experts * torch.sum(
            importance * load
        )

        return RouterOutput(
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            probabilities=probabilities,
            expert_counts=expert_counts,
            aux_loss=aux_loss,
        )
