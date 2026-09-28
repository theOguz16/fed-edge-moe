# Expert, MoE içindeki küçük uzman sinir ağıdır. Bizde her expert şimdilik bir FFN/MLP olacak:
# Multi-Layer Perceptron(MLS), birkaç doğrusal katmandan oluşan sinir ağı parçası. Transformer’larda attention’dan sonra token bilgisini dönüştüren temel yapılardan biri.


import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)
