# LM loss — Language Model Loss, modelin bir sonraki token tahminindeki hatası. Eğitimde temel hedefimiz bunu azaltmak.
# Weight initialization, neural network başlamadan önce parametrelerin hangi değerlerle başlatıldığıdır.

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.model.moe import SparseMoE


@dataclass
class MiniMoEOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    lm_loss: torch.Tensor | None
    router_aux_loss: torch.Tensor
    expert_counts: torch.Tensor


class TransformerMoEBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int,
    ):
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

        self.moe_norm = nn.LayerNorm(d_model)

        self.moe = SparseMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
        )

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[1]

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        attn_input = self.attn_norm(x)

        attn_output, _ = self.attention(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=causal_mask,
            need_weights=False,
        )

        x = x + attn_output

        moe_input = self.moe_norm(x)

        moe_output = self.moe(moe_input)

        x = x + moe_output.hidden_states

        return (
            x,
            moe_output.aux_loss,
            moe_output.expert_counts,
        )


class MiniMoELM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        max_seq_len: int = 32,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden_dim: int = 256,
        router_aux_loss_weight: float = 0.01,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.router_aux_loss_weight = router_aux_loss_weight

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

        self.blocks = nn.ModuleList([
            TransformerMoEBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_experts=num_experts,
                top_k=top_k,
                expert_hidden_dim=expert_hidden_dim,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> MiniMoEOutput:

        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError("Sequence too long")

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)[None, :, :]
        )

        aux_losses = []
        expert_counts = []

        for block in self.blocks:
            x, aux_loss, counts = block(x)

            aux_losses.append(aux_loss)
            expert_counts.append(counts)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        router_aux_loss = torch.stack(
            aux_losses
        ).mean()

        expert_counts = torch.stack(
            expert_counts
        )

        loss = None
        lm_loss = None

        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
            )

            loss = (
                lm_loss
                + self.router_aux_loss_weight
                * router_aux_loss
            )

        return MiniMoEOutput(
            logits=logits,
            loss=loss,
            lm_loss=lm_loss,
            router_aux_loss=router_aux_loss,
            expert_counts=expert_counts,
        )
