from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ml.model.transformer import MiniMoELM
from ml.model.lora_expert import LoRAExpertMLP
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot


CHECKPOINT = (
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

OUTPUT = Path(
    "checkpoints/m16d/"
    "L0_E7_lora_rank8.safetensors"
)

STEPS = 60
LR = 5e-4


def build_model():
    return MiniMoELM(
        vocab_size=64,
        max_seq_len=32,
        d_model=128,
        num_heads=4,
        num_layers=3,
        num_experts=8,
        top_k=2,
        expert_hidden_dim=256,
        router_aux_loss_weight=0.01,
    )


def make_lora_expert():
    model = build_model()

    load_global_snapshot(
        model,
        CHECKPOINT,
    )

    base = (
        model.blocks[0]
        .moe.experts[7]
    )

    return LoRAExpertMLP(
        base,
        rank=8,
        alpha=8.0,
    )


def train(expert):
    pool = build_split_tensor(
        domain=1,
        split="train",
        seq_len=16,
    )

    optimizer = torch.optim.AdamW(
        [
            p for p in expert.parameters()
            if p.requires_grad
        ],
        lr=LR,
        weight_decay=0.0,
    )

    expert.train()

    for _ in range(STEPS):
        idx = torch.randint(
            0,
            pool.shape[0],
            (32,),
        )

        seq = pool[idx]

        x = torch.randn(
            32,
            16,
            128,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        out = expert(x)

        loss = (
            out.pow(2).mean()
        )

        loss.backward()
        optimizer.step()


def load_adapter(expert, state):
    params = dict(
        expert.named_parameters()
    )

    for name, tensor in state.items():
        params[name].data.copy_(
            tensor
        )


def main():
    torch.manual_seed(20260927)

    trained = make_lora_expert()
    train(trained)

    adapter = {
        k: v.detach().cpu()
        for k, v
        in trained.adapter_state_dict().items()
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        adapter,
        str(OUTPUT),
    )

    fresh = make_lora_expert()

    load_adapter(
        fresh,
        adapter,
    )

    x = torch.randn(
        64,
        128,
    )

    trained.eval()
    fresh.eval()

    with torch.no_grad():
        a = trained(x)
        b = fresh(x)

    max_diff = (
        a - b
    ).abs().max().item()

    full_expert_bytes = (
        393_472
    )

    adapter_bytes = (
        OUTPUT.stat().st_size
    )

    print()
    print("FedEdgeMoE - M16D")
    print("LoRA Adapter Checkpoint")
    print()

    print(
        "Adapter params:",
        sum(
            t.numel()
            for t in adapter.values()
        ),
    )

    print(
        "Adapter bytes:",
        f"{adapter_bytes:,}",
    )

    print(
        "Full expert bytes:",
        f"{full_expert_bytes:,}",
    )

    print(
        "Payload reduction:",
        f"{(1 - adapter_bytes/full_expert_bytes)*100:.1f}%",
    )

    print(
        "Reload max diff:",
        max_diff,
    )

    print()
    print(
        "ADAPTER CHECKPOINT EXACT:",
        max_diff == 0.0,
    )


if __name__ == "__main__":
    main()
