import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.model.lora_expert import LoRAExpertMLP
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot


CHECKPOINT = "checkpoints/m10_candidates/global_step_0100"

CONFIGS = [
    (8, 1e-4),
    (8, 2.5e-4),
    (8, 5e-4),
    (16, 1e-4),
    (16, 2.5e-4),
    (16, 5e-4),
]

STEPS = 120
BATCH_SIZE = 32


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


@torch.no_grad()
def eval_domain(model, domain, device):
    seq = build_split_tensor(
        domain,
        "validation",
        16,
    ).to(device)

    x = seq[:, :-1]
    y = seq[:, 1:]

    out = model(input_ids=x)

    pred = out.logits[:, 1:, :].argmax(-1)
    labels = y[:, 1:]

    return float(
        (pred == labels)
        .float()
        .mean()
        .cpu()
    )


def run(domain, rank, lr):
    torch.manual_seed(
        20260927 + domain
    )

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model = build_model()
    load_global_snapshot(
        model,
        CHECKPOINT,
    )

    for p in model.parameters():
        p.requires_grad = False

    base = model.blocks[0].moe.experts[7]

    lora = LoRAExpertMLP(
        base,
        rank=rank,
        alpha=float(rank),
    )

    model.blocks[0].moe.experts[7] = lora
    model = model.to(device)

    before = eval_domain(
        model,
        domain,
        device,
    )

    optimizer = torch.optim.AdamW(
        [
            p for p in model.parameters()
            if p.requires_grad
        ],
        lr=lr,
        weight_decay=0.0,
    )

    pool = build_split_tensor(
        domain,
        "train",
        16,
    )

    model.train()

    for _ in range(STEPS):
        idx = torch.randint(
            0,
            pool.shape[0],
            (BATCH_SIZE,),
        )

        seq = pool[idx].to(device)

        x = seq[:, :-1]
        y = seq[:, 1:].clone()
        y[:, 0] = -100

        optimizer.zero_grad(
            set_to_none=True
        )

        out = model(input_ids=x)

        loss = F.cross_entropy(
            out.logits.reshape(-1, 64),
            y.reshape(-1),
        )

        loss.backward()
        optimizer.step()

    after = eval_domain(
        model,
        domain,
        device,
    )

    return before, after


def main():
    print()
    print("FedEdgeMoE - M16G")
    print("LoRA Rank/LR Sensitivity")
    print()

    for rank, lr in CONFIGS:
        a0, a1 = run(0, rank, lr)
        b0, b1 = run(1, rank, lr)

        print(
            f"rank={rank:<2} "
            f"lr={lr:.1e} "
            f"| D0 {(a1-a0)*100:+5.2f}pp "
            f"| D1 {(b1-b0)*100:+5.2f}pp"
        )


if __name__ == "__main__":
    main()
