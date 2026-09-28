import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot


CHECKPOINT = "checkpoints/m10_candidates/global_step_0100"

STEPS = 120
BATCH_SIZE = 32
LR = 5e-4

RANKS = [8, 16]


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
def evaluate(model, domain, device):
    seq = build_split_tensor(
        domain,
        "validation",
        16,
    ).to(device)

    x = seq[:, :-1]
    y = seq[:, 1:]

    out = model(input_ids=x)

    pred = (
        out.logits[:, 1:, :]
        .argmax(-1)
    )

    labels = y[:, 1:]

    return float(
        (pred == labels)
        .float()
        .mean()
        .cpu()
    )


def low_rank(delta, rank):
    u, s, vh = torch.linalg.svd(
        delta.float(),
        full_matrices=False,
    )

    return (
        u[:, :rank]
        @ torch.diag(s[:rank])
        @ vh[:rank, :]
    )


def train_full(domain):
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

    expert = (
        model.blocks[0]
        .moe.experts[7]
    )

    before = {
        name: p.detach().cpu().clone()
        for name, p in expert.named_parameters()
    }

    for p in expert.parameters():
        p.requires_grad = True

    model = model.to(device)

    base_acc = evaluate(
        model,
        domain,
        device,
    )

    optimizer = torch.optim.AdamW(
        expert.parameters(),
        lr=LR,
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

        out = model(
            input_ids=x,
        )

        loss = F.cross_entropy(
            out.logits.reshape(-1, 64),
            y.reshape(-1),
        )

        loss.backward()
        optimizer.step()

    full_acc = evaluate(
        model,
        domain,
        device,
    )

    after = {
        name: p.detach().cpu().clone()
        for name, p in expert.named_parameters()
    }

    deltas = {
        name: after[name] - before[name]
        for name in before
    }

    return (
        base_acc,
        full_acc,
        before,
        deltas,
    )


def evaluate_oracle(
    domain,
    before,
    deltas,
    rank,
):
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

    expert = (
        model.blocks[0]
        .moe.experts[7]
    )

    params = dict(
        expert.named_parameters()
    )

    with torch.no_grad():
        for name in params:
            approx = low_rank(
                deltas[name],
                rank,
            )

            params[name].copy_(
                before[name]
                + approx
            )

    model = model.to(device)

    return evaluate(
        model,
        domain,
        device,
    )


def main():
    print()
    print("FedEdgeMoE - M16J")
    print("Oracle Low-Rank Delta")
    print()

    for domain in [0, 1]:
        (
            base_acc,
            full_acc,
            before,
            deltas,
        ) = train_full(domain)

        print(f"D{domain}")

        print(
            f"  FULL: "
            f"{(full_acc-base_acc)*100:+.2f}pp"
        )

        for rank in RANKS:
            oracle_acc = evaluate_oracle(
                domain,
                before,
                deltas,
                rank,
            )

            print(
                f"  rank-{rank} oracle: "
                f"{(oracle_acc-base_acc)*100:+.2f}pp"
            )

        print()


if __name__ == "__main__":
    main()
