import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot


CHECKPOINT = "checkpoints/m10_candidates/global_step_0100"

STEPS = 120
BATCH_SIZE = 32
LR = 5e-4

RANKS = [2, 4, 8, 16, 32, 64, 128]


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

        out = model(input_ids=x)

        loss = F.cross_entropy(
            out.logits.reshape(-1, 64),
            y.reshape(-1),
        )

        loss.backward()
        optimizer.step()

    after = {
        name: p.detach().cpu().clone()
        for name, p in expert.named_parameters()
    }

    return {
        name: after[name] - before[name]
        for name in before
    }


def energy_by_rank(delta):
    singular = torch.linalg.svdvals(
        delta.float()
    )

    energy = singular.square()
    total = energy.sum()

    results = {}

    for rank in RANKS:
        r = min(
            rank,
            len(singular),
        )

        captured = (
            energy[:r].sum()
            / total
            * 100
        )

        results[rank] = float(
            captured
        )

    return results


def main():
    print()
    print("FedEdgeMoE - M16I")
    print("Full-Expert Delta Rank Analysis")
    print()

    for domain in [0, 1]:
        deltas = train_full(
            domain
        )

        print(f"D{domain}")

        for name, delta in deltas.items():
            energies = energy_by_rank(
                delta
            )

            print(
                f"  {name}"
            )

            print(
                "   "
                + " | ".join(
                    f"r{rank}="
                    f"{energies[rank]:5.1f}%"
                    for rank in RANKS
                )
            )

        print()


if __name__ == "__main__":
    main()
