import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.model.lora_expert import LoRAExpertMLP
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot


SEED = 20260927

CHECKPOINT = (
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

TARGET_LAYER = 0
TARGET_EXPERT = 7

STEPS = 120
BATCH_SIZE = 32
SEQ_LEN = 16
LR = 5e-4

OUTPUT = Path(
    "results/m16f_lora_client_diagnostic.json"
)


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


def attach_lora(model):
    base = (
        model.blocks[TARGET_LAYER]
        .moe.experts[TARGET_EXPERT]
    )

    lora = LoRAExpertMLP(
        base,
        rank=8,
        alpha=8.0,
    )

    model.blocks[TARGET_LAYER].moe.experts[
        TARGET_EXPERT
    ] = lora

    return lora


def adapter_state(expert):
    return {
        k: v.detach().cpu().clone()
        for k, v
        in expert.adapter_state_dict().items()
    }


def load_adapter(expert, state):
    params = dict(
        expert.named_parameters()
    )

    for name, tensor in state.items():
        params[name].data.copy_(
            tensor.to(
                params[name].device
            )
        )


def subtract(a, b):
    return {
        k: a[k] - b[k]
        for k in a
    }


def flatten(state):
    return torch.cat([
        value.reshape(-1)
        for value in state.values()
    ])


def train_client(
    domain,
    initial_adapter,
):
    torch.manual_seed(
        SEED + domain
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

    expert = attach_lora(model)

    load_adapter(
        expert,
        initial_adapter,
    )

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=LR,
        weight_decay=0.0,
    )

    pool = build_split_tensor(
        domain,
        "train",
        SEQ_LEN,
    )

    model.train()

    first_loss = None
    last_loss = None

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

        output = model(
            input_ids=x,
        )

        loss = F.cross_entropy(
            output.logits.reshape(
                -1,
                64,
            ),
            y.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        value = float(
            loss.detach().cpu()
        )

        if first_loss is None:
            first_loss = value

        last_loss = value

    final_adapter = adapter_state(
        expert
    )

    delta = subtract(
        final_adapter,
        initial_adapter,
    )

    return (
        final_adapter,
        delta,
        first_loss,
        last_loss,
    )


@torch.no_grad()
def evaluate(adapter):
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

    expert = attach_lora(model)

    load_adapter(
        expert,
        adapter,
    )

    model = model.to(device)
    model.eval()

    accuracies = []

    for domain in range(4):
        seq = build_split_tensor(
            domain,
            "validation",
            SEQ_LEN,
        ).to(device)

        x = seq[:, :-1]
        y = seq[:, 1:]

        output = model(
            input_ids=x,
        )

        logits = output.logits[
            :, 1:, :
        ]

        labels = y[:, 1:]

        acc = (
            logits.argmax(dim=-1)
            == labels
        ).float().mean()

        accuracies.append(
            float(acc.cpu())
        )

    return accuracies


def main():
    torch.manual_seed(SEED)

    initial_model = build_model()

    load_global_snapshot(
        initial_model,
        CHECKPOINT,
    )

    initial_expert = attach_lora(
        initial_model
    )

    initial_adapter = adapter_state(
        initial_expert
    )

    base_eval = evaluate(
        initial_adapter
    )

    (
        adapter_a,
        delta_a,
        loss_a0,
        loss_a1,
    ) = train_client(
        0,
        initial_adapter,
    )

    (
        adapter_b,
        delta_b,
        loss_b0,
        loss_b1,
    ) = train_client(
        1,
        initial_adapter,
    )

    eval_a = evaluate(
        adapter_a
    )

    eval_b = evaluate(
        adapter_b
    )

    flat_a = flatten(
        delta_a
    )

    flat_b = flatten(
        delta_b
    )

    cosine = (
        F.cosine_similarity(
            flat_a.unsqueeze(0),
            flat_b.unsqueeze(0),
        ).item()
    )

    print()
    print("FedEdgeMoE - M16F")
    print("Client-Level LoRA Diagnostic")
    print()

    print(
        f"Client A loss: "
        f"{loss_a0:.6f} -> "
        f"{loss_a1:.6f}"
    )

    print(
        f"Client B loss: "
        f"{loss_b0:.6f} -> "
        f"{loss_b1:.6f}"
    )

    print()

    print(
        "Client A validation:"
    )

    for i in range(4):
        print(
            f"D{i}: "
            f"{base_eval[i]*100:6.2f}% "
            f"-> "
            f"{eval_a[i]*100:6.2f}% "
            f"({(eval_a[i]-base_eval[i])*100:+.2f}pp)"
        )

    print()

    print(
        "Client B validation:"
    )

    for i in range(4):
        print(
            f"D{i}: "
            f"{base_eval[i]*100:6.2f}% "
            f"-> "
            f"{eval_b[i]*100:6.2f}% "
            f"({(eval_b[i]-base_eval[i])*100:+.2f}pp)"
        )

    print()

    print(
        "LoRA update cosine:",
        f"{cosine:+.6f}",
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "base_validation":
                    base_eval,
                "client_a_validation":
                    eval_a,
                "client_b_validation":
                    eval_b,
                "client_a_loss": [
                    loss_a0,
                    loss_a1,
                ],
                "client_b_loss": [
                    loss_b0,
                    loss_b1,
                ],
                "update_cosine":
                    cosine,
            },
            f,
            indent=2,
        )

    print(
        "Report:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
