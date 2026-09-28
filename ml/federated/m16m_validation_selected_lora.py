import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.model.lora_expert import LoRAExpertMLP
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot


CHECKPOINT = "checkpoints/m10_candidates/global_step_0100"

RANK = 16
ALPHA = 16.0
LR = 2.5e-4
BATCH_SIZE = 32

CHECK_STEPS = [30, 60, 120, 240, 480]

REPORT = Path(
    "results/m16m_validation_selected_lora.json"
)

PREFIXES = [
    "gate_proj",
    "up_proj",
    "down_proj",
]


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
    base = model.blocks[0].moe.experts[7]

    expert = LoRAExpertMLP(
        base,
        rank=RANK,
        alpha=ALPHA,
    )

    model.blocks[0].moe.experts[7] = expert

    return expert


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

    for name, value in state.items():
        params[name].data.copy_(
            value.to(params[name].device)
        )


@torch.no_grad()
def eval_domain(
    model,
    domain,
    device,
):
    seq = build_split_tensor(
        domain,
        "validation",
        16,
    ).to(device)

    x = seq[:, :-1]
    y = seq[:, 1:]

    out = model(
        input_ids=x,
    )

    pred = (
        out.logits[:, 1:, :]
        .argmax(dim=-1)
    )

    labels = y[:, 1:]

    return float(
        (pred == labels)
        .float()
        .mean()
        .cpu()
    )


@torch.no_grad()
def evaluate_adapter(adapter):
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

    return [
        eval_domain(
            model,
            domain,
            device,
        )
        for domain in range(4)
    ]


def train_client(
    domain,
    initial_adapter,
):
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
        16,
    )

    baseline = eval_domain(
        model,
        domain,
        device,
    )

    best_score = baseline
    best_step = 0
    best_adapter = adapter_state(
        expert
    )

    history = []

    model.train()

    for step in range(
        1,
        max(CHECK_STEPS) + 1,
    ):
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

        if step in CHECK_STEPS:
            model.eval()

            score = eval_domain(
                model,
                domain,
                device,
            )

            history.append({
                "step": step,
                "validation": score,
            })

            if score > best_score:
                best_score = score
                best_step = step
                best_adapter = adapter_state(
                    expert
                )

            model.train()

    final_adapter = adapter_state(
        expert
    )

    return {
        "baseline": baseline,
        "best_step": best_step,
        "best_score": best_score,
        "best_adapter": best_adapter,
        "final_adapter": final_adapter,
        "history": history,
    }


def factorize_delta(delta):
    u, s, vh = torch.linalg.svd(
        delta.float(),
        full_matrices=False,
    )

    u = u[:, :RANK]
    s = s[:RANK]
    vh = vh[:RANK, :]

    sqrt_s = torch.sqrt(s)

    B = (
        u
        * sqrt_s.unsqueeze(0)
    )

    A = (
        sqrt_s.unsqueeze(1)
        * vh
    )

    return A, B


def effective_delta_average(
    adapter_a,
    adapter_b,
):
    result = {}

    for prefix in PREFIXES:
        A_a = adapter_a[
            f"{prefix}.lora_A"
        ]

        B_a = adapter_a[
            f"{prefix}.lora_B"
        ]

        A_b = adapter_b[
            f"{prefix}.lora_A"
        ]

        B_b = adapter_b[
            f"{prefix}.lora_B"
        ]

        delta_a = B_a @ A_a
        delta_b = B_b @ A_b

        delta_global = (
            delta_a + delta_b
        ) / 2.0

        A_global, B_global = (
            factorize_delta(
                delta_global
            )
        )

        result[
            f"{prefix}.lora_A"
        ] = A_global

        result[
            f"{prefix}.lora_B"
        ] = B_global

    return result


def print_eval(
    title,
    baseline,
    result,
):
    print(title)

    for i in range(4):
        gain = (
            result[i]
            - baseline[i]
        ) * 100

        print(
            f"  D{i}: "
            f"{result[i]*100:6.2f}% "
            f"({gain:+.2f}pp)"
        )

    mean_gain = (
        sum(result)
        - sum(baseline)
    ) / 4 * 100

    print(
        f"  Mean gain: "
        f"{mean_gain:+.2f}pp"
    )


def main():
    torch.manual_seed(20260927)

    base_model = build_model()

    load_global_snapshot(
        base_model,
        CHECKPOINT,
    )

    base_expert = attach_lora(
        base_model
    )

    initial_adapter = adapter_state(
        base_expert
    )

    baseline = evaluate_adapter(
        initial_adapter
    )

    client_a = train_client(
        0,
        initial_adapter,
    )

    client_b = train_client(
        1,
        initial_adapter,
    )

    selected_global = (
        effective_delta_average(
            client_a["best_adapter"],
            client_b["best_adapter"],
        )
    )

    final_global = (
        effective_delta_average(
            client_a["final_adapter"],
            client_b["final_adapter"],
        )
    )

    selected_eval = evaluate_adapter(
        selected_global
    )

    final_eval = evaluate_adapter(
        final_global
    )

    print()
    print("FedEdgeMoE - M16M")
    print("Validation-Selected Federated LoRA")
    print()

    print(
        "Client A selected step:",
        client_a["best_step"],
        "| target gain:",
        f"{(client_a['best_score'] - client_a['baseline'])*100:+.2f}pp",
    )

    print(
        "Client B selected step:",
        client_b["best_step"],
        "| target gain:",
        f"{(client_b['best_score'] - client_b['baseline'])*100:+.2f}pp",
    )

    print()

    print_eval(
        "Final-step aggregation (480)",
        baseline,
        final_eval,
    )

    print()

    print_eval(
        "Validation-selected aggregation",
        baseline,
        selected_eval,
    )

    final_mean = sum(final_eval) / 4
    selected_mean = sum(selected_eval) / 4

    print()

    print(
        "Selection advantage:",
        f"{(selected_mean-final_mean)*100:+.2f}pp",
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "client_a": {
                    "best_step":
                        client_a["best_step"],
                    "best_score":
                        client_a["best_score"],
                    "history":
                        client_a["history"],
                },
                "client_b": {
                    "best_step":
                        client_b["best_step"],
                    "best_score":
                        client_b["best_score"],
                    "history":
                        client_b["history"],
                },
                "baseline":
                    baseline,
                "final_480":
                    final_eval,
                "validation_selected":
                    selected_eval,
                "selection_advantage":
                    selected_mean
                    - final_mean,
            },
            f,
            indent=2,
        )

    print(
        "Report:",
        REPORT,
    )


if __name__ == "__main__":
    main()
