import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.model.lora_expert import LoRAExpertMLP
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot


CHECKPOINT = "checkpoints/m10_candidates/global_step_0100"

RANK = 16
ALPHA = 16.0
STEPS = 480
LR = 2.5e-4
BATCH_SIZE = 32

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

    lora = LoRAExpertMLP(
        base,
        rank=RANK,
        alpha=ALPHA,
    )

    model.blocks[0].moe.experts[7] = lora

    return lora


def state(expert):
    return {
        k: v.detach().cpu().clone()
        for k, v
        in expert.adapter_state_dict().items()
    }


def load_state(expert, adapter):
    params = dict(
        expert.named_parameters()
    )

    for name, value in adapter.items():
        params[name].data.copy_(
            value.to(params[name].device)
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
    load_state(expert, adapter)

    model = model.to(device)
    model.eval()

    result = []

    for domain in range(4):
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

        acc = (
            (pred == labels)
            .float()
            .mean()
        )

        result.append(
            float(acc.cpu())
        )

    return result


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

    load_state(
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

    return state(expert)


def naive_average(a, b):
    return {
        k: (a[k] + b[k]) / 2.0
        for k in a
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

    B = u * sqrt_s.unsqueeze(0)

    A = (
        sqrt_s.unsqueeze(1)
        * vh
    )

    return A, B


def effective_delta_average(a, b):
    result = {}

    for prefix in PREFIXES:
        A_a = a[f"{prefix}.lora_A"]
        B_a = a[f"{prefix}.lora_B"]

        A_b = b[f"{prefix}.lora_A"]
        B_b = b[f"{prefix}.lora_B"]

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


def show(
    name,
    baseline,
    result,
):
    print(name)

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

    model = build_model()

    load_global_snapshot(
        model,
        CHECKPOINT,
    )

    initial_expert = attach_lora(
        model
    )

    initial = state(
        initial_expert
    )

    baseline = evaluate(
        initial
    )

    client_a = train_client(
        0,
        initial,
    )

    client_b = train_client(
        1,
        initial,
    )

    naive = naive_average(
        client_a,
        client_b,
    )

    effective = (
        effective_delta_average(
            client_a,
            client_b,
        )
    )

    eval_a = evaluate(
        client_a
    )

    eval_b = evaluate(
        client_b
    )

    eval_naive = evaluate(
        naive
    )

    eval_effective = evaluate(
        effective
    )

    print()
    print("FedEdgeMoE - M16L")
    print("LoRA Aggregation Comparison")
    print()

    show(
        "Client A",
        baseline,
        eval_a,
    )

    print()

    show(
        "Client B",
        baseline,
        eval_b,
    )

    print()

    show(
        "Naive A/B FedAvg",
        baseline,
        eval_naive,
    )

    print()

    show(
        "Effective Delta Avg",
        baseline,
        eval_effective,
    )


if __name__ == "__main__":
    main()
