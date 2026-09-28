import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

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
    "results/m16e_federated_lora.json"
)

ADAPTER_OUT = Path(
    "checkpoints/m16e/"
    "global_L0_E7_lora.safetensors"
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


def add(a, b):
    return {
        k: a[k] + b[k]
        for k in a
    }


def mean_delta(a, b):
    return {
        k: (
            a[k] + b[k]
        ) / 2.0
        for k in a
    }


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

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=0.0,
    )

    pool = build_split_tensor(
        domain,
        "train",
        SEQ_LEN,
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

    final_state = adapter_state(
        expert
    )

    return subtract(
        final_state,
        initial_adapter,
    )


@torch.no_grad()
def evaluate(
    adapter,
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

    delta_a = train_client(
        0,
        initial_adapter,
    )

    delta_b = train_client(
        1,
        initial_adapter,
    )

    aggregate_delta = mean_delta(
        delta_a,
        delta_b,
    )

    global_adapter = add(
        initial_adapter,
        aggregate_delta,
    )

    global_eval = evaluate(
        global_adapter
    )

    ADAPTER_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        global_adapter,
        str(ADAPTER_OUT),
    )

    adapter_bytes = (
        ADAPTER_OUT.stat().st_size
    )

    base_mean = (
        sum(base_eval) / 4
    )

    final_mean = (
        sum(global_eval) / 4
    )

    print()
    print("FedEdgeMoE - M16E")
    print("Federated LoRA Aggregation")
    print()

    print(
        "Trainable params:",
        "9,216",
    )

    print(
        "Client uploads:",
        f"{adapter_bytes * 2 / 1000:.1f} KB",
    )

    print()

    for i in range(4):
        print(
            f"D{i}: "
            f"{base_eval[i]*100:6.2f}% "
            f"-> "
            f"{global_eval[i]*100:6.2f}% "
            f"({(global_eval[i]-base_eval[i])*100:+.2f}pp)"
        )

    print()

    print(
        "Mean:",
        f"{base_mean*100:.2f}% "
        f"-> {final_mean*100:.2f}% "
        f"({(final_mean-base_mean)*100:+.2f}pp)"
    )

    print(
        "Adapter bytes:",
        f"{adapter_bytes:,}",
    )

    print(
        "Full-expert-equivalent uploads:",
        f"{393472 * 2:,}",
    )

    print(
        "Upload reduction:",
        f"{(1 - (adapter_bytes*2)/(393472*2))*100:.1f}%",
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
                "global_validation":
                    global_eval,
                "base_mean":
                    base_mean,
                "global_mean":
                    final_mean,
                "adapter_bytes":
                    adapter_bytes,
                "upload_reduction":
                    (
                        1
                        - adapter_bytes
                        / 393472
                    ),
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
