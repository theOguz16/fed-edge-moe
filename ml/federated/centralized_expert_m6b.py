import json
import random
from pathlib import Path

import torch
from safetensors.torch import save_file

from ml.model.transformer import MiniMoELM
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
    export_global_snapshot,
)


GLOBAL_BASE = "checkpoints/m5_candidates/global_step_0030"
EXPERT_FEDAVG = "checkpoints/m5c/global_v0031"

TARGET_LAYER = 0
TARGET_EXPERT = 6

STEPS = 240
BATCH_SIZE = 32
SEQ_LEN = 16
LEARNING_RATE = 5e-4

SEED = 2026

DOMAIN_NAMES = [
    "D0 (+1)",
    "D1 (+2)",
    "D2 (+3)",
    "D3 (+5)",
]

DOMAIN_STEPS = [1, 2, 3, 5]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def generate_domain_batch(
    domain,
    batch_size,
    seq_len,
    device,
):
    starts = torch.randint(
        0,
        16,
        (batch_size,),
        device=device,
    )

    positions = torch.arange(
        seq_len + 1,
        device=device,
    )

    base = domain * 16
    step_size = DOMAIN_STEPS[domain]

    sequences = (
        starts[:, None]
        + step_size * positions[None, :]
    ) % 16

    sequences = sequences + base

    return (
        sequences[:, :-1].long(),
        sequences[:, 1:].long(),
    )


def build_eval_set(domain, device):
    # 16 olası başlangıcın tamamı.
    starts = torch.arange(
        0,
        16,
        device=device,
    )

    positions = torch.arange(
        SEQ_LEN + 1,
        device=device,
    )

    base = domain * 16
    step_size = DOMAIN_STEPS[domain]

    sequences = (
        starts[:, None]
        + step_size * positions[None, :]
    ) % 16

    sequences = sequences + base

    return (
        sequences[:, :-1].long(),
        sequences[:, 1:].long(),
    )


@torch.no_grad()
def evaluate_domain(model, domain, device):
    model.eval()

    x, y = build_eval_set(
        domain,
        device,
    )

    output = model(
        input_ids=x,
        labels=y,
    )

    predictions = output.logits.argmax(
        dim=-1
    )

    accuracy = (
        predictions == y
    ).float().mean()

    return {
        "loss": float(
            output.lm_loss.detach().cpu()
        ),
        "accuracy": float(
            accuracy.detach().cpu()
        ),
    }


def evaluate_all(model, device):
    return [
        evaluate_domain(
            model,
            domain,
            device,
        )
        for domain in range(4)
    ]


def print_results(title, results):
    mean_loss = sum(
        r["loss"]
        for r in results
    ) / 4

    mean_accuracy = sum(
        r["accuracy"]
        for r in results
    ) / 4

    print()
    print(title)

    for domain, result in enumerate(results):
        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss={result['loss']:.6f} "
            f"acc={result['accuracy'] * 100:6.2f}%"
        )

    print(
        f"MEAN     "
        f"loss={mean_loss:.6f} "
        f"acc={mean_accuracy * 100:6.2f}%"
    )

    return mean_loss, mean_accuracy


def clone_expert_state(expert):
    return {
        key: value.detach().cpu().clone()
        for key, value
        in expert.state_dict().items()
    }


def expert_delta(before, after):
    return {
        key: (
            after[key] - before[key]
        ).contiguous()
        for key in before
    }


def delta_norm(delta):
    total = 0.0

    for tensor in delta.values():
        total += float(
            torch.sum(
                tensor.float() ** 2
            )
        )

    return total ** 0.5


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()

    print()
    print("FedEdgeMoE - M6B")
    print("Centralized Expert Training Baseline")
    print()
    print("Device:", device)

    model = build_model()

    load_global_snapshot(
        model,
        GLOBAL_BASE,
    )

    model = model.to(device)

    # Her şeyi freeze et.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Yalnız L0-E6 trainable.
    expert = (
        model
        .blocks[TARGET_LAYER]
        .moe
        .experts[TARGET_EXPERT]
    )

    for parameter in expert.parameters():
        parameter.requires_grad = True

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in trainable_parameters
    )

    print()
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    print(
        f"Trainable ratio: "
        f"{100 * trainable_count / total_parameters:.2f}%"
    )

    print(
        f"Training tokens: "
        f"{STEPS * BATCH_SIZE * SEQ_LEN:,}"
    )

    baseline = evaluate_all(
        model,
        device,
    )

    print_results(
        "GLOBAL STEP-30 BEFORE TRAINING",
        baseline,
    )

    before_expert = clone_expert_state(
        expert
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    model.train()

    running_loss = 0.0

    for step in range(
        1,
        STEPS + 1,
    ):
        # 120 D0 batch + 120 D1 batch.
        domain = 0 if step % 2 == 1 else 1

        x, y = generate_domain_batch(
            domain=domain,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            input_ids=x,
            labels=y,
        )

        loss = output.lm_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += float(
            loss.detach().cpu()
        )

        if step % 40 == 0:
            print(
                f"step {step:03d}/{STEPS} "
                f"avg_loss="
                f"{running_loss / 40:.6f}"
            )

            running_loss = 0.0

    centralized = evaluate_all(
        model,
        device,
    )

    print_results(
        "CENTRALIZED EXPERT RESULT",
        centralized,
    )

    after_expert = clone_expert_state(
        expert
    )

    delta = expert_delta(
        before_expert,
        after_expert,
    )

    norm = delta_norm(delta)

    print()
    print(
        f"Centralized expert delta L2 norm: "
        f"{norm:.6f}"
    )

    output_dir = Path(
        "checkpoints/m6b"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    delta_file = (
        output_dir
        / "centralized_expert_delta.safetensors"
    )

    save_file(
        delta,
        str(delta_file),
    )

    export_global_snapshot(
        model=model,
        output_dir=(
            output_dir
            / "centralized_global"
        ),
        global_version=31,
    )

    # --------------------------------------------------------
    # Aynı evaluation setinde Expert FedAvg'i tekrar ölç.
    # --------------------------------------------------------

    expert_fedavg_model = build_model()

    load_global_snapshot(
        expert_fedavg_model,
        EXPERT_FEDAVG,
    )

    expert_fedavg_model = (
        expert_fedavg_model.to(device)
    )

    expert_fedavg_results = evaluate_all(
        expert_fedavg_model,
        device,
    )

    # --------------------------------------------------------
    # Full-model FedAvg raporu.
    # Bu M6A'da aynı exact evaluation seti kullanıldı.
    # --------------------------------------------------------

    full_fedavg_path = Path(
        "results/m6_full_model_fedavg.json"
    )

    with open(
        full_fedavg_path,
        "r",
        encoding="utf-8",
    ) as f:
        full_report = json.load(f)

    full_fedavg_results = (
        full_report["federated"]
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    methods = {
        "Global V0":
            baseline,

        "Expert FedAvg":
            expert_fedavg_results,

        "Full FedAvg":
            full_fedavg_results,

        "Centralized Expert":
            centralized,
    }

    print()
    print("=" * 90)
    print("FINAL BASELINE COMPARISON")
    print("=" * 90)

    print(
        f"{'Method':<22}"
        f"{'D0':>10}"
        f"{'D1':>10}"
        f"{'D2':>10}"
        f"{'D3':>10}"
        f"{'Mean':>10}"
    )

    print("-" * 90)

    comparison = {}

    for name, results in methods.items():
        accuracies = [
            result["accuracy"] * 100
            for result in results
        ]

        mean_accuracy = (
            sum(accuracies) / 4
        )

        comparison[name] = {
            "accuracies": accuracies,
            "mean_accuracy":
                mean_accuracy,
        }

        print(
            f"{name:<22}"
            f"{accuracies[0]:>9.2f}%"
            f"{accuracies[1]:>9.2f}%"
            f"{accuracies[2]:>9.2f}%"
            f"{accuracies[3]:>9.2f}%"
            f"{mean_accuracy:>9.2f}%"
        )

    print()
    print("PARAMETER / COMMUNICATION REFERENCE")
    print()

    print(
        "Expert trainable params : "
        f"{trainable_count:,} "
        f"({100 * trainable_count / total_parameters:.2f}%)"
    )

    print(
        "Expert FedAvg upload    : "
        "393,472 bytes/client"
    )

    print(
        "Full FedAvg upload      : "
        "10,341,944 bytes/client"
    )

    report = {
        "method":
            "centralized_expert",

        "base_checkpoint":
            GLOBAL_BASE,

        "target_layer":
            TARGET_LAYER,

        "target_expert":
            TARGET_EXPERT,

        "training_steps":
            STEPS,

        "training_tokens":
            STEPS
            * BATCH_SIZE
            * SEQ_LEN,

        "trainable_parameters":
            trainable_count,

        "delta_l2_norm":
            norm,

        "baseline":
            baseline,

        "centralized":
            centralized,

        "comparison":
            comparison,
    }

    report_path = Path(
        "results/"
        "m6b_centralized_expert.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()
