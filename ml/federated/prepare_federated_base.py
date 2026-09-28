import random
from pathlib import Path

import torch

from ml.model.transformer import MiniMoELM
from ml.training.synthetic import generate_batch
from ml.federated.checkpoint_manager import export_global_snapshot


SEED = 1234

TRAIN_STEPS = 75

SNAPSHOT_STEPS = {
    0,
    5,
    10,
    20,
    30,
    50,
    75,
}

BATCH_SIZE = 32
SEQ_LEN = 16
LEARNING_RATE = 3e-4

DOMAIN_STEPS = [1, 2, 3, 5]

DOMAIN_NAMES = [
    "D0 (+1)",
    "D1 (+2)",
    "D2 (+3)",
    "D3 (+5)",
]


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


def build_exact_domain_set(domain, device):
    """
    Her domain için mümkün olan 16 başlangıç değerinin
    tamamını kullanır.

    Böylece evaluation rastgele değildir.
    """

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

    x, y = build_exact_domain_set(
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

    result = {
        "loss": float(
            output.lm_loss.detach().cpu()
        ),

        "accuracy": float(
            accuracy.detach().cpu()
        ),
    }

    model.train()

    return result


def evaluate_all(model, device):

    return [
        evaluate_domain(
            model,
            domain,
            device,
        )
        for domain in range(4)
    ]


def print_evaluation(step, results):

    print()
    print("=" * 72)

    print(
        f"GLOBAL CHECKPOINT @ STEP {step}"
    )

    print("=" * 72)

    mean_accuracy = 0.0
    mean_loss = 0.0

    for domain, result in enumerate(
        results
    ):
        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss={result['loss']:.4f} "
            f"acc="
            f"{result['accuracy'] * 100:6.2f}%"
        )

        mean_accuracy += (
            result["accuracy"]
        )

        mean_loss += (
            result["loss"]
        )

    mean_accuracy /= 4
    mean_loss /= 4

    print()

    print(
        f"MEAN     "
        f"loss={mean_loss:.4f} "
        f"acc={mean_accuracy * 100:6.2f}%"
    )


def save_snapshot(
    model,
    step,
    results,
):
    path = Path(
        f"checkpoints/m5_candidates/"
        f"global_step_{step:04d}"
    )

    export_global_snapshot(
        model=model,
        output_dir=path,

        # Şimdilik step'i versiyon olarak
        # kullanıyoruz.
        global_version=step,
    )

    return path


def main():

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()

    print()
    print(
        "FedEdgeMoE - M5A"
    )

    print(
        "Pre-Federation Base Selection"
    )

    print()

    print(
        "Device:",
        device,
    )

    model = build_model().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=0.01,
    )

    model.train()

    # --------------------------------------------------------
    # STEP 0
    # --------------------------------------------------------

    results = evaluate_all(
        model,
        device,
    )

    print_evaluation(
        0,
        results,
    )

    path = save_snapshot(
        model,
        0,
        results,
    )

    print(
        "Saved:",
        path,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):

        x, y, _ = generate_batch(
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

        output.loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if step in SNAPSHOT_STEPS:

            results = evaluate_all(
                model,
                device,
            )

            print_evaluation(
                step,
                results,
            )

            path = save_snapshot(
                model,
                step,
                results,
            )

            print(
                "Saved:",
                path,
            )

    print()
    print("=" * 72)

    print(
        "CANDIDATE GENERATION COMPLETE"
    )

    print("=" * 72)

    print()
    print(
        "We want a checkpoint that is learned,"
    )

    print(
        "but NOT saturated at 100%."
    )


if __name__ == "__main__":
    main()
