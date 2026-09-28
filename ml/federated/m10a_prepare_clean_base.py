import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v2 import (
    VOCAB_SIZE,
    NUM_DOMAINS,
    build_split_tensor,
    generate_balanced_batch,
    print_split_summary,
)
from ml.federated.checkpoint_manager import (
    export_global_snapshot,
)


SEED = 20260926

TRAIN_STEPS = 300

SNAPSHOT_STEPS = {
    0,
    25,
    50,
    75,
    100,
    150,
    200,
    250,
    300,
}

BATCH_SIZE = 32
SEQ_LEN = 16

LEARNING_RATE = 3e-4

OUTPUT_ROOT = Path(
    "checkpoints/m10_candidates"
)

REPORT_PATH = Path(
    "results/m10a_base_candidates.json"
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model():
    return MiniMoELM(
        vocab_size=VOCAB_SIZE,
        max_seq_len=32,

        d_model=128,
        num_heads=4,
        num_layers=3,

        num_experts=8,
        top_k=2,

        expert_hidden_dim=256,

        router_aux_loss_weight=0.01,
    )


def prediction_loss(
    output,
    targets,
):
    # İlk target'ı bilinçli olarak
    # değerlendirmiyoruz.
    #
    # Çünkü recurrence iki başlangıç
    # token'ına ihtiyaç duyuyor.
    #
    # Position 1'den itibaren model
    # iki-token context'e sahip.

    logits = output.logits[:, 1:, :]

    labels = targets[:, 1:]

    lm_loss = F.cross_entropy(
        logits.reshape(
            -1,
            logits.shape[-1],
        ),
        labels.reshape(-1),
    )

    return (
        lm_loss
        + 0.01
        * output.router_aux_loss
    )


@torch.no_grad()
def evaluate_split(
    model,
    split,
    device,
):
    model.eval()

    all_losses = []
    total_correct = 0
    total_tokens = 0

    for domain in range(
        NUM_DOMAINS
    ):
        sequences = (
            build_split_tensor(
                domain,
                split,
                SEQ_LEN,
            )
            .to(device)
        )

        x = sequences[:, :-1]
        y = sequences[:, 1:]

        output = model(
            input_ids=x,
        )

        logits = (
            output.logits[:, 1:, :]
        )

        labels = y[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.shape[-1],
            ),
            labels.reshape(-1),
        )

        predictions = (
            logits.argmax(dim=-1)
        )

        correct = (
            predictions == labels
        ).sum()

        count = labels.numel()

        all_losses.append(
            float(
                loss.detach().cpu()
            )
        )

        total_correct += int(
            correct.detach().cpu()
        )

        total_tokens += count

    return {
        "mean_loss":
            sum(all_losses)
            / len(all_losses),

        "accuracy":
            total_correct
            / total_tokens,
    }


def save_checkpoint(
    model,
    step,
):
    path = (
        OUTPUT_ROOT
        / f"global_step_{step:04d}"
    )

    export_global_snapshot(
        model=model,
        output_dir=path,
        global_version=step,
    )

    return path


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print()
    print("FedEdgeMoE - M10A")
    print(
        "Clean Train / Validation / "
        "Sealed-Test Dataset"
    )

    print()
    print("Device:", device)
    print()

    print_split_summary()

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "TEST SPLIT WILL NOT BE "
        "EVALUATED IN M10A."
    )

    model = (
        build_model()
        .to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    history = []

    def evaluate_and_save(step):
        train_result = (
            evaluate_split(
                model,
                "train",
                device,
            )
        )

        validation_result = (
            evaluate_split(
                model,
                "validation",
                device,
            )
        )

        checkpoint = (
            save_checkpoint(
                model,
                step,
            )
        )

        record = {
            "step":
                step,

            "train":
                train_result,

            "validation":
                validation_result,

            "checkpoint":
                str(checkpoint),
        }

        history.append(
            record
        )

        print()
        print("=" * 76)

        print(
            f"CHECKPOINT STEP {step}"
        )

        print("=" * 76)

        print(
            "TRAIN      "
            f"loss="
            f"{train_result['mean_loss']:.4f} "
            f"acc="
            f"{train_result['accuracy'] * 100:6.2f}%"
        )

        print(
            "VALIDATION "
            f"loss="
            f"{validation_result['mean_loss']:.4f} "
            f"acc="
            f"{validation_result['accuracy'] * 100:6.2f}%"
        )

        print(
            "Saved:",
            checkpoint,
        )

    evaluate_and_save(0)

    model.train()

    for step in range(
        1,
        TRAIN_STEPS + 1,
    ):
        x, y = (
            generate_balanced_batch(
                split="train",
                batch_size=BATCH_SIZE,
                seq_len=SEQ_LEN,
                device=device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            input_ids=x,
        )

        loss = prediction_loss(
            output,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if step in SNAPSHOT_STEPS:
            evaluate_and_save(
                step
            )

            model.train()

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "seed":
                    SEED,

                "train_steps":
                    TRAIN_STEPS,

                "batch_size":
                    BATCH_SIZE,

                "seq_len":
                    SEQ_LEN,

                "history":
                    history,

                "test_evaluated":
                    False,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 76)

    print(
        "M10A COMPLETE"
    )

    print("=" * 76)

    print(
        "TEST SPLIT REMAINS SEALED."
    )

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
