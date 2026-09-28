import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v3 import (
    VOCAB_SIZE,
    NUM_DOMAINS,
    build_split_tensor,
    generate_balanced_batch,
    print_split_summary,
)
from ml.federated.checkpoint_manager import (
    export_global_snapshot,
)


SEED = 20260928

TRAIN_STEPS = 120

SNAPSHOT_STEPS = {
    0,
    25,
    50,
    75,
    100,
    120,
}

# PRE-REGISTERED.
# Scheduler sonuçlarına bakmadan önce sabit.
FEDERATED_BASE_STEP = 100

BATCH_SIZE = 32
SEQ_LEN = 16
LEARNING_RATE = 3e-4

OUTPUT_ROOT = Path(
    "checkpoints/m11e_replication"
)

REPORT_PATH = Path(
    "results/m11e1_replication_base.json"
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
    logits = (
        output.logits[:, 1:, :]
    )

    labels = (
        targets[:, 1:]
    )

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

    domain_results = []

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

        labels = (
            y[:, 1:]
        )

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

        accuracy = (
            predictions == labels
        ).float().mean()

        domain_results.append({
            "domain":
                domain,

            "loss":
                float(
                    loss.detach().cpu()
                ),

            "accuracy":
                float(
                    accuracy.detach().cpu()
                ),
        })

    return domain_results


def mean_accuracy(results):
    return (
        sum(
            item["accuracy"]
            for item in results
        )
        / len(results)
    )


def mean_loss(results):
    return (
        sum(
            item["loss"]
            for item in results
        )
        / len(results)
    )


def print_results(
    label,
    results,
):
    print(label)

    for item in results:
        print(
            f"D{item['domain']} "
            f"loss="
            f"{item['loss']:.4f} "
            f"acc="
            f"{item['accuracy'] * 100:6.2f}%"
        )

    print(
        f"MEAN "
        f"loss={mean_loss(results):.4f} "
        f"acc="
        f"{mean_accuracy(results) * 100:6.2f}%"
    )


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
    print("FedEdgeMoE - M11E1")
    print(
        "Independent Scheduler "
        "Replication Base"
    )

    print()
    print("Device:", device)
    print("Seed:", SEED)

    print()
    print_split_summary()

    print()
    print(
        "PRE-REGISTERED FEDERATED "
        "BASE STEP:",
        FEDERATED_BASE_STEP,
    )

    print()
    print(
        "TEST SPLIT REMAINS SEALED."
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

    def snapshot(step):
        train_results = (
            evaluate_split(
                model,
                "train",
                device,
            )
        )

        validation_results = (
            evaluate_split(
                model,
                "validation",
                device,
            )
        )

        checkpoint = (
            OUTPUT_ROOT
            / (
                f"global_step_"
                f"{step:04d}"
            )
        )

        export_global_snapshot(
            model=model,
            output_dir=checkpoint,
            global_version=step,
        )

        print()
        print("=" * 78)

        print(
            f"CHECKPOINT STEP {step}"
        )

        print("=" * 78)

        print_results(
            "TRAIN",
            train_results,
        )

        print()

        print_results(
            "VALIDATION",
            validation_results,
        )

        if (
            step
            == FEDERATED_BASE_STEP
        ):
            print()
            print(
                "*** PRE-REGISTERED "
                "FEDERATED BASE ***"
            )

        history.append({
            "step":
                step,

            "train":
                train_results,

            "validation":
                validation_results,

            "checkpoint":
                str(checkpoint),

            "federated_base":
                (
                    step
                    == FEDERATED_BASE_STEP
                ),
        })

    snapshot(0)

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
            snapshot(step)
            model.train()

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "experiment":
                    "M11E independent replication",

                "seed":
                    SEED,

                "synthetic_version":
                    "V3",

                "federated_base_step":
                    FEDERATED_BASE_STEP,

                "federated_base_checkpoint":
                    str(
                        OUTPUT_ROOT
                        / "global_step_0100"
                    ),

                "balanced_scheduler_formula":
                    (
                        "min(A,B) * "
                        "(min(A,B)/max(A,B))"
                    ),

                "scheduler_formula_fixed_before_results":
                    True,

                "test_evaluated":
                    False,

                "history":
                    history,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("M11E1 COMPLETE")
    print("=" * 78)

    print(
        "Federated base:",
        OUTPUT_ROOT
        / "global_step_0100",
    )

    print(
        "TEST SPLIT REMAINS SEALED."
    )

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
