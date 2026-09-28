import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v2 import (
    VOCAB_SIZE,
    NUM_DOMAINS,
    build_split_tensor,
)
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
)


BASE_CHECKPOINT = (
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

FINAL_CHECKPOINT = (
    "checkpoints/m10c/"
    "global_v0105"
)

SEQ_LEN = 16

REPORT_PATH = Path(
    "results/m10d_sealed_test.json"
)

DOMAIN_NAMES = [
    "D0",
    "D1",
    "D2",
    "D3",
]


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


@torch.no_grad()
def evaluate_test(
    model,
    device,
):
    model.eval()

    results = []

    for domain in range(
        NUM_DOMAINS
    ):
        sequences = (
            build_split_tensor(
                domain,
                "test",
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

        results.append({
            "loss": float(
                loss.detach().cpu()
            ),

            "accuracy": float(
                accuracy.detach().cpu()
            ),

            "num_sequences":
                int(
                    sequences.shape[0]
                ),

            "num_tokens":
                int(
                    labels.numel()
                ),
        })

    return results


def print_results(
    title,
    results,
):
    mean_loss = (
        sum(
            result["loss"]
            for result in results
        )
        / len(results)
    )

    mean_accuracy = (
        sum(
            result["accuracy"]
            for result in results
        )
        / len(results)
    )

    print()
    print(title)

    for domain, result in enumerate(
        results
    ):
        print(
            f"{DOMAIN_NAMES[domain]} "
            f"loss={result['loss']:.6f} "
            f"acc={result['accuracy'] * 100:6.2f}% "
            f"seq={result['num_sequences']}"
        )

    print(
        "MEAN "
        f"loss={mean_loss:.6f} "
        f"acc={mean_accuracy * 100:6.2f}%"
    )

    return (
        mean_loss,
        mean_accuracy,
    )


def main():
    device = get_device()

    print()
    print("FedEdgeMoE - M10D")
    print("SEALED HELD-OUT TEST")
    print()
    print("Device:", device)

    print()
    print(
        "TEST SPLIT IS NOW BEING "
        "EVALUATED FOR THE FIRST TIME."
    )

    base_model = build_model()

    load_global_snapshot(
        base_model,
        BASE_CHECKPOINT,
    )

    base_model = (
        base_model.to(device)
    )

    final_model = build_model()

    load_global_snapshot(
        final_model,
        FINAL_CHECKPOINT,
    )

    final_model = (
        final_model.to(device)
    )

    base_results = evaluate_test(
        base_model,
        device,
    )

    final_results = evaluate_test(
        final_model,
        device,
    )

    (
        base_mean_loss,
        base_mean_accuracy,
    ) = print_results(
        "BASE V100 - SEALED TEST",
        base_results,
    )

    (
        final_mean_loss,
        final_mean_accuracy,
    ) = print_results(
        "FINAL V105 - SEALED TEST",
        final_results,
    )

    print()
    print("=" * 80)
    print("SEALED TEST CHANGE")
    print("=" * 80)

    per_domain = []

    for domain in range(
        NUM_DOMAINS
    ):
        accuracy_delta = (
            final_results[domain][
                "accuracy"
            ]
            - base_results[domain][
                "accuracy"
            ]
        )

        loss_delta = (
            final_results[domain][
                "loss"
            ]
            - base_results[domain][
                "loss"
            ]
        )

        per_domain.append({
            "domain":
                domain,

            "accuracy_delta":
                accuracy_delta,

            "loss_delta":
                loss_delta,
        })

        print(
            f"{DOMAIN_NAMES[domain]} "
            f"loss_delta="
            f"{loss_delta:+.6f} "
            f"acc_delta="
            f"{accuracy_delta * 100:+.2f}pp"
        )

    mean_accuracy_delta = (
        final_mean_accuracy
        - base_mean_accuracy
    )

    mean_loss_delta = (
        final_mean_loss
        - base_mean_loss
    )

    print()
    print(
        "MEAN "
        f"loss_delta="
        f"{mean_loss_delta:+.6f} "
        f"acc_delta="
        f"{mean_accuracy_delta * 100:+.2f}pp"
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "base_checkpoint":
                    BASE_CHECKPOINT,

                "final_checkpoint":
                    FINAL_CHECKPOINT,

                "selection_used_test":
                    False,

                "test_opened_after_selection":
                    True,

                "base":
                    base_results,

                "final":
                    final_results,

                "per_domain_change":
                    per_domain,

                "base_mean_accuracy":
                    base_mean_accuracy,

                "final_mean_accuracy":
                    final_mean_accuracy,

                "mean_accuracy_delta":
                    mean_accuracy_delta,

                "base_mean_loss":
                    base_mean_loss,

                "final_mean_loss":
                    final_mean_loss,

                "mean_loss_delta":
                    mean_loss_delta,
            },
            f,
            indent=2,
        )

    print()
    print(
        "Report:",
        REPORT_PATH,
    )

    print()
    print("=" * 80)
    print("M10 SEALED TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
