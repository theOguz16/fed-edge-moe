from pathlib import Path

import torch
import torch.nn.functional as F

import ml.network.m8_coordinator as base
from ml.training.synthetic_v2 import (
    VOCAB_SIZE,
    build_split_tensor,
)


base.START_VERSION = 100
base.NUM_ROUNDS = 1

base.TARGET_LAYER = 0
base.TARGET_EXPERT = 7

base.INITIAL_SNAPSHOT = Path(
    "checkpoints/m10_candidates/global_step_0100"
)

base.ROOT = Path("checkpoints/m12m")
base.RESULT_PATH = Path(
    "results/m12m_physical_cost.json"
)

base.DOMAIN_NAMES = [
    "D0",
    "D1",
    "D2",
    "D3",
]


def generate_domain_batch(
    domain,
    device,
):
    pool = build_split_tensor(
        domain,
        "train",
        base.SEQ_LEN,
    )

    indices = torch.randint(
        0,
        pool.shape[0],
        (base.BATCH_SIZE,),
    )

    sequences = pool[indices].to(device)

    x = sequences[:, :-1]
    y = sequences[:, 1:].clone()

    # V2 recurrence iki önceki token'a bağlı.
    # İlk prediction position loss'a katılmıyor.
    y[:, 0] = -100

    return x, y


@torch.no_grad()
def evaluate_all(
    model,
    device,
):
    model.eval()

    results = []

    for domain in range(4):
        sequences = build_split_tensor(
            domain,
            "validation",
            base.SEQ_LEN,
        ).to(device)

        x = sequences[:, :-1]
        y = sequences[:, 1:]

        output = model(
            input_ids=x,
        )

        logits = output.logits[:, 1:, :]
        labels = y[:, 1:]

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE,
            ),
            labels.reshape(-1),
        )

        prediction = logits.argmax(
            dim=-1
        )

        accuracy = (
            prediction == labels
        ).float().mean()

        results.append({
            "loss": float(
                loss.detach().cpu()
            ),
            "accuracy": float(
                accuracy.detach().cpu()
            ),
        })

    return results


base.generate_domain_batch = (
    generate_domain_batch
)

base.evaluate_all = evaluate_all


if __name__ == "__main__":
    base.main()
