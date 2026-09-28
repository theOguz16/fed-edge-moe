import torch
import torch.nn.functional as F

import ml.network.m8_worker_loop as base
from ml.training.synthetic_v2 import (
    VOCAB_SIZE,
    build_split_tensor,
)


base.DOMAIN_NAMES = [
    "D0",
    "D1",
    "D2",
    "D3",
]


def generate_domain_batch(
    domain,
    batch_size,
    seq_len,
    device,
):
    pool = build_split_tensor(
        domain,
        "train",
        seq_len,
    )

    indices = torch.randint(
        0,
        pool.shape[0],
        (batch_size,),
    )

    sequences = pool[indices].to(device)

    x = sequences[:, :-1]
    y = sequences[:, 1:].clone()

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


# M8 worker geçmiş checkpointlerini ezmemek için
# yalnız bu path'i M12M'e yönlendir.
_original_path = base.Path


def redirected_path(*args):
    if (
        len(args) == 1
        and str(args[0])
        == "checkpoints/m8_worker"
    ):
        return _original_path(
            "checkpoints/m12m_worker"
        )

    return _original_path(*args)


base.Path = redirected_path


if __name__ == "__main__":
    base.main()
