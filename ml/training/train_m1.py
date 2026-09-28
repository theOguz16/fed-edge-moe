# Specialization, İleride ideal olarak bazı expert'lerin belirli veri türlerine daha fazla cevap vermesini istiyoruz.
# Ablation, modelin bir parçasını kontrollü olarak çıkarıp: bu parça gerçekten ne işe yarıyordu? diye bakmak demek.
import random
from pathlib import Path

import numpy as np
import torch

from ml.model.transformer import MiniMoELM
from ml.training.synthetic import generate_batch


SEED = 42

STEPS = 400
BATCH_SIZE = 32
SEQ_LEN = 16

LEARNING_RATE = 3e-4

LOG_EVERY = 25


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, device):
    model.eval()

    input_ids, labels, _ = generate_batch(
        batch_size=128,
        seq_len=SEQ_LEN,
        device=device,
    )

    output = model(
        input_ids=input_ids,
        labels=labels,
    )

    predictions = output.logits.argmax(dim=-1)

    accuracy = (
        predictions == labels
    ).float().mean()

    model.train()

    return (
        float(output.lm_loss.detach().cpu()),
        float(accuracy.detach().cpu()),
    )


def main():
    set_seed(SEED)

    device = get_device()

    print()
    print("FedEdgeMoE - M1 Training")
    print("Device:", device)

    model = MiniMoELM(
        vocab_size=64,
        max_seq_len=32,

        d_model=128,
        num_heads=4,
        num_layers=3,

        num_experts=8,
        top_k=2,

        expert_hidden_dim=256,

        router_aux_loss_weight=0.01,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    initial_loss, initial_accuracy = evaluate(
        model,
        device,
    )

    print()
    print(
        f"Initial eval loss: {initial_loss:.4f}"
    )

    print(
        f"Initial accuracy: {initial_accuracy * 100:.2f}%"
    )

    model.train()

    running_loss = 0.0

    running_counts = torch.zeros(
        3,
        8,
        device=device,
    )

    for step in range(1, STEPS + 1):

        input_ids, labels, _ = generate_batch(
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            input_ids=input_ids,
            labels=labels,
        )

        output.loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += float(
            output.loss.detach().cpu()
        )

        running_counts += (
            output.expert_counts.detach()
        )

        if step % LOG_EVERY == 0:

            avg_loss = (
                running_loss / LOG_EVERY
            )

            counts = (
                running_counts
                .detach()
                .cpu()
            )

            utilization = (
                counts
                / counts.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1)
            )

            print()
            print(
                f"Step {step:03d}/{STEPS}"
            )

            print(
                f"Train loss: {avg_loss:.4f}"
            )

            print(
                "Router aux:",
                f"{float(output.router_aux_loss.detach().cpu()):.4f}"
            )

            for layer in range(3):
                values = [
                    f"{v * 100:5.1f}%"
                    for v in utilization[layer].tolist()
                ]

                print(
                    f"Layer {layer}:",
                    " ".join(values),
                )

            running_loss = 0.0
            running_counts.zero_()

    final_loss, final_accuracy = evaluate(
        model,
        device,
    )

    print()
    print("=" * 60)

    print("TRAINING COMPLETE")

    print(
        f"Initial loss : {initial_loss:.4f}"
    )

    print(
        f"Final loss   : {final_loss:.4f}"
    )

    print(
        f"Initial acc  : {initial_accuracy * 100:.2f}%"
    )

    print(
        f"Final acc    : {final_accuracy * 100:.2f}%"
    )

    checkpoint_dir = Path(
        "checkpoints/m1"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = checkpoint_dir / "minimoe_m1.pt"

    torch.save(
        model.state_dict(),
        path,
    )

    print()
    print(
        "Checkpoint:",
        path,
    )


if __name__ == "__main__":
    main()
