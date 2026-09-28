import argparse
import json
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

from ml.model.transformer import MiniMoELM
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
)


GLOBAL_BASE = (
    "checkpoints/"
    "m5_candidates/"
    "global_step_0030"
)

BASE_VERSION = 30

TARGET_LAYER = 0
TARGET_EXPERT = 6

LOCAL_DOMAIN = 1

LOCAL_STEPS = 120
BATCH_SIZE = 32
SEQ_LEN = 16
LEARNING_RATE = 5e-4

DOMAIN_STEPS = [1, 2, 3, 5]

DOMAIN_NAMES = [
    "D0 (+1)",
    "D1 (+2)",
    "D2 (+3)",
    "D3 (+5)",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

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
    device,
):
    starts = torch.randint(
        0,
        16,
        (BATCH_SIZE,),
        device=device,
    )

    positions = torch.arange(
        SEQ_LEN + 1,
        device=device,
    )

    base = domain * 16
    step = DOMAIN_STEPS[domain]

    sequences = (
        starts[:, None]
        + step * positions[None, :]
    ) % 16

    sequences = sequences + base

    return (
        sequences[:, :-1].long(),
        sequences[:, 1:].long(),
    )


def build_eval_set(
    domain,
    device,
):
    starts = torch.arange(
        16,
        device=device,
    )

    positions = torch.arange(
        SEQ_LEN + 1,
        device=device,
    )

    base = domain * 16
    step = DOMAIN_STEPS[domain]

    sequences = (
        starts[:, None]
        + step * positions[None, :]
    ) % 16

    sequences = sequences + base

    return (
        sequences[:, :-1].long(),
        sequences[:, 1:].long(),
    )


@torch.no_grad()
def evaluate(
    model,
    domain,
    device,
):
    model.eval()

    x, y = build_eval_set(
        domain,
        device,
    )

    output = model(
        input_ids=x,
        labels=y,
    )

    predictions = (
        output.logits.argmax(dim=-1)
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


def clone_expert(expert):
    return {
        key: value.detach().cpu().clone()
        for key, value
        in expert.state_dict().items()
    }


def calculate_delta(
    before,
    after,
):
    return {
        key: (
            after[key]
            - before[key]
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


def post_bytes(
    url,
    data,
    content_type,
):
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type":
                content_type,
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        return response.read().decode(
            "utf-8"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--server",
        required=True,
    )

    args = parser.parse_args()

    server = args.server.rstrip("/")

    device = get_device()

    print()
    print("FedEdgeMoE - M7 Physical Worker")
    print("Client: MSI / client_B")
    print("Local domain: D1")
    print("Target: L0-E6")
    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    model = build_model()

    load_global_snapshot(
        model,
        GLOBAL_BASE,
    )

    model = model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False

    expert = (
        model
        .blocks[TARGET_LAYER]
        .moe
        .experts[TARGET_EXPERT]
    )

    for parameter in expert.parameters():
        parameter.requires_grad = True

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Total parameters: "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Trainable ratio: "
        f"{100 * trainable_params / total_params:.2f}%"
    )

    before_all = [
        evaluate(
            model,
            domain,
            device,
        )
        for domain in range(4)
    ]

    print()
    print("Before local training:")

    for domain, result in enumerate(
        before_all
    ):
        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss={result['loss']:.6f} "
            f"acc="
            f"{result['accuracy'] * 100:.2f}%"
        )

    before_expert = clone_expert(
        expert
    )

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    model.train()

    running_loss = 0.0

    for step in range(
        1,
        LOCAL_STEPS + 1,
    ):
        x, y = generate_domain_batch(
            LOCAL_DOMAIN,
            device,
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
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += float(
            loss.detach().cpu()
        )

        if step % 30 == 0:
            print(
                f"step {step:03d}/{LOCAL_STEPS} "
                f"avg_loss="
                f"{running_loss / 30:.6f}"
            )

            running_loss = 0.0

    after_all = [
        evaluate(
            model,
            domain,
            device,
        )
        for domain in range(4)
    ]

    print()
    print("After local training:")

    for domain, result in enumerate(
        after_all
    ):
        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss={result['loss']:.6f} "
            f"acc="
            f"{result['accuracy'] * 100:.2f}%"
        )

    after_expert = clone_expert(
        expert
    )

    delta = calculate_delta(
        before_expert,
        after_expert,
    )

    norm = delta_norm(delta)

    output_dir = Path(
        "checkpoints/m7/client_B"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    delta_path = (
        output_dir
        / "expert_delta.safetensors"
    )

    save_file(
        delta,
        str(delta_path),
    )

    tokens = (
        LOCAL_STEPS
        * BATCH_SIZE
        * SEQ_LEN
    )

    metadata = {
        "client_id":
            "client_B_physical_msi",
        "domain":
            LOCAL_DOMAIN,
        "base_global_version":
            BASE_VERSION,
        "layer":
            TARGET_LAYER,
        "expert":
            TARGET_EXPERT,
        "local_steps":
            LOCAL_STEPS,
        "batch_size":
            BATCH_SIZE,
        "local_tokens_seen":
            tokens,
        "trainable_parameters":
            trainable_params,
        "delta_l2_norm":
            norm,
        "before":
            before_all,
        "after":
            after_all,
        "device":
            str(device),
        "gpu":
            (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),
    }

    metadata_bytes = json.dumps(
        metadata,
        indent=2,
    ).encode("utf-8")

    delta_bytes = (
        delta_path.read_bytes()
    )

    print()
    print(
        "Delta L2 norm:",
        f"{norm:.6f}",
    )

    print(
        "Delta bytes:",
        len(delta_bytes),
    )

    print()
    print("Uploading metadata...")

    print(
        post_bytes(
            server
            + "/upload/client_B/metadata",
            metadata_bytes,
            "application/json",
        )
    )

    print("Uploading delta...")

    print(
        post_bytes(
            server
            + "/upload/client_B/delta",
            delta_bytes,
            "application/octet-stream",
        )
    )

    print()
    print(
        "PHYSICAL EDGE UPDATE COMPLETE"
    )


if __name__ == "__main__":
    main()
