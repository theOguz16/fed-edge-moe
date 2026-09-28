import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ml.model.transformer import MiniMoELM
from ml.federated.checkpoint_manager import load_global_snapshot


TARGET_LAYER = 0
TARGET_EXPERT = 6

LOCAL_STEPS = 120
BATCH_SIZE = 32
SEQ_LEN = 16
LEARNING_RATE = 5e-4

GLOBAL_SNAPSHOT = "checkpoints/m2/global_v0000"


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
    steps = [1, 2, 3, 5]

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
    step_size = steps[domain]

    sequences = (
        starts[:, None]
        + step_size * positions[None, :]
    ) % 16

    sequences = sequences + base

    input_ids = sequences[:, :-1].long()
    labels = sequences[:, 1:].long()

    return input_ids, labels


@torch.no_grad()
def evaluate_domain(
    model,
    domain,
    device,
    batches=10,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for _ in range(batches):

        x, y = generate_domain_batch(
            domain=domain,
            batch_size=128,
            seq_len=SEQ_LEN,
            device=device,
        )

        output = model(
            input_ids=x,
            labels=y,
        )

        predictions = output.logits.argmax(
            dim=-1
        )

        total_loss += float(
            output.lm_loss.detach().cpu()
        )

        total_correct += int(
            (predictions == y)
            .sum()
            .detach()
            .cpu()
        )

        total_tokens += y.numel()

    return {
        "loss": total_loss / batches,
        "accuracy": (
            total_correct / total_tokens
        ),
    }


def clone_expert_state(expert):
    return {
        key: value.detach().cpu().clone()
        for key, value
        in expert.state_dict().items()
    }


def calculate_delta(
    before_state,
    after_state,
):
    delta = {}

    total_squared_norm = 0.0

    for key in before_state:

        difference = (
            after_state[key].detach().cpu()
            - before_state[key]
        )

        delta[key] = (
            difference.contiguous()
        )

        total_squared_norm += float(
            torch.sum(
                difference.float() ** 2
            )
        )

    delta_norm = (
        total_squared_norm ** 0.5
    )

    return delta, delta_norm


def save_state(
    state,
    path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        {
            key: value.contiguous()
            for key, value
            in state.items()
        },
        str(path),
    )


def train_client(
    client_id,
    domain,
    output_dir,
    device,
):
    print()
    print("=" * 70)

    print(
        f"CLIENT {client_id}"
    )

    print(
        f"Local domain: D{domain}"
    )

    print(
        f"Trainable expert: "
        f"L{TARGET_LAYER}-E{TARGET_EXPERT}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Her client global modelin AYNI başlangıç sürümünü alır.
    # --------------------------------------------------------

    model = build_model()

    manifest = load_global_snapshot(
        model=model,
        snapshot_dir=GLOBAL_SNAPSHOT,
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Tüm global model freeze.
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # Sadece atanmış expert trainable.
    # --------------------------------------------------------

    expert = (
        model
        .blocks[TARGET_LAYER]
        .moe
        .experts[TARGET_EXPERT]
    )

    for parameter in expert.parameters():
        parameter.requires_grad = True

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    total_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        f"Total parameters: "
        f"{total_count:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    print(
        f"Trainable ratio: "
        f"{100 * trainable_count / total_count:.2f}%"
    )

    # --------------------------------------------------------
    # Başlangıç expert state
    # --------------------------------------------------------

    base_expert_state = clone_expert_state(
        expert
    )

    before_local = evaluate_domain(
        model,
        domain,
        device,
    )

    print()
    print(
        f"Before local training:"
    )

    print(
        f"loss={before_local['loss']:.6f} "
        f"acc={before_local['accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # LOCAL TRAINING
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
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

        # Burada router frozen olduğu için
        # sadece ana LM loss ile expert öğreniyor.
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

        if step % 30 == 0:

            print(
                f"step {step:03d}/{LOCAL_STEPS} "
                f"avg_loss="
                f"{running_loss / 30:.6f}"
            )

            running_loss = 0.0

    # --------------------------------------------------------
    # Local evaluation
    # --------------------------------------------------------

    after_local = evaluate_domain(
        model,
        domain,
        device,
    )

    print()
    print(
        f"After local training:"
    )

    print(
        f"loss={after_local['loss']:.6f} "
        f"acc={after_local['accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Expert delta
    # --------------------------------------------------------

    after_expert_state = clone_expert_state(
        expert
    )

    delta, delta_norm = calculate_delta(
        before_state=base_expert_state,
        after_state=after_expert_state,
    )

    print()
    print(
        f"Expert delta L2 norm: "
        f"{delta_norm:.6f}"
    )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Tam local expert
    save_state(
        after_expert_state,
        output_dir
        / "expert_full.safetensors",
    )

    # Sadece değişim
    save_state(
        delta,
        output_dir
        / "expert_delta.safetensors",
    )

    metadata = {
        "client_id": client_id,

        "domain": domain,

        "base_global_version":
            manifest["global_version"],

        "layer": TARGET_LAYER,

        "expert": TARGET_EXPERT,

        "local_steps": LOCAL_STEPS,

        "batch_size": BATCH_SIZE,

        "local_sequences_seen":
            LOCAL_STEPS * BATCH_SIZE,

        "local_tokens_seen":
            LOCAL_STEPS
            * BATCH_SIZE
            * SEQ_LEN,

        "trainable_parameters":
            trainable_count,

        "delta_l2_norm":
            delta_norm,

        "before_loss":
            before_local["loss"],

        "after_loss":
            after_local["loss"],

        "before_accuracy":
            before_local["accuracy"],

        "after_accuracy":
            after_local["accuracy"],
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        "Saved:"
    )

    print(
        output_dir
        / "expert_full.safetensors"
    )

    print(
        output_dir
        / "expert_delta.safetensors"
    )

    return metadata


def main():

    device = get_device()

    print()
    print("FedEdgeMoE - M3")
    print("Simulated Local Edge Learning")
    print()

    print(
        "Device:",
        device,
    )

    client_a = train_client(
        client_id="client_A",
        domain=0,
        output_dir=(
            "checkpoints/m3/client_A"
        ),
        device=device,
    )

    client_b = train_client(
        client_id="client_B",
        domain=1,
        output_dir=(
            "checkpoints/m3/client_B"
        ),
        device=device,
    )

    print()
    print("=" * 70)
    print("M3 SUMMARY")
    print("=" * 70)

    for client in [
        client_a,
        client_b,
    ]:

        print()

        print(
            client["client_id"]
        )

        print(
            f"domain=D{client['domain']}"
        )

        print(
            f"base_version="
            f"{client['base_global_version']}"
        )

        print(
            f"delta_norm="
            f"{client['delta_l2_norm']:.6f}"
        )

        print(
            f"loss: "
            f"{client['before_loss']:.6f}"
            f" -> "
            f"{client['after_loss']:.6f}"
        )

    print()
    print(
        "Both clients trained the SAME "
        "global expert independently."
    )

    print()
    print(
        "Next milestone: "
        "expert-wise federated aggregation."
    )


if __name__ == "__main__":
    main()
