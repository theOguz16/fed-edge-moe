import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ml.model.transformer import MiniMoELM
from ml.federated.checkpoint_manager import load_global_snapshot


GLOBAL_BASE = "checkpoints/m5_candidates/global_step_0030"

LOCAL_STEPS = 120
BATCH_SIZE = 32
SEQ_LEN = 16
LEARNING_RATE = 5e-4

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


def generate_domain_batch(domain, batch_size, seq_len, device):
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


def build_eval_set(domain):
    starts = torch.arange(16)
    positions = torch.arange(SEQ_LEN + 1)

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
def evaluate_domain(model, domain, device):
    model.eval()

    x, y = build_eval_set(domain)

    x = x.to(device)
    y = y.to(device)

    output = model(
        input_ids=x,
        labels=y,
    )

    predictions = output.logits.argmax(dim=-1)

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


def clone_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def calculate_delta(base_state, trained_state):
    return {
        key: (
            trained_state[key]
            - base_state[key]
        ).contiguous()
        for key in base_state
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


def flatten_state(state):
    return torch.cat([
        tensor.float().reshape(-1)
        for tensor in state.values()
    ])


def cosine_similarity(a, b):
    va = flatten_state(a)
    vb = flatten_state(b)

    return float(
        torch.nn.functional.cosine_similarity(
            va.unsqueeze(0),
            vb.unsqueeze(0),
        ).item()
    )


def train_client(client_id, domain, device):
    print()
    print("=" * 72)
    print(f"FULL MODEL CLIENT {client_id}")
    print(f"Local domain: {DOMAIN_NAMES[domain]}")
    print("=" * 72)

    model = build_model()

    load_global_snapshot(
        model,
        GLOBAL_BASE,
    )

    model = model.to(device)

    for parameter in model.parameters():
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
        f"Total parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: {trainable_params:,}"
    )

    print(
        f"Trainable ratio: "
        f"{100 * trainable_params / total_params:.2f}%"
    )

    base_state = clone_state(model)

    before = evaluate_all(
        model,
        device,
    )

    print()
    print(
        f"Before local training "
        f"{DOMAIN_NAMES[domain]}:"
    )

    print(
        f"loss={before[domain]['loss']:.6f} "
        f"acc={before[domain]['accuracy'] * 100:.2f}%"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

        loss = output.lm_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
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

    after = evaluate_all(
        model,
        device,
    )

    print()
    print(
        f"After local training "
        f"{DOMAIN_NAMES[domain]}:"
    )

    print(
        f"loss={after[domain]['loss']:.6f} "
        f"acc={after[domain]['accuracy'] * 100:.2f}%"
    )

    trained_state = clone_state(model)

    delta = calculate_delta(
        base_state,
        trained_state,
    )

    norm = delta_norm(delta)

    print()
    print(
        f"Full-model delta L2 norm: "
        f"{norm:.6f}"
    )

    output_dir = Path(
        f"checkpoints/m6/full_{client_id}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    delta_file = (
        output_dir
        / "full_model_delta.safetensors"
    )

    save_file(
        delta,
        str(delta_file),
    )

    print(
        "Delta file:",
        delta_file,
    )

    print(
        "Upload bytes:",
        delta_file.stat().st_size,
    )

    return {
        "client_id": client_id,
        "domain": domain,
        "delta": delta,
        "tokens":
            LOCAL_STEPS
            * BATCH_SIZE
            * SEQ_LEN,
        "before": before,
        "after": after,
        "delta_norm": norm,
        "upload_bytes":
            delta_file.stat().st_size,
    }


def aggregate(a, b):
    wa = a["tokens"]
    wb = b["tokens"]

    total = wa + wb

    result = {}

    for key in a["delta"]:
        result[key] = (
            a["delta"][key] * wa
            + b["delta"][key] * wb
        ) / total

    return result


def apply_delta(model, delta):
    state = model.state_dict()

    updated = {}

    for key in state:
        updated[key] = (
            state[key]
            + delta[key].to(
                dtype=state[key].dtype
            )
        )

    model.load_state_dict(updated)


def print_results(title, results):
    print()
    print(title)

    total_acc = 0.0
    total_loss = 0.0

    for domain, result in enumerate(results):
        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss={result['loss']:.6f} "
            f"acc={result['accuracy'] * 100:6.2f}%"
        )

        total_acc += result["accuracy"]
        total_loss += result["loss"]

    print(
        f"MEAN     "
        f"loss={total_loss / 4:.6f} "
        f"acc={total_acc / 4 * 100:6.2f}%"
    )


def main():
    device = get_device()

    print()
    print("FedEdgeMoE - M6A")
    print("Full-Model FedAvg Baseline")
    print()
    print("Device:", device)

    client_a = train_client(
        client_id="client_A",
        domain=0,
        device=device,
    )

    client_b = train_client(
        client_id="client_B",
        domain=1,
        device=device,
    )

    similarity = cosine_similarity(
        client_a["delta"],
        client_b["delta"],
    )

    print()
    print("=" * 72)
    print("FULL-MODEL DELTA SIMILARITY")
    print("=" * 72)

    print(
        f"Cosine similarity: "
        f"{similarity:+.6f}"
    )

    global_base = build_model()

    load_global_snapshot(
        global_base,
        GLOBAL_BASE,
    )

    global_base = global_base.to(device)

    baseline = evaluate_all(
        global_base,
        device,
    )

    aggregated = aggregate(
        client_a,
        client_b,
    )

    global_fedavg = build_model()

    load_global_snapshot(
        global_fedavg,
        GLOBAL_BASE,
    )

    apply_delta(
        global_fedavg,
        aggregated,
    )

    global_fedavg = global_fedavg.to(
        device
    )

    federated = evaluate_all(
        global_fedavg,
        device,
    )

    print_results(
        "GLOBAL V0",
        baseline,
    )

    print_results(
        "FULL-MODEL FEDAVG",
        federated,
    )

    print()
    print("=" * 72)
    print("GLOBAL V0 -> FULL FEDAVG")
    print("=" * 72)

    for domain in range(4):
        acc_delta = (
            federated[domain]["accuracy"]
            - baseline[domain]["accuracy"]
        )

        loss_delta = (
            federated[domain]["loss"]
            - baseline[domain]["loss"]
        )

        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss_delta={loss_delta:+.6f} "
            f"acc_delta="
            f"{acc_delta * 100:+.2f}pp"
        )

    total_upload = (
        client_a["upload_bytes"]
        + client_b["upload_bytes"]
    )

    print()
    print("=" * 72)
    print("COMMUNICATION")
    print("=" * 72)

    print(
        f"Client A upload: "
        f"{client_a['upload_bytes']:,} bytes"
    )

    print(
        f"Client B upload: "
        f"{client_b['upload_bytes']:,} bytes"
    )

    print(
        f"Total upload: "
        f"{total_upload:,} bytes"
    )

    report = {
        "method": "full_model_fedavg",
        "base_checkpoint": GLOBAL_BASE,
        "delta_cosine_similarity":
            similarity,
        "client_A_upload_bytes":
            client_a["upload_bytes"],
        "client_B_upload_bytes":
            client_b["upload_bytes"],
        "baseline": baseline,
        "federated": federated,
    }

    report_path = Path(
        "results/m6_full_model_fedavg.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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
