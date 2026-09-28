import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from ml.model.transformer import MiniMoELM

from ml.federated.checkpoint_manager import (
    export_global_snapshot,
    load_global_snapshot,
)


GLOBAL_V0 = "checkpoints/m5_candidates/global_step_0030"
GLOBAL_V1 = "checkpoints/m5c/global_v0031"

CLIENT_A_DIR = Path("checkpoints/m5b/client_A")
CLIENT_B_DIR = Path("checkpoints/m5b/client_B")

TARGET_LAYER = 0
TARGET_EXPERT = 6

NUM_DOMAINS = 4
SEQ_LEN = 16
EVAL_SAMPLES = 2048
BATCH_SIZE = 256

DOMAIN_NAMES = [
    "D0 (+1)",
    "D1 (+2)",
    "D2 (+3)",
    "D3 (+5)",
]

DOMAIN_STEPS = [
    1,
    2,
    3,
    5,
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


def read_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def build_domain_dataset(domain):
    generator = torch.Generator()

    generator.manual_seed(
        5000 + domain
    )

    starts = torch.randint(
        0,
        16,
        (EVAL_SAMPLES,),
        generator=generator,
    )

    positions = torch.arange(
        SEQ_LEN + 1
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
def evaluate_domain(
    model,
    input_ids,
    labels,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for start in range(
        0,
        input_ids.shape[0],
        BATCH_SIZE,
    ):

        x = input_ids[
            start:start + BATCH_SIZE
        ].to(device)

        y = labels[
            start:start + BATCH_SIZE
        ].to(device)

        output = model(
            input_ids=x,
            labels=y,
        )

        predictions = (
            output.logits.argmax(
                dim=-1
            )
        )

        total_loss += float(
            output.lm_loss.cpu()
        )

        total_correct += int(
            (predictions == y)
            .sum()
            .cpu()
        )

        total_tokens += y.numel()

        num_batches += 1

    return {
        "loss":
            total_loss / num_batches,

        "accuracy":
            total_correct / total_tokens,
    }


def evaluate_all(
    model,
    datasets,
    device,
):
    return [
        evaluate_domain(
            model,
            x,
            y,
            device,
        )
        for x, y in datasets
    ]


def flatten_state(state):
    return torch.cat([
        tensor
        .detach()
        .float()
        .reshape(-1)
        .cpu()

        for tensor in state.values()
    ])


def cosine_similarity(
    state_a,
    state_b,
):
    a = flatten_state(state_a)
    b = flatten_state(state_b)

    return float(
        torch.nn.functional.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
        ).item()
    )


def aggregate_deltas(
    delta_a,
    delta_b,
    weight_a,
    weight_b,
):
    total_weight = (
        weight_a + weight_b
    )

    result = {}

    for key in delta_a:

        result[key] = (
            delta_a[key] * weight_a
            + delta_b[key] * weight_b
        ) / total_weight

    return result


def apply_delta(
    model,
    delta,
):
    expert = (
        model
        .blocks[TARGET_LAYER]
        .moe
        .experts[TARGET_EXPERT]
    )

    state = expert.state_dict()

    updated_state = {}

    for key in state:

        updated_state[key] = (
            state[key]
            + delta[key].to(
                state[key].device,
                dtype=state[key].dtype,
            )
        )

    expert.load_state_dict(
        updated_state
    )


def print_results(
    title,
    results,
):
    print()
    print(title)

    for domain, result in enumerate(
        results
    ):
        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss={result['loss']:.6f} "
            f"acc="
            f"{result['accuracy'] * 100:6.2f}%"
        )


def main():

    device = get_device()

    print()
    print("FedEdgeMoE - M5C")
    print("Expert-wise Federated Aggregation")
    print()
    print("Device:", device)

    metadata_a = read_json(
        CLIENT_A_DIR / "metadata.json"
    )

    metadata_b = read_json(
        CLIENT_B_DIR / "metadata.json"
    )

    assert (
        metadata_a["base_global_version"]
        ==
        metadata_b["base_global_version"]
        ==
        30
    )

    assert (
        metadata_a["layer"]
        ==
        metadata_b["layer"]
        ==
        TARGET_LAYER
    )

    assert (
        metadata_a["expert"]
        ==
        metadata_b["expert"]
        ==
        TARGET_EXPERT
    )

    delta_a = load_file(
        str(
            CLIENT_A_DIR
            / "expert_delta.safetensors"
        )
    )

    delta_b = load_file(
        str(
            CLIENT_B_DIR
            / "expert_delta.safetensors"
        )
    )

    weight_a = metadata_a[
        "local_tokens_seen"
    ]

    weight_b = metadata_b[
        "local_tokens_seen"
    ]

    print()
    print("Client weights:")

    print(
        f"A: {weight_a:,} tokens"
    )

    print(
        f"B: {weight_b:,} tokens"
    )

    similarity = cosine_similarity(
        delta_a,
        delta_b,
    )

    print()
    print(
        "Delta cosine similarity:"
    )

    print(
        f"{similarity:+.6f}"
    )

    aggregated_delta = (
        aggregate_deltas(
            delta_a=delta_a,
            delta_b=delta_b,

            weight_a=weight_a,
            weight_b=weight_b,
        )
    )

    # --------------------------------------------------------
    # Deterministic evaluation datasets
    # --------------------------------------------------------

    datasets = [
        build_domain_dataset(domain)
        for domain in range(
            NUM_DOMAINS
        )
    ]

    # --------------------------------------------------------
    # Global V0
    # --------------------------------------------------------

    global_v0 = build_model()

    load_global_snapshot(
        global_v0,
        GLOBAL_V0,
    )

    global_v0 = global_v0.to(
        device
    )

    baseline = evaluate_all(
        global_v0,
        datasets,
        device,
    )

    print_results(
        "GLOBAL V0",
        baseline,
    )

    # --------------------------------------------------------
    # Client A-only model
    # --------------------------------------------------------

    model_a = build_model()

    load_global_snapshot(
        model_a,
        GLOBAL_V0,
    )

    model_a = model_a.to(
        device
    )

    apply_delta(
        model_a,
        delta_a,
    )

    result_a = evaluate_all(
        model_a,
        datasets,
        device,
    )

    print_results(
        "CLIENT A UPDATE ONLY",
        result_a,
    )

    # --------------------------------------------------------
    # Client B-only model
    # --------------------------------------------------------

    model_b = build_model()

    load_global_snapshot(
        model_b,
        GLOBAL_V0,
    )

    model_b = model_b.to(
        device
    )

    apply_delta(
        model_b,
        delta_b,
    )

    result_b = evaluate_all(
        model_b,
        datasets,
        device,
    )

    print_results(
        "CLIENT B UPDATE ONLY",
        result_b,
    )

    # --------------------------------------------------------
    # FEDAVG MODEL
    # --------------------------------------------------------

    global_v1 = build_model()

    load_global_snapshot(
        global_v1,
        GLOBAL_V0,
    )

    global_v1 = global_v1.to(
        device
    )

    apply_delta(
        global_v1,
        aggregated_delta,
    )

    federated = evaluate_all(
        global_v1,
        datasets,
        device,
    )

    print_results(
        "GLOBAL V1 - FEDAVG",
        federated,
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("GLOBAL V0 -> GLOBAL V1")
    print("=" * 78)

    for domain in range(
        NUM_DOMAINS
    ):

        loss_delta = (
            federated[domain]["loss"]
            - baseline[domain]["loss"]
        )

        accuracy_delta = (
            federated[domain]["accuracy"]
            - baseline[domain]["accuracy"]
        )

        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss_delta={loss_delta:+.6f} "
            f"acc_delta="
            f"{accuracy_delta * 100:+.2f}pp"
        )

    # --------------------------------------------------------
    # Communication size
    # --------------------------------------------------------

    file_a = (
        CLIENT_A_DIR
        / "expert_delta.safetensors"
    )

    file_b = (
        CLIENT_B_DIR
        / "expert_delta.safetensors"
    )

    bytes_a = file_a.stat().st_size
    bytes_b = file_b.stat().st_size

    print()
    print("=" * 78)
    print("COMMUNICATION")
    print("=" * 78)

    print(
        f"Client A upload: "
        f"{bytes_a:,} bytes"
    )

    print(
        f"Client B upload: "
        f"{bytes_b:,} bytes"
    )

    print(
        f"Total upload: "
        f"{bytes_a + bytes_b:,} bytes"
    )

    # --------------------------------------------------------
    # Save GLOBAL V1
    # --------------------------------------------------------

    export_global_snapshot(
        model=global_v1,
        output_dir=GLOBAL_V1,
        global_version=31,
    )

    report = {
        "from_global_version": 30,
        "to_global_version": 31,

        "layer": TARGET_LAYER,
        "expert": TARGET_EXPERT,

        "client_A_tokens":
            weight_a,

        "client_B_tokens":
            weight_b,

        "delta_cosine_similarity":
            similarity,

        "client_A_upload_bytes":
            bytes_a,

        "client_B_upload_bytes":
            bytes_b,

        "baseline":
            baseline,

        "federated":
            federated,
    }

    report_path = Path(
        "results/m5c_round_0001.json"
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
        "Global V1 saved:"
    )

    print(
        GLOBAL_V1
    )

    print()
    print(
        "Report:"
    )

    print(
        report_path
    )

    print()
    print("=" * 78)
    print(
        "FIRST FEDERATED ROUND COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
