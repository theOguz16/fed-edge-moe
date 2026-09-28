import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ml.model.transformer import MiniMoELM
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
    export_global_snapshot,
)


GLOBAL_BASE = "checkpoints/m5_candidates/global_step_0030"

CLIENT_A_DELTA = Path(
    "checkpoints/m5b/client_A/expert_delta.safetensors"
)

CLIENT_A_META = Path(
    "checkpoints/m5b/client_A/metadata.json"
)

CLIENT_B_DELTA = Path(
    "checkpoints/m7/uploads/client_B_expert_delta.safetensors"
)

CLIENT_B_META = Path(
    "checkpoints/m7/uploads/client_B_metadata.json"
)

TARGET_LAYER = 0
TARGET_EXPERT = 6

OUTPUT_GLOBAL = "checkpoints/m7/global_v0031_physical"
OUTPUT_AGG_DELTA = Path(
    "checkpoints/m7/aggregated_expert_delta.safetensors"
)

REPORT_PATH = Path(
    "results/m7_physical_round.json"
)

SEQ_LEN = 16

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


def build_eval_set(domain, device):
    starts = torch.arange(
        0,
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
def evaluate_domain(model, domain, device):
    model.eval()

    x, y = build_eval_set(
        domain,
        device,
    )

    output = model(
        input_ids=x,
        labels=y,
    )

    predictions = output.logits.argmax(
        dim=-1
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


def evaluate_all(model, device):
    return [
        evaluate_domain(
            model,
            domain,
            device,
        )
        for domain in range(4)
    ]


def print_results(title, results):
    print()
    print(title)

    mean_loss = 0.0
    mean_accuracy = 0.0

    for domain, result in enumerate(results):
        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss={result['loss']:.6f} "
            f"acc={result['accuracy'] * 100:6.2f}%"
        )

        mean_loss += result["loss"]
        mean_accuracy += result["accuracy"]

    mean_loss /= 4
    mean_accuracy /= 4

    print(
        f"MEAN     "
        f"loss={mean_loss:.6f} "
        f"acc={mean_accuracy * 100:6.2f}%"
    )

    return mean_loss, mean_accuracy


def flatten_delta(delta):
    return torch.cat([
        delta[key].float().reshape(-1)
        for key in sorted(delta.keys())
    ])


def cosine_similarity(a, b):
    va = flatten_delta(a)
    vb = flatten_delta(b)

    return float(
        torch.nn.functional.cosine_similarity(
            va.unsqueeze(0),
            vb.unsqueeze(0),
        ).item()
    )


def weighted_fedavg(
    delta_a,
    weight_a,
    delta_b,
    weight_b,
):
    total = weight_a + weight_b

    result = {}

    for key in delta_a:
        result[key] = (
            delta_a[key] * weight_a
            + delta_b[key] * weight_b
        ) / total

        result[key] = (
            result[key]
            .contiguous()
        )

    return result


def apply_expert_delta(
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

    updated = {}

    for key, tensor in state.items():
        updated[key] = (
            tensor
            + delta[key].to(
                device=tensor.device,
                dtype=tensor.dtype,
            )
        )

    expert.load_state_dict(
        updated
    )


def main():
    device = get_device()

    print()
    print("FedEdgeMoE - M7B")
    print("Physical Federated Round Finalization")
    print()
    print("Aggregator device:", device)

    for path in [
        CLIENT_A_DELTA,
        CLIENT_A_META,
        CLIENT_B_DELTA,
        CLIENT_B_META,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    with open(
        CLIENT_A_META,
        "r",
        encoding="utf-8",
    ) as f:
        meta_a = json.load(f)

    with open(
        CLIENT_B_META,
        "r",
        encoding="utf-8",
    ) as f:
        meta_b = json.load(f)

    assert (
        meta_a["base_global_version"]
        == 30
    )

    assert (
        meta_b["base_global_version"]
        == 30
    )

    assert meta_a["layer"] == TARGET_LAYER
    assert meta_b["layer"] == TARGET_LAYER

    assert meta_a["expert"] == TARGET_EXPERT
    assert meta_b["expert"] == TARGET_EXPERT

    delta_a = load_file(
        str(CLIENT_A_DELTA)
    )

    delta_b = load_file(
        str(CLIENT_B_DELTA)
    )

    if set(delta_a.keys()) != set(delta_b.keys()):
        raise RuntimeError(
            "Client delta parameter keys do not match."
        )

    weight_a = int(
        meta_a["local_tokens_seen"]
    )

    weight_b = int(
        meta_b["local_tokens_seen"]
    )

    print()
    print("CLIENTS")
    print(
        f"A: Mac simulation / D0 / "
        f"{weight_a:,} tokens"
    )

    print(
        f"B: Physical MSI CUDA / D1 / "
        f"{weight_b:,} tokens"
    )

    similarity = cosine_similarity(
        delta_a,
        delta_b,
    )

    print()
    print(
        "Delta cosine similarity:",
        f"{similarity:+.6f}",
    )

    aggregated = weighted_fedavg(
        delta_a,
        weight_a,
        delta_b,
        weight_b,
    )

    OUTPUT_AGG_DELTA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        aggregated,
        str(OUTPUT_AGG_DELTA),
    )

    base_model = build_model()

    load_global_snapshot(
        base_model,
        GLOBAL_BASE,
    )

    base_model = base_model.to(device)

    baseline = evaluate_all(
        base_model,
        device,
    )

    global_model = build_model()

    load_global_snapshot(
        global_model,
        GLOBAL_BASE,
    )

    global_model = global_model.to(device)

    apply_expert_delta(
        global_model,
        aggregated,
    )

    physical_fedavg = evaluate_all(
        global_model,
        device,
    )

    print_results(
        "GLOBAL V30",
        baseline,
    )

    print_results(
        "GLOBAL V31 - PHYSICAL FEDAVG",
        physical_fedavg,
    )

    print()
    print("=" * 78)
    print("GLOBAL V30 -> PHYSICAL GLOBAL V31")
    print("=" * 78)

    for domain in range(4):
        loss_delta = (
            physical_fedavg[domain]["loss"]
            - baseline[domain]["loss"]
        )

        acc_delta = (
            physical_fedavg[domain]["accuracy"]
            - baseline[domain]["accuracy"]
        )

        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss_delta={loss_delta:+.6f} "
            f"acc_delta={acc_delta * 100:+.2f}pp"
        )

    export_global_snapshot(
        model=global_model,
        output_dir=OUTPUT_GLOBAL,
        global_version=31,
    )

    client_a_bytes = (
        CLIENT_A_DELTA.stat().st_size
    )

    client_b_bytes = (
        CLIENT_B_DELTA.stat().st_size
    )

    metadata_b_bytes = (
        CLIENT_B_META.stat().st_size
    )

    print()
    print("=" * 78)
    print("PHYSICAL NETWORK UPDATE")
    print("=" * 78)

    print(
        f"Client A delta: "
        f"{client_a_bytes:,} bytes"
    )

    print(
        f"MSI Client B delta: "
        f"{client_b_bytes:,} bytes"
    )

    print(
        f"MSI metadata: "
        f"{metadata_b_bytes:,} bytes"
    )

    print(
        f"MSI upload payload total: "
        f"{client_b_bytes + metadata_b_bytes:,} bytes"
    )

    report = {
        "round": 1,
        "base_global_version": 30,
        "new_global_version": 31,

        "target_layer": TARGET_LAYER,
        "target_expert": TARGET_EXPERT,

        "client_A": meta_a,
        "client_B_physical": meta_b,

        "delta_cosine_similarity":
            similarity,

        "baseline":
            baseline,

        "physical_fedavg":
            physical_fedavg,

        "client_A_delta_bytes":
            client_a_bytes,

        "client_B_delta_bytes":
            client_b_bytes,

        "client_B_metadata_bytes":
            metadata_b_bytes,

        "aggregated_delta_file":
            str(OUTPUT_AGG_DELTA),

        "global_checkpoint":
            OUTPUT_GLOBAL,
    }

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
            report,
            f,
            indent=2,
        )

    print()
    print(
        "Global checkpoint:",
        OUTPUT_GLOBAL,
    )

    print(
        "Report:",
        REPORT_PATH,
    )

    print()
    print("=" * 78)
    print("FIRST PHYSICAL FEDERATED ROUND COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
