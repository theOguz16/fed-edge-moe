import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v2 import (
    VOCAB_SIZE,
    NUM_DOMAINS,
    build_split_tensor,
)
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
    export_global_snapshot,
)


SEED = 20260927

GLOBAL_BASE = (
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

START_VERSION = 100
MAX_ROUNDS = 5

TARGET_LAYER = 1
TARGET_EXPERT = 4

CLIENT_A_DOMAIN = 0
CLIENT_B_DOMAIN = 1

LOCAL_STEPS = 120
BATCH_SIZE = 32
SEQ_LEN = 16
LEARNING_RATE = 5e-4

MIN_VALIDATION_IMPROVEMENT = 0.001

OUTPUT_ROOT = Path(
    "checkpoints/m10c"
)

REPORT_PATH = Path(
    "results/m10c_clean_validation_fl.json"
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


def generate_domain_train_batch(
    domain,
    device,
):
    pool = build_split_tensor(
        domain,
        "train",
        SEQ_LEN,
    )

    indices = torch.randint(
        0,
        pool.shape[0],
        (BATCH_SIZE,),
    )

    sequences = (
        pool[indices]
        .to(device)
    )

    return (
        sequences[:, :-1],
        sequences[:, 1:],
    )


@torch.no_grad()
def evaluate_domain(
    model,
    domain,
    split,
    device,
):
    model.eval()

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

    return {
        "loss": float(
            loss.detach().cpu()
        ),

        "accuracy": float(
            accuracy.detach().cpu()
        ),
    }


def evaluate_all_validation(
    model,
    device,
):
    return [
        evaluate_domain(
            model,
            domain,
            "validation",
            device,
        )
        for domain in range(
            NUM_DOMAINS
        )
    ]


def mean_accuracy(results):
    return (
        sum(
            result["accuracy"]
            for result in results
        )
        / len(results)
    )


def mean_loss(results):
    return (
        sum(
            result["loss"]
            for result in results
        )
        / len(results)
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
            f"{DOMAIN_NAMES[domain]} "
            f"loss={result['loss']:.6f} "
            f"acc="
            f"{result['accuracy'] * 100:6.2f}%"
        )

    print(
        "MEAN "
        f"loss={mean_loss(results):.6f} "
        f"acc={mean_accuracy(results) * 100:6.2f}%"
    )


def clone_expert(
    expert,
):
    return {
        key:
            value.detach()
            .cpu()
            .clone()

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


def delta_norm(
    delta,
):
    total = 0.0

    for tensor in delta.values():
        total += float(
            torch.sum(
                tensor.float() ** 2
            )
        )

    return total ** 0.5


def flatten_delta(
    delta,
):
    return torch.cat([
        delta[key]
        .float()
        .reshape(-1)

        for key in sorted(
            delta.keys()
        )
    ])


def cosine_similarity(
    delta_a,
    delta_b,
):
    a = flatten_delta(
        delta_a
    )

    b = flatten_delta(
        delta_b
    )

    return float(
        F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
        ).item()
    )


def train_client(
    snapshot,
    domain,
    client_id,
    round_number,
    base_version,
    device,
):
    torch.manual_seed(
        SEED
        + round_number * 100
        + domain
    )

    model = build_model()

    load_global_snapshot(
        model,
        snapshot,
    )

    model = model.to(device)

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    expert = (
        model
        .blocks[TARGET_LAYER]
        .moe
        .experts[TARGET_EXPERT]
    )

    for parameter in (
        expert.parameters()
    ):
        parameter.requires_grad = True

    trainable = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    before = clone_expert(
        expert
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    model.train()

    running_loss = 0.0

    for step in range(
        1,
        LOCAL_STEPS + 1,
    ):
        x, y = (
            generate_domain_train_batch(
                domain,
                device,
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
            trainable,
            max_norm=1.0,
        )

        optimizer.step()

        running_loss += float(
            loss.detach().cpu()
        )

        if step % 40 == 0:
            print(
                f"{client_id} "
                f"step {step:03d}/"
                f"{LOCAL_STEPS} "
                f"avg_loss="
                f"{running_loss / 40:.6f}"
            )

            running_loss = 0.0

    after = clone_expert(
        expert
    )

    delta = calculate_delta(
        before,
        after,
    )

    output_dir = (
        OUTPUT_ROOT
        / f"round_{round_number:02d}"
        / client_id
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

    metadata = {
        "client_id":
            client_id,

        "round":
            round_number,

        "domain":
            domain,

        "data_split":
            "train",

        "base_global_version":
            base_version,

        "layer":
            TARGET_LAYER,

        "expert":
            TARGET_EXPERT,

        "local_steps":
            LOCAL_STEPS,

        "batch_size":
            BATCH_SIZE,

        "local_tokens_seen":
            LOCAL_STEPS
            * BATCH_SIZE
            * (SEQ_LEN - 1),

        "trainable_parameters":
            sum(
                parameter.numel()
                for parameter
                in trainable
            ),

        "delta_l2_norm":
            delta_norm(delta),

        "delta_bytes":
            delta_path.stat().st_size,
    }

    return (
        delta,
        metadata,
    )


def fedavg(
    delta_a,
    weight_a,
    delta_b,
    weight_b,
):
    total = (
        weight_a + weight_b
    )

    result = {}

    for key in delta_a:
        result[key] = (
            (
                delta_a[key]
                * weight_a
                +
                delta_b[key]
                * weight_b
            )
            / total
        ).contiguous()

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

    state = (
        expert.state_dict()
    )

    updated = {}

    for key, tensor in (
        state.items()
    ):
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
    print("FedEdgeMoE - M10C")
    print(
        "Clean Train-Only Federated Learning"
    )

    print()
    print("Device:", device)

    print(
        f"Shared expert: "
        f"L{TARGET_LAYER}-E{TARGET_EXPERT}"
    )

    print()
    print(
        "CLIENTS USE: TRAIN"
    )

    print(
        "SERVER GATE USES: VALIDATION"
    )

    print(
        "TEST SPLIT: SEALED"
    )

    current_snapshot = (
        GLOBAL_BASE
    )

    current_version = (
        START_VERSION
    )

    base_model = build_model()

    load_global_snapshot(
        base_model,
        current_snapshot,
    )

    base_model = (
        base_model.to(device)
    )

    current_validation = (
        evaluate_all_validation(
            base_model,
            device,
        )
    )

    current_mean = (
        mean_accuracy(
            current_validation
        )
    )

    print_results(
        f"GLOBAL V{current_version} "
        f"VALIDATION",
        current_validation,
    )

    history = [{
        "round":
            0,

        "version":
            current_version,

        "accepted":
            True,

        "validation":
            current_validation,

        "validation_mean_accuracy":
            current_mean,
    }]

    for round_number in range(
        1,
        MAX_ROUNDS + 1,
    ):
        print()
        print("=" * 80)

        print(
            f"ROUND {round_number}/"
            f"{MAX_ROUNDS}"
        )

        print(
            f"BASE V{current_version}"
        )

        print("=" * 80)

        print()
        print(
            "Training Client A "
            "on D0 TRAIN..."
        )

        delta_a, meta_a = (
            train_client(
                snapshot=current_snapshot,

                domain=
                    CLIENT_A_DOMAIN,

                client_id=
                    "client_A",

                round_number=
                    round_number,

                base_version=
                    current_version,

                device=device,
            )
        )

        print()
        print(
            "Training Client B "
            "on D1 TRAIN..."
        )

        delta_b, meta_b = (
            train_client(
                snapshot=current_snapshot,

                domain=
                    CLIENT_B_DOMAIN,

                client_id=
                    "client_B",

                round_number=
                    round_number,

                base_version=
                    current_version,

                device=device,
            )
        )

        similarity = (
            cosine_similarity(
                delta_a,
                delta_b,
            )
        )

        aggregated = fedavg(
            delta_a,
            meta_a[
                "local_tokens_seen"
            ],
            delta_b,
            meta_b[
                "local_tokens_seen"
            ],
        )

        candidate = (
            build_model()
        )

        load_global_snapshot(
            candidate,
            current_snapshot,
        )

        candidate = (
            candidate.to(device)
        )

        apply_expert_delta(
            candidate,
            aggregated,
        )

        validation = (
            evaluate_all_validation(
                candidate,
                device,
            )
        )

        candidate_mean = (
            mean_accuracy(
                validation
            )
        )

        improvement = (
            candidate_mean
            - current_mean
        )

        accepted = (
            improvement
            >=
            MIN_VALIDATION_IMPROVEMENT
        )

        print()
        print(
            "Delta cosine similarity:",
            f"{similarity:+.6f}",
        )

        print_results(
            "CANDIDATE VALIDATION",
            validation,
        )

        print()
        print(
            "Validation improvement:",
            f"{improvement * 100:+.2f} pp",
        )

        print(
            "Decision:",
            "ACCEPT"
            if accepted
            else "REJECT",
        )

        candidate_version = (
            current_version + 1
        )

        history.append({
            "round":
                round_number,

            "base_version":
                current_version,

            "candidate_version":
                candidate_version,

            "accepted":
                accepted,

            "delta_cosine_similarity":
                similarity,

            "validation":
                validation,

            "validation_mean_accuracy":
                candidate_mean,

            "validation_improvement":
                improvement,

            "client_A":
                meta_a,

            "client_B":
                meta_b,
        })

        if not accepted:
            print()
            print(
                "EARLY STOP."
            )

            break

        next_snapshot = (
            OUTPUT_ROOT
            / (
                f"global_v"
                f"{candidate_version:04d}"
            )
        )

        export_global_snapshot(
            model=candidate,
            output_dir=next_snapshot,
            global_version=
                candidate_version,
        )

        current_snapshot = (
            next_snapshot
        )

        current_version = (
            candidate_version
        )

        current_validation = (
            validation
        )

        current_mean = (
            candidate_mean
        )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "seed":
                    SEED,

                "base_checkpoint":
                    GLOBAL_BASE,

                "final_checkpoint":
                    str(
                        current_snapshot
                    ),

                "final_version":
                    current_version,

                "target_layer":
                    TARGET_LAYER,

                "target_expert":
                    TARGET_EXPERT,

                "client_data_split":
                    "train",

                "selection_split":
                    "validation",

                "test_evaluated":
                    False,

                "history":
                    history,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 80)

    print(
        "M10C FINAL ACCEPTED MODEL"
    )

    print("=" * 80)

    print(
        "Version:",
        current_version,
    )

    print(
        "Checkpoint:",
        current_snapshot,
    )

    print_results(
        "FINAL VALIDATION",
        current_validation,
    )

    print()
    print(
        "TEST SPLIT REMAINS SEALED."
    )

    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
