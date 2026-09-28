import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v3 import (
    VOCAB_SIZE,
    NUM_DOMAINS,
    build_split_tensor,
)
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
    export_global_snapshot,
)


SEED = 20260929

BASE_CHECKPOINT = (
    "checkpoints/m11e_replication/"
    "global_step_0100"
)

OUTPUT_ROOT = Path(
    "checkpoints/m11e2_scheduler_test"
)

REPORT_PATH = Path(
    "results/m11e2_independent_scheduler_test.json"
)

SEQ_LEN = 16
BATCH_SIZE = 32
LOCAL_STEPS = 120
LEARNING_RATE = 5e-4
ROUNDS = 5

NUM_LAYERS = 3
NUM_EXPERTS = 8


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


@torch.no_grad()
def evaluate_validation(
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
                "validation",
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

        accuracy = (
            logits.argmax(dim=-1)
            == labels
        ).float().mean()

        results.append({
            "domain": domain,

            "loss": float(
                loss.detach().cpu()
            ),

            "accuracy": float(
                accuracy.detach().cpu()
            ),
        })

    return results


def global_mean(results):
    return (
        sum(
            x["accuracy"]
            for x in results
        )
        / len(results)
    )


def target_mean(results):
    return (
        results[0]["accuracy"]
        + results[1]["accuracy"]
    ) / 2


def retention_mean(results):
    return (
        results[2]["accuracy"]
        + results[3]["accuracy"]
    ) / 2


def collect_train_routing(
    model,
    domain,
    device,
):
    captured = [
        None
        for _ in range(NUM_LAYERS)
    ]

    hooks = []

    for layer_index, block in enumerate(
        model.blocks
    ):
        def make_hook(index):
            def hook(
                module,
                inputs,
                output,
            ):
                captured[index] = (
                    output.topk_indices
                    .detach()
                    .cpu()
                )

            return hook

        hooks.append(
            block.moe.router
            .register_forward_hook(
                make_hook(layer_index)
            )
        )

    sequences = (
        build_split_tensor(
            domain,
            "train",
            SEQ_LEN,
        )
        .to(device)
    )

    model.eval()

    with torch.no_grad():
        _ = model(
            input_ids=
                sequences[:, :-1]
        )

    shares = torch.zeros(
        NUM_LAYERS,
        NUM_EXPERTS,
        dtype=torch.float32,
    )

    for layer in range(
        NUM_LAYERS
    ):
        selected = (
            captured[layer][:, 1:, :]
            .reshape(-1)
        )

        counts = torch.bincount(
            selected,
            minlength=NUM_EXPERTS,
        ).float()

        shares[layer] = (
            counts / counts.sum()
        )

    for handle in hooks:
        handle.remove()

    return shares


def build_scheduler_rankings(
    shares_a,
    shares_b,
):
    candidates = []

    for layer in range(
        NUM_LAYERS
    ):
        for expert in range(
            NUM_EXPERTS
        ):
            a = float(
                shares_a[
                    layer,
                    expert,
                ]
            )

            b = float(
                shares_b[
                    layer,
                    expert,
                ]
            )

            minimum = min(a, b)
            maximum = max(a, b)

            if maximum > 0:
                balance = (
                    minimum / maximum
                )
            else:
                balance = 0.0

            balanced = (
                minimum * balance
            )

            candidates.append({
                "layer":
                    layer,

                "expert":
                    expert,

                "A":
                    a,

                "B":
                    b,

                "shared_min":
                    minimum,

                "balance_ratio":
                    balance,

                "balanced_score":
                    balanced,
            })

    original = sorted(
        candidates,
        key=lambda x: (
            x["shared_min"],
            (x["A"] + x["B"]) / 2,
        ),
        reverse=True,
    )

    balanced = sorted(
        candidates,
        key=lambda x: (
            x["balanced_score"],
            x["shared_min"],
        ),
        reverse=True,
    )

    return (
        original,
        balanced,
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
    a,
    b,
):
    va = flatten_delta(a)
    vb = flatten_delta(b)

    return float(
        F.cosine_similarity(
            va.unsqueeze(0),
            vb.unsqueeze(0),
        ).item()
    )


def train_client(
    snapshot,
    domain,
    layer,
    expert_index,
    round_number,
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
        .blocks[layer]
        .moe
        .experts[expert_index]
    )

    for parameter in (
        expert.parameters()
    ):
        parameter.requires_grad = True

    trainable = list(
        expert.parameters()
    )

    before = clone_expert(
        expert
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    pool = build_split_tensor(
        domain,
        "train",
        SEQ_LEN,
    )

    model.train()

    running = 0.0

    for step in range(
        1,
        LOCAL_STEPS + 1,
    ):
        indices = torch.randint(
            0,
            pool.shape[0],
            (BATCH_SIZE,),
        )

        sequences = (
            pool[indices]
            .to(device)
        )

        x = sequences[:, :-1]
        y = sequences[:, 1:]

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

        running += float(
            loss.detach().cpu()
        )

        if step % 40 == 0:
            print(
                f"D{domain} "
                f"step {step:03d}/"
                f"{LOCAL_STEPS} "
                f"avg_loss="
                f"{running / 40:.6f}"
            )

            running = 0.0

    after = clone_expert(
        expert
    )

    return calculate_delta(
        before,
        after,
    )


def fedavg(
    delta_a,
    delta_b,
):
    return {
        key: (
            (
                delta_a[key]
                + delta_b[key]
            )
            / 2
        ).contiguous()

        for key in delta_a
    }


def apply_delta(
    model,
    layer,
    expert_index,
    delta,
):
    expert = (
        model
        .blocks[layer]
        .moe
        .experts[expert_index]
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


def run_fixed_horizon(
    label,
    layer,
    expert_index,
    device,
):
    print()
    print("=" * 88)
    print(
        f"{label}: "
        f"L{layer}-E{expert_index}"
    )
    print("=" * 88)

    output_dir = (
        OUTPUT_ROOT
        / label
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_snapshot = (
        BASE_CHECKPOINT
    )

    current_version = 100

    base_model = build_model()

    load_global_snapshot(
        base_model,
        current_snapshot,
    )

    base_model = (
        base_model.to(device)
    )

    baseline = (
        evaluate_validation(
            base_model,
            device,
        )
    )

    base_global = (
        global_mean(
            baseline
        )
    )

    base_target = (
        target_mean(
            baseline
        )
    )

    base_retention = (
        retention_mean(
            baseline
        )
    )

    history = []

    for round_number in range(
        1,
        ROUNDS + 1,
    ):
        print()
        print(
            f"Round {round_number}/"
            f"{ROUNDS}"
        )

        delta_a = train_client(
            snapshot=current_snapshot,
            domain=0,
            layer=layer,
            expert_index=expert_index,
            round_number=round_number,
            device=device,
        )

        delta_b = train_client(
            snapshot=current_snapshot,
            domain=1,
            layer=layer,
            expert_index=expert_index,
            round_number=round_number,
            device=device,
        )

        cosine = (
            cosine_similarity(
                delta_a,
                delta_b,
            )
        )

        aggregate = fedavg(
            delta_a,
            delta_b,
        )

        candidate = build_model()

        load_global_snapshot(
            candidate,
            current_snapshot,
        )

        candidate = (
            candidate.to(device)
        )

        apply_delta(
            candidate,
            layer,
            expert_index,
            aggregate,
        )

        validation = (
            evaluate_validation(
                candidate,
                device,
            )
        )

        current_global = (
            global_mean(
                validation
            )
        )

        current_target = (
            target_mean(
                validation
            )
        )

        current_retention = (
            retention_mean(
                validation
            )
        )

        print(
            f"cos={cosine:+.6f} "
            f"| global="
            f"{current_global * 100:.2f}% "
            f"| gain="
            f"{(current_global - base_global) * 100:+.2f}pp "
            f"| D0/D1="
            f"{current_target * 100:.2f}% "
            f"| D2/D3="
            f"{current_retention * 100:.2f}%"
        )

        next_version = (
            current_version + 1
        )

        next_snapshot = (
            output_dir
            / (
                f"global_v"
                f"{next_version:04d}"
            )
        )

        export_global_snapshot(
            model=candidate,
            output_dir=next_snapshot,
            global_version=
                next_version,
        )

        history.append({
            "round":
                round_number,

            "version":
                next_version,

            "cosine":
                cosine,

            "global_accuracy":
                current_global,

            "global_gain":
                current_global
                - base_global,

            "target_accuracy":
                current_target,

            "target_gain":
                current_target
                - base_target,

            "retention_accuracy":
                current_retention,

            "retention_change":
                current_retention
                - base_retention,
        })

        current_snapshot = (
            next_snapshot
        )

        current_version = (
            next_version
        )

    best = max(
        history,
        key=lambda x:
            x["global_accuracy"],
    )

    final = history[-1]

    return {
        "label":
            label,

        "layer":
            layer,

        "expert":
            expert_index,

        "baseline_global":
            base_global,

        "baseline_target":
            base_target,

        "baseline_retention":
            base_retention,

        "best_round":
            best["round"],

        "best_global":
            best[
                "global_accuracy"
            ],

        "best_global_gain":
            best[
                "global_gain"
            ],

        "final_global":
            final[
                "global_accuracy"
            ],

        "final_global_gain":
            final[
                "global_gain"
            ],

        "final_target":
            final[
                "target_accuracy"
            ],

        "final_target_gain":
            final[
                "target_gain"
            ],

        "final_retention":
            final[
                "retention_accuracy"
            ],

        "final_retention_change":
            final[
                "retention_change"
            ],

        "history":
            history,
    }


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
    print("FedEdgeMoE - M11E2")
    print(
        "Independent Scheduler Test"
    )

    print()
    print("Fresh dataset: Synthetic V3")
    print("Base: pre-registered Step 100")
    print("Scheduler formulas: LOCKED")
    print("Fixed horizon: 5 rounds")
    print("TEST: SEALED / NOT USED")

    model = build_model()

    load_global_snapshot(
        model,
        BASE_CHECKPOINT,
    )

    model = model.to(device)

    shares_a = (
        collect_train_routing(
            model,
            0,
            device,
        )
    )

    shares_b = (
        collect_train_routing(
            model,
            1,
            device,
        )
    )

    (
        original_ranking,
        balanced_ranking,
    ) = build_scheduler_rankings(
        shares_a,
        shares_b,
    )

    print()
    print("=" * 90)
    print("ORIGINAL SHARED-MIN TOP 5")
    print("=" * 90)

    for rank, item in enumerate(
        original_ranking[:5],
        start=1,
    ):
        print(
            f"{rank}. "
            f"L{item['layer']}-"
            f"E{item['expert']} "
            f"| A={item['A'] * 100:5.1f}% "
            f"B={item['B'] * 100:5.1f}% "
            f"| min="
            f"{item['shared_min'] * 100:5.1f}%"
        )

    print()
    print("=" * 90)
    print("BALANCED TOP 5")
    print("=" * 90)

    for rank, item in enumerate(
        balanced_ranking[:5],
        start=1,
    ):
        print(
            f"{rank}. "
            f"L{item['layer']}-"
            f"E{item['expert']} "
            f"| A={item['A'] * 100:5.1f}% "
            f"B={item['B'] * 100:5.1f}% "
            f"| min="
            f"{item['shared_min'] * 100:5.1f}% "
            f"| balance="
            f"{item['balance_ratio'] * 100:5.1f}% "
            f"| score="
            f"{item['balanced_score'] * 100:6.2f}"
        )

    original_pick = (
        original_ranking[0]
    )

    balanced_pick = (
        balanced_ranking[0]
    )

    print()
    print("=" * 90)
    print("LOCKED SCHEDULER DECISIONS")
    print("=" * 90)

    print(
        "Original:",
        f"L{original_pick['layer']}-"
        f"E{original_pick['expert']}",
    )

    print(
        "Balanced:",
        f"L{balanced_pick['layer']}-"
        f"E{balanced_pick['expert']}",
    )

    results = []

    result_original = (
        run_fixed_horizon(
            label="original",
            layer=
                original_pick["layer"],
            expert_index=
                original_pick["expert"],
            device=device,
        )
    )

    results.append(
        result_original
    )

    same_selection = (
        original_pick["layer"]
        == balanced_pick["layer"]
        and
        original_pick["expert"]
        == balanced_pick["expert"]
    )

    if same_selection:
        result_balanced = dict(
            result_original
        )

        result_balanced[
            "label"
        ] = "balanced"

        result_balanced[
            "reused_same_experiment"
        ] = True

    else:
        result_balanced = (
            run_fixed_horizon(
                label="balanced",
                layer=
                    balanced_pick[
                        "layer"
                    ],
                expert_index=
                    balanced_pick[
                        "expert"
                    ],
                device=device,
            )
        )

    results.append(
        result_balanced
    )

    print()
    print("=" * 96)
    print("M11E2 INDEPENDENT RESULT")
    print("=" * 96)

    for result in results:
        print(
            f"{result['label']:<10} "
            f"L{result['layer']}-"
            f"E{result['expert']} "
            f"| best="
            f"{result['best_global'] * 100:.2f}% "
            f"(R{result['best_round']}) "
            f"| final="
            f"{result['final_global'] * 100:.2f}% "
            f"| gain="
            f"{result['final_global_gain'] * 100:+.2f}pp "
            f"| target="
            f"{result['final_target_gain'] * 100:+.2f}pp "
            f"| retention="
            f"{result['final_retention_change'] * 100:+.2f}pp"
        )

    if same_selection:
        verdict = (
            "same_expert_selected"
        )

    elif (
        result_balanced[
            "final_global"
        ]
        >
        result_original[
            "final_global"
        ]
    ):
        verdict = (
            "balanced_higher"
        )

    elif (
        result_balanced[
            "final_global"
        ]
        <
        result_original[
            "final_global"
        ]
    ):
        verdict = (
            "original_higher"
        )

    else:
        verdict = "tie"

    print()
    print(
        "Independent comparison:",
        verdict,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "experiment":
                    "M11E2 independent scheduler replication",

                "synthetic_version":
                    "V3",

                "base_checkpoint":
                    BASE_CHECKPOINT,

                "scheduler_formulas_locked":
                    True,

                "test_used":
                    False,

                "fixed_horizon_rounds":
                    ROUNDS,

                "original_pick":
                    original_pick,

                "balanced_pick":
                    balanced_pick,

                "same_selection":
                    same_selection,

                "results":
                    results,

                "comparison":
                    verdict,

                "original_ranking":
                    original_ranking,

                "balanced_ranking":
                    balanced_ranking,
            },
            f,
            indent=2,
        )

    print()
    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
