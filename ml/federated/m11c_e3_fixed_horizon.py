import json
from pathlib import Path

import torch

import ml.federated.m10c_clean_federated_training as core


TARGET_LAYER = 0
TARGET_EXPERT = 3

ROUNDS = 5

OUTPUT_ROOT = Path(
    "checkpoints/m11c_e3_diagnostic"
)

REPORT_PATH = Path(
    "results/m11c_e3_fixed_horizon.json"
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


def main():
    device = core.get_device()

    core.TARGET_LAYER = TARGET_LAYER
    core.TARGET_EXPERT = TARGET_EXPERT

    core.OUTPUT_ROOT = OUTPUT_ROOT

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("FedEdgeMoE - M11C")
    print("L0-E3 Fixed-Horizon Diagnostic")
    print()
    print("Device:", device)
    print("Expert: L0-E3")
    print()
    print("TRAIN → local updates")
    print("VALIDATION → observation only")
    print("NO EARLY STOP")
    print("TEST → NOT USED")

    current_snapshot = (
        core.GLOBAL_BASE
    )

    current_version = (
        core.START_VERSION
    )

    model = core.build_model()

    core.load_global_snapshot(
        model,
        current_snapshot,
    )

    model = model.to(device)

    baseline = (
        core.evaluate_all_validation(
            model,
            device,
        )
    )

    baseline_global = (
        core.mean_accuracy(
            baseline
        )
    )

    baseline_target = (
        target_mean(
            baseline
        )
    )

    baseline_retention = (
        retention_mean(
            baseline
        )
    )

    print()
    print(
        "BASELINE"
    )

    print(
        f"Global:   "
        f"{baseline_global * 100:.2f}%"
    )

    print(
        f"D0/D1:    "
        f"{baseline_target * 100:.2f}%"
    )

    print(
        f"D2/D3:    "
        f"{baseline_retention * 100:.2f}%"
    )

    history = []

    previous_global = (
        baseline_global
    )

    for round_number in range(
        1,
        ROUNDS + 1,
    ):
        print()
        print("=" * 84)

        print(
            f"ROUND {round_number}/{ROUNDS}"
        )

        print("=" * 84)

        delta_a, meta_a = (
            core.train_client(
                snapshot=current_snapshot,
                domain=0,
                client_id="client_A",
                round_number=round_number,
                base_version=current_version,
                device=device,
            )
        )

        delta_b, meta_b = (
            core.train_client(
                snapshot=current_snapshot,
                domain=1,
                client_id="client_B",
                round_number=round_number,
                base_version=current_version,
                device=device,
            )
        )

        cosine = (
            core.cosine_similarity(
                delta_a,
                delta_b,
            )
        )

        aggregated = (
            core.fedavg(
                delta_a,
                meta_a[
                    "local_tokens_seen"
                ],
                delta_b,
                meta_b[
                    "local_tokens_seen"
                ],
            )
        )

        candidate = (
            core.build_model()
        )

        core.load_global_snapshot(
            candidate,
            current_snapshot,
        )

        candidate = (
            candidate.to(device)
        )

        core.apply_expert_delta(
            candidate,
            aggregated,
        )

        validation = (
            core.evaluate_all_validation(
                candidate,
                device,
            )
        )

        global_acc = (
            core.mean_accuracy(
                validation
            )
        )

        target_acc = (
            target_mean(
                validation
            )
        )

        retention_acc = (
            retention_mean(
                validation
            )
        )

        round_gain = (
            global_acc
            - previous_global
        )

        cumulative_gain = (
            global_acc
            - baseline_global
        )

        target_gain = (
            target_acc
            - baseline_target
        )

        retention_change = (
            retention_acc
            - baseline_retention
        )

        print()
        print(
            f"cos={cosine:+.6f}"
        )

        print(
            f"Global: "
            f"{global_acc * 100:.2f}% "
            f"| round="
            f"{round_gain * 100:+.2f}pp "
            f"| cumulative="
            f"{cumulative_gain * 100:+.2f}pp"
        )

        print(
            f"D0/D1: "
            f"{target_acc * 100:.2f}% "
            f"| gain="
            f"{target_gain * 100:+.2f}pp"
        )

        print(
            f"D2/D3: "
            f"{retention_acc * 100:.2f}% "
            f"| change="
            f"{retention_change * 100:+.2f}pp"
        )

        next_version = (
            current_version + 1
        )

        next_snapshot = (
            OUTPUT_ROOT
            / (
                f"global_v"
                f"{next_version:04d}"
            )
        )

        core.export_global_snapshot(
            model=candidate,
            output_dir=next_snapshot,
            global_version=next_version,
        )

        history.append({
            "round":
                round_number,

            "version":
                next_version,

            "delta_cosine_similarity":
                cosine,

            "global_accuracy":
                global_acc,

            "round_gain":
                round_gain,

            "cumulative_gain":
                cumulative_gain,

            "target_accuracy":
                target_acc,

            "target_gain":
                target_gain,

            "retention_accuracy":
                retention_acc,

            "retention_change":
                retention_change,

            "validation":
                validation,
        })

        current_snapshot = (
            next_snapshot
        )

        current_version = (
            next_version
        )

        previous_global = (
            global_acc
        )

    best = max(
        history,
        key=lambda item:
            item["global_accuracy"],
    )

    final = history[-1]

    print()
    print("=" * 84)
    print("M11C SUMMARY")
    print("=" * 84)

    print(
        "Best round:",
        best["round"],
    )

    print(
        "Best global:",
        f"{best['global_accuracy'] * 100:.2f}%",
    )

    print(
        "Best cumulative gain:",
        f"{best['cumulative_gain'] * 100:+.2f}pp",
    )

    print()
    print(
        "Final round:",
        final["round"],
    )

    print(
        "Final global:",
        f"{final['global_accuracy'] * 100:.2f}%",
    )

    print(
        "Final target gain:",
        f"{final['target_gain'] * 100:+.2f}pp",
    )

    print(
        "Final retention change:",
        f"{final['retention_change'] * 100:+.2f}pp",
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "experiment":
                    "L0-E3 fixed horizon",

                "target_layer":
                    TARGET_LAYER,

                "target_expert":
                    TARGET_EXPERT,

                "rounds":
                    ROUNDS,

                "early_stop":
                    False,

                "validation_used_for_selection":
                    False,

                "test_used":
                    False,

                "baseline_global_accuracy":
                    baseline_global,

                "baseline_target_accuracy":
                    baseline_target,

                "baseline_retention_accuracy":
                    baseline_retention,

                "best_round":
                    best["round"],

                "best_global_accuracy":
                    best[
                        "global_accuracy"
                    ],

                "history":
                    history,
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
