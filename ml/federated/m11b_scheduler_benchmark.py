import json
from pathlib import Path

import torch

import ml.federated.m10c_clean_federated_training as core


SCHEDULER_REPORT = Path(
    "results/m11a_affinity_scheduler.json"
)

REPORT_PATH = Path(
    "results/m11b_scheduler_benchmark.json"
)

OUTPUT_ROOT = Path(
    "checkpoints/m11b"
)

TOP_CANDIDATES = 3
MAX_ROUNDS = 5

MIN_VALIDATION_IMPROVEMENT = 0.001


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


def run_candidate(
    layer,
    expert,
    rank,
    affinity,
    device,
):
    print()
    print("=" * 88)
    print(
        f"CANDIDATE #{rank}: "
        f"L{layer}-E{expert}"
    )
    print(
        f"Scheduler shared affinity: "
        f"{affinity * 100:.2f}%"
    )
    print("=" * 88)

    # m10c helper fonksiyonları bu global
    # değerleri dinamik olarak kullanıyor.
    core.TARGET_LAYER = layer
    core.TARGET_EXPERT = expert

    core.OUTPUT_ROOT = (
        OUTPUT_ROOT
        / f"L{layer}_E{expert}"
    )

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

    current_mean = (
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

    best_validation = baseline
    accepted_rounds = 0

    history = []

    print()
    print(
        f"Baseline global mean: "
        f"{current_mean * 100:.2f}%"
    )

    print(
        f"Baseline D0/D1 mean: "
        f"{baseline_target * 100:.2f}%"
    )

    print(
        f"Baseline D2/D3 mean: "
        f"{baseline_retention * 100:.2f}%"
    )

    for round_number in range(
        1,
        MAX_ROUNDS + 1,
    ):
        print()
        print(
            f"Round {round_number}/"
            f"{MAX_ROUNDS}"
        )

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

        candidate_mean = (
            core.mean_accuracy(
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

        candidate_target = (
            target_mean(
                validation
            )
        )

        candidate_retention = (
            retention_mean(
                validation
            )
        )

        print(
            f"cos={cosine:+.6f} "
            f"| global="
            f"{candidate_mean * 100:.2f}% "
            f"| D0/D1="
            f"{candidate_target * 100:.2f}% "
            f"| D2/D3="
            f"{candidate_retention * 100:.2f}% "
            f"| gain="
            f"{improvement * 100:+.2f}pp "
            f"| "
            f"{'ACCEPT' if accepted else 'REJECT'}"
        )

        history.append({
            "round":
                round_number,

            "base_version":
                current_version,

            "delta_cosine_similarity":
                cosine,

            "global_mean_accuracy":
                candidate_mean,

            "target_mean_accuracy":
                candidate_target,

            "retention_mean_accuracy":
                candidate_retention,

            "improvement":
                improvement,

            "accepted":
                accepted,
        })

        if not accepted:
            break

        next_version = (
            current_version + 1
        )

        next_snapshot = (
            core.OUTPUT_ROOT
            / (
                f"global_v"
                f"{next_version:04d}"
            )
        )

        core.export_global_snapshot(
            model=candidate,
            output_dir=next_snapshot,
            global_version=
                next_version,
        )

        current_snapshot = (
            next_snapshot
        )

        current_version = (
            next_version
        )

        current_mean = (
            candidate_mean
        )

        best_validation = (
            validation
        )

        accepted_rounds += 1

    final_global = (
        core.mean_accuracy(
            best_validation
        )
    )

    final_target = (
        target_mean(
            best_validation
        )
    )

    final_retention = (
        retention_mean(
            best_validation
        )
    )

    return {
        "scheduler_rank":
            rank,

        "layer":
            layer,

        "expert":
            expert,

        "shared_affinity":
            affinity,

        "accepted_rounds":
            accepted_rounds,

        "baseline_global_accuracy":
            core.mean_accuracy(
                baseline
            ),

        "final_global_accuracy":
            final_global,

        "global_gain":
            final_global
            - core.mean_accuracy(
                baseline
            ),

        "baseline_target_accuracy":
            baseline_target,

        "final_target_accuracy":
            final_target,

        "target_gain":
            final_target
            - baseline_target,

        "baseline_retention_accuracy":
            baseline_retention,

        "final_retention_accuracy":
            final_retention,

        "retention_change":
            final_retention
            - baseline_retention,

        "final_checkpoint":
            str(
                current_snapshot
            ),

        "history":
            history,
    }


def main():
    torch.manual_seed(
        core.SEED
    )

    device = (
        core.get_device()
    )

    print()
    print("FedEdgeMoE - M11B")
    print(
        "Automatic Scheduler Benchmark"
    )

    print()
    print("Device:", device)

    print(
        "TRAIN data → local learning"
    )

    print(
        "VALIDATION → benchmark"
    )

    print(
        "TEST → NOT USED"
    )

    with open(
        SCHEDULER_REPORT,
        "r",
        encoding="utf-8",
    ) as f:
        scheduler = json.load(f)

    candidates = (
        scheduler["ranking"][
            :TOP_CANDIDATES
        ]
    )

    results = []

    for rank, item in enumerate(
        candidates,
        start=1,
    ):
        result = run_candidate(
            layer=item["layer"],
            expert=item["expert"],
            rank=rank,
            affinity=item[
                "shared_min_affinity"
            ],
            device=device,
        )

        results.append(
            result
        )

    by_global = sorted(
        results,
        key=lambda item:
            item[
                "final_global_accuracy"
            ],
        reverse=True,
    )

    print()
    print("=" * 96)
    print("M11B FINAL COMPARISON")
    print("=" * 96)

    print(
        f"{'Expert':<12}"
        f"{'Affinity':>11}"
        f"{'Rounds':>9}"
        f"{'Global':>11}"
        f"{'Gain':>10}"
        f"{'D0/D1':>11}"
        f"{'TargetGain':>12}"
        f"{'D2/D3':>11}"
        f"{'Retention':>12}"
    )

    print("-" * 96)

    for result in results:
        label = (
            f"L{result['layer']}"
            f"-E{result['expert']}"
        )

        print(
            f"{label:<12}"
            f"{result['shared_affinity'] * 100:>10.2f}%"
            f"{result['accepted_rounds']:>9}"
            f"{result['final_global_accuracy'] * 100:>10.2f}%"
            f"{result['global_gain'] * 100:>+9.2f}pp"
            f"{result['final_target_accuracy'] * 100:>10.2f}%"
            f"{result['target_gain'] * 100:>+11.2f}pp"
            f"{result['final_retention_accuracy'] * 100:>10.2f}%"
            f"{result['retention_change'] * 100:>+11.2f}pp"
        )

    scheduler_pick = (
        results[0]
    )

    validation_best = (
        by_global[0]
    )

    scheduler_correct = (
        scheduler_pick["layer"]
        == validation_best["layer"]
        and
        scheduler_pick["expert"]
        == validation_best["expert"]
    )

    print()
    print("Scheduler selected:")

    print(
        f"L{scheduler_pick['layer']}"
        f"-E{scheduler_pick['expert']}"
    )

    print(
        "Validation-best candidate:"
    )

    print(
        f"L{validation_best['layer']}"
        f"-E{validation_best['expert']}"
    )

    print(
        "Scheduler prediction correct:",
        scheduler_correct,
    )

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
            {
                "scheduler_method":
                    (
                        "max shared minimum "
                        "train-routing affinity"
                    ),

                "test_used":
                    False,

                "results":
                    results,

                "scheduler_selected":
                    {
                        "layer":
                            scheduler_pick[
                                "layer"
                            ],

                        "expert":
                            scheduler_pick[
                                "expert"
                            ],
                    },

                "validation_best":
                    {
                        "layer":
                            validation_best[
                                "layer"
                            ],

                        "expert":
                            validation_best[
                                "expert"
                            ],
                    },

                "scheduler_prediction_correct":
                    scheduler_correct,
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
