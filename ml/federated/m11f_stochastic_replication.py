import json
from pathlib import Path

import ml.federated.m11e1_prepare_replication_base as prep
import ml.federated.m11e2_independent_scheduler_test as test


NEW_BASE_SEEDS = [
    20260930,
    20261002,
]

ROOT = Path(
    "checkpoints/m11f"
)

RESULT_ROOT = Path(
    "results/m11f"
)

SUMMARY_PATH = Path(
    "results/m11f_stochastic_summary.json"
)

EXISTING_REPORT = Path(
    "results/m11e2_independent_scheduler_test.json"
)


def extract_summary(
    report,
    label,
    base_seed,
    fl_seed,
):
    results = {
        item["label"]: item
        for item in report["results"]
    }

    original = results["original"]
    balanced = results["balanced"]

    return {
        "label":
            label,

        "base_seed":
            base_seed,

        "fl_seed":
            fl_seed,

        "original_pick":
            report["original_pick"],

        "balanced_pick":
            report["balanced_pick"],

        "same_selection":
            report["same_selection"],

        "comparison":
            report["comparison"],

        "original_final_global":
            original["final_global"],

        "balanced_final_global":
            balanced["final_global"],

        "global_difference":
            (
                balanced["final_global"]
                - original["final_global"]
            ),

        "original_target_gain":
            original["final_target_gain"],

        "balanced_target_gain":
            balanced["final_target_gain"],

        "target_gain_difference":
            (
                balanced["final_target_gain"]
                - original["final_target_gain"]
            ),

        "original_retention_change":
            original[
                "final_retention_change"
            ],

        "balanced_retention_change":
            balanced[
                "final_retention_change"
            ],

        "retention_difference":
            (
                balanced[
                    "final_retention_change"
                ]
                - original[
                    "final_retention_change"
                ]
            ),
    }


def main():
    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    replications = []

    if EXISTING_REPORT.exists():
        with open(
            EXISTING_REPORT,
            "r",
            encoding="utf-8",
        ) as f:
            existing = json.load(f)

        replications.append(
            extract_summary(
                existing,
                label="existing_M11E2",
                base_seed=20260928,
                fl_seed=20260929,
            )
        )

    for base_seed in NEW_BASE_SEEDS:
        fl_seed = base_seed + 1

        label = (
            f"seed_{base_seed}"
        )

        print()
        print("#" * 96)
        print(
            "STOCHASTIC REPLICATION:",
            label,
        )
        print("#" * 96)

        base_root = (
            ROOT
            / label
            / "base"
        )

        base_report = (
            RESULT_ROOT
            / f"{label}_base.json"
        )

        prep.SEED = (
            base_seed
        )

        prep.OUTPUT_ROOT = (
            base_root
        )

        prep.REPORT_PATH = (
            base_report
        )

        prep.FEDERATED_BASE_STEP = (
            100
        )

        prep.main()

        base_checkpoint = (
            base_root
            / "global_step_0100"
        )

        scheduler_root = (
            ROOT
            / label
            / "scheduler_test"
        )

        scheduler_report = (
            RESULT_ROOT
            / f"{label}_scheduler.json"
        )

        test.SEED = (
            fl_seed
        )

        test.BASE_CHECKPOINT = (
            str(
                base_checkpoint
            )
        )

        test.OUTPUT_ROOT = (
            scheduler_root
        )

        test.REPORT_PATH = (
            scheduler_report
        )

        test.ROUNDS = 5

        test.main()

        with open(
            scheduler_report,
            "r",
            encoding="utf-8",
        ) as f:
            report = json.load(f)

        replications.append(
            extract_summary(
                report,
                label=label,
                base_seed=base_seed,
                fl_seed=fl_seed,
            )
        )

    balanced_higher = sum(
        item["comparison"]
        == "balanced_higher"
        for item in replications
    )

    original_higher = sum(
        item["comparison"]
        == "original_higher"
        for item in replications
    )

    same_selection = sum(
        item["comparison"]
        == "same_expert_selected"
        for item in replications
    )

    ties = sum(
        item["comparison"]
        == "tie"
        for item in replications
    )

    mean_global_difference = (
        sum(
            item[
                "global_difference"
            ]
            for item in replications
        )
        / len(replications)
    )

    mean_target_difference = (
        sum(
            item[
                "target_gain_difference"
            ]
            for item in replications
        )
        / len(replications)
    )

    mean_retention_difference = (
        sum(
            item[
                "retention_difference"
            ]
            for item in replications
        )
        / len(replications)
    )

    print()
    print("=" * 104)
    print(
        "M11F STOCHASTIC REPLICATION SUMMARY"
    )
    print("=" * 104)

    for item in replications:
        op = item["original_pick"]
        bp = item["balanced_pick"]

        print(
            f"{item['label']:<20} "
            f"original=L{op['layer']}-"
            f"E{op['expert']} "
            f"balanced=L{bp['layer']}-"
            f"E{bp['expert']} "
            f"| result="
            f"{item['comparison']:<20} "
            f"| global_diff="
            f"{item['global_difference'] * 100:+.2f}pp "
            f"| target_diff="
            f"{item['target_gain_difference'] * 100:+.2f}pp "
            f"| retention_diff="
            f"{item['retention_difference'] * 100:+.2f}pp"
        )

    print()
    print(
        "Balanced higher:",
        balanced_higher,
    )

    print(
        "Original higher:",
        original_higher,
    )

    print(
        "Same expert:",
        same_selection,
    )

    print(
        "Tie:",
        ties,
    )

    print()
    print(
        "Mean balanced-original "
        "global difference:",
        f"{mean_global_difference * 100:+.3f} pp",
    )

    print(
        "Mean target-gain difference:",
        f"{mean_target_difference * 100:+.3f} pp",
    )

    print(
        "Mean retention difference:",
        f"{mean_retention_difference * 100:+.3f} pp",
    )

    summary = {
        "experiment":
            "M11F stochastic replication",

        "synthetic_version":
            "V3",

        "scheduler_formulas_locked":
            True,

        "federated_base_step":
            100,

        "fixed_horizon_rounds":
            5,

        "test_used":
            False,

        "num_replications":
            len(replications),

        "balanced_higher":
            balanced_higher,

        "original_higher":
            original_higher,

        "same_selection":
            same_selection,

        "tie":
            ties,

        "mean_global_difference":
            mean_global_difference,

        "mean_target_gain_difference":
            mean_target_difference,

        "mean_retention_difference":
            mean_retention_difference,

        "replications":
            replications,
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "Summary:",
        SUMMARY_PATH,
    )


if __name__ == "__main__":
    main()
