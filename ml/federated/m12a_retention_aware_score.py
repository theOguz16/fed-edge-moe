import json
from pathlib import Path


LAMBDA_RETENTION = 2.0

M11B = Path(
    "results/m11b_scheduler_benchmark.json"
)

M11F = Path(
    "results/m11f_stochastic_summary.json"
)

OUTPUT = Path(
    "results/m12a_retention_aware_score.json"
)


def utility(
    target_gain,
    retention_change,
):
    forgetting = max(
        0.0,
        -retention_change,
    )

    return (
        target_gain
        - LAMBDA_RETENTION
        * forgetting
    )


def main():
    print()
    print("FedEdgeMoE - M12A")
    print("Retention-Aware Utility")
    print()
    print(
        "utility = target_gain "
        "- 2 * forgetting"
    )

    results = {}

    # -----------------------------
    # M11B candidate comparison
    # -----------------------------

    with open(
        M11B,
        "r",
        encoding="utf-8",
    ) as f:
        m11b = json.load(f)

    candidates = []

    for item in m11b["results"]:
        score = utility(
            item["target_gain"],
            item["retention_change"],
        )

        candidates.append({
            "expert":
                f"L{item['layer']}-E{item['expert']}",

            "target_gain":
                item["target_gain"],

            "retention_change":
                item["retention_change"],

            "utility":
                score,
        })

    candidates.sort(
        key=lambda x: x["utility"],
        reverse=True,
    )

    print()
    print("=" * 72)
    print("M11B RETENTION-AWARE RANKING")
    print("=" * 72)

    for rank, item in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{item['expert']} "
            f"| target="
            f"{item['target_gain'] * 100:+.2f}pp "
            f"| retention="
            f"{item['retention_change'] * 100:+.2f}pp "
            f"| utility="
            f"{item['utility'] * 100:+.2f}"
        )

    results["m11b"] = candidates

    # -----------------------------
    # M11F replication comparison
    # -----------------------------

    with open(
        M11F,
        "r",
        encoding="utf-8",
    ) as f:
        m11f = json.load(f)

    replications = []

    print()
    print("=" * 72)
    print("M11F RETENTION-AWARE COMPARISON")
    print("=" * 72)

    for rep in m11f["replications"]:
        original_score = utility(
            rep["original_target_gain"],
            rep["original_retention_change"],
        )

        balanced_score = utility(
            rep["balanced_target_gain"],
            rep["balanced_retention_change"],
        )

        if balanced_score > original_score:
            winner = "balanced"

        elif original_score > balanced_score:
            winner = "original"

        else:
            winner = "tie"

        print(
            f"{rep['label']:<20} "
            f"original={original_score * 100:+.2f} "
            f"balanced={balanced_score * 100:+.2f} "
            f"→ {winner}"
        )

        replications.append({
            "label":
                rep["label"],

            "original_utility":
                original_score,

            "balanced_utility":
                balanced_score,

            "winner":
                winner,
        })

    results["m11f"] = replications

    results["lambda_retention"] = (
        LAMBDA_RETENTION
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        "Report:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
