import json
from pathlib import Path


INPUT_REPORT = Path(
    "results/m11a_affinity_scheduler.json"
)

BENCHMARK_REPORT = Path(
    "results/m11b_scheduler_benchmark.json"
)

OUTPUT_REPORT = Path(
    "results/m11d_balanced_affinity_scheduler.json"
)


def main():
    print()
    print("FedEdgeMoE - M11D")
    print("Balanced Shared-Affinity Scheduler")
    print()

    with open(
        INPUT_REPORT,
        "r",
        encoding="utf-8",
    ) as f:
        original = json.load(f)

    candidates = []

    for item in original["ranking"]:
        a = item[
            "client_A_affinity"
        ]

        b = item[
            "client_B_affinity"
        ]

        minimum = min(a, b)
        maximum = max(a, b)

        if maximum > 0:
            balance_ratio = (
                minimum / maximum
            )
        else:
            balance_ratio = 0.0

        balanced_score = (
            minimum
            * balance_ratio
        )

        candidate = dict(item)

        candidate[
            "balance_ratio"
        ] = balance_ratio

        candidate[
            "balanced_score"
        ] = balanced_score

        candidates.append(
            candidate
        )

    candidates.sort(
        key=lambda item: (
            item["balanced_score"],
            item["shared_min_affinity"],
            item["shared_mean_affinity"],
        ),
        reverse=True,
    )

    print(
        "Score = shared_min_affinity"
        " × balance_ratio"
    )

    print()

    print("=" * 96)
    print("BALANCED AFFINITY RANKING")
    print("=" * 96)

    for rank, item in enumerate(
        candidates[:12],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"L{item['layer']}-"
            f"E{item['expert']} "
            f"| A="
            f"{item['client_A_affinity'] * 100:5.1f}% "
            f"B="
            f"{item['client_B_affinity'] * 100:5.1f}% "
            f"| min="
            f"{item['shared_min_affinity'] * 100:5.1f}% "
            f"| balance="
            f"{item['balance_ratio'] * 100:5.1f}% "
            f"| score="
            f"{item['balanced_score'] * 100:6.2f}"
        )

    selected = (
        candidates[0]
    )

    old_selected = (
        original["selected"]
    )

    print()
    print("=" * 96)
    print("M11D DECISION")
    print("=" * 96)

    print(
        "Old scheduler:",
        f"L{old_selected['layer']}-"
        f"E{old_selected['expert']}",
    )

    print(
        "Balanced scheduler:",
        f"L{selected['layer']}-"
        f"E{selected['expert']}",
    )

    print(
        "Balanced score:",
        f"{selected['balanced_score'] * 100:.2f}",
    )

    validation_best = None

    if BENCHMARK_REPORT.exists():
        with open(
            BENCHMARK_REPORT,
            "r",
            encoding="utf-8",
        ) as f:
            benchmark = json.load(f)

        validation_best = (
            benchmark[
                "validation_best"
            ]
        )

        print()
        print(
            "Previously observed "
            "validation-best:",
            f"L{validation_best['layer']}-"
            f"E{validation_best['expert']}",
        )

        matches = (
            selected["layer"]
            == validation_best["layer"]
            and
            selected["expert"]
            == validation_best["expert"]
        )

        print(
            "Diagnostic match:",
            matches,
        )

        print()
        print(
            "WARNING:"
        )

        print(
            "This is a post-hoc diagnostic."
        )

        print(
            "It is NOT independent "
            "validation of the new score."
        )

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "method":
                    (
                        "balanced shared "
                        "train-routing affinity"
                    ),

                "formula":
                    (
                        "min(A,B) * "
                        "(min(A,B)/max(A,B))"
                    ),

                "validation_used_for_score":
                    False,

                "test_used":
                    False,

                "post_hoc_design":
                    True,

                "old_selected":
                    old_selected,

                "selected":
                    selected,

                "previous_validation_best":
                    validation_best,

                "ranking":
                    candidates,
            },
            f,
            indent=2,
        )

    print()
    print(
        "Report:",
        OUTPUT_REPORT,
    )


if __name__ == "__main__":
    main()
