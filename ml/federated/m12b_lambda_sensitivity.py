import json
from pathlib import Path


LAMBDAS = [
    0.0,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
]

M11B = Path(
    "results/m11b_scheduler_benchmark.json"
)

M11C = Path(
    "results/m11c_e3_fixed_horizon.json"
)

M11F = Path(
    "results/m11f_stochastic_summary.json"
)

OUTPUT = Path(
    "results/m12b_lambda_sensitivity.json"
)


def utility(
    target_gain,
    retention_change,
    lam,
):
    forgetting = max(
        0.0,
        -retention_change,
    )

    return (
        target_gain
        - lam * forgetting
    )


def main():
    print()
    print("FedEdgeMoE - M12B")
    print("Lambda Sensitivity")
    print()

    with open(
        M11B,
        "r",
        encoding="utf-8",
    ) as f:
        m11b = json.load(f)

    candidates = []

    for item in m11b["results"]:
        candidates.append({
            "expert":
                f"L{item['layer']}-E{item['expert']}",

            "target_gain":
                item["target_gain"],

            "retention_change":
                item["retention_change"],
        })

    # M11B'deki E3 early-stop sonucu yerine
    # fixed-horizon M11C sonucu kullan.
    with open(
        M11C,
        "r",
        encoding="utf-8",
    ) as f:
        m11c = json.load(f)

    for candidate in candidates:
        if candidate["expert"] == "L0-E3":
            last = (
                m11c["history"][-1]
            )

            candidate["target_gain"] = (
                last["target_gain"]
            )

            candidate[
                "retention_change"
            ] = (
                last["retention_change"]
            )

    print("=" * 76)
    print("M11 EXPERT RANKING VS LAMBDA")
    print("=" * 76)

    expert_results = []

    for lam in LAMBDAS:
        scored = []

        for candidate in candidates:
            score = utility(
                candidate["target_gain"],
                candidate[
                    "retention_change"
                ],
                lam,
            )

            scored.append({
                **candidate,
                "utility":
                    score,
            })

        scored.sort(
            key=lambda x:
                x["utility"],
            reverse=True,
        )

        winner = scored[0]

        print(
            f"lambda={lam:>4.1f} "
            f"→ winner="
            f"{winner['expert']} "
            f"| utility="
            f"{winner['utility'] * 100:+.2f}"
        )

        expert_results.append({
            "lambda":
                lam,

            "winner":
                winner["expert"],

            "ranking":
                scored,
        })

    with open(
        M11F,
        "r",
        encoding="utf-8",
    ) as f:
        m11f = json.load(f)

    print()
    print("=" * 76)
    print("M11F ORIGINAL VS BALANCED")
    print("=" * 76)

    replication_results = []

    for lam in LAMBDAS:
        original_wins = 0
        balanced_wins = 0
        ties = 0

        for rep in (
            m11f["replications"]
        ):
            original = utility(
                rep[
                    "original_target_gain"
                ],
                rep[
                    "original_retention_change"
                ],
                lam,
            )

            balanced = utility(
                rep[
                    "balanced_target_gain"
                ],
                rep[
                    "balanced_retention_change"
                ],
                lam,
            )

            if original > balanced:
                original_wins += 1

            elif balanced > original:
                balanced_wins += 1

            else:
                ties += 1

        print(
            f"lambda={lam:>4.1f} "
            f"→ original={original_wins} "
            f"balanced={balanced_wins} "
            f"tie={ties}"
        )

        replication_results.append({
            "lambda":
                lam,

            "original_wins":
                original_wins,

            "balanced_wins":
                balanced_wins,

            "ties":
                ties,
        })

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
            {
                "lambdas":
                    LAMBDAS,

                "expert_results":
                    expert_results,

                "replication_results":
                    replication_results,
            },
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
