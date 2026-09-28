import json
from pathlib import Path


BUDGETS_PP = [
    0.00,
    0.10,
    0.20,
    0.50,
    1.00,
]

INPUT = Path(
    "results/m11f_stochastic_summary.json"
)

OUTPUT = Path(
    "results/m12e_budget_replication.json"
)


def feasible(
    retention_change,
    budget_pp,
):
    forgetting_pp = max(
        0.0,
        -retention_change * 100,
    )

    return (
        forgetting_pp <= budget_pp
    )


def main():
    print()
    print("FedEdgeMoE - M12E")
    print("Retention Budget Replication")
    print()

    with open(
        INPUT,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    all_results = []

    for budget_pp in BUDGETS_PP:
        print("=" * 76)
        print(
            f"RETENTION BUDGET = "
            f"{budget_pp:.2f} pp"
        )
        print("=" * 76)

        budget_results = []

        for rep in data["replications"]:
            options = [
                {
                    "name":
                        "original",

                    "target_gain":
                        rep[
                            "original_target_gain"
                        ],

                    "retention_change":
                        rep[
                            "original_retention_change"
                        ],
                },
                {
                    "name":
                        "balanced",

                    "target_gain":
                        rep[
                            "balanced_target_gain"
                        ],

                    "retention_change":
                        rep[
                            "balanced_retention_change"
                        ],
                },
            ]

            valid = [
                item
                for item in options
                if feasible(
                    item[
                        "retention_change"
                    ],
                    budget_pp,
                )
            ]

            if valid:
                valid.sort(
                    key=lambda x:
                        x["target_gain"],
                    reverse=True,
                )

                winner = valid[0]

                decision = (
                    winner["name"]
                )

                print(
                    f"{rep['label']:<20} "
                    f"→ {decision:<8} "
                    f"| target="
                    f"{winner['target_gain'] * 100:+.2f}pp "
                    f"| retention="
                    f"{winner['retention_change'] * 100:+.2f}pp"
                )

            else:
                decision = None

                print(
                    f"{rep['label']:<20} "
                    f"→ NO FEASIBLE OPTION"
                )

            budget_results.append({
                "label":
                    rep["label"],

                "decision":
                    decision,
            })

        all_results.append({
            "budget_pp":
                budget_pp,

            "decisions":
                budget_results,
        })

        print()

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
                "policy":
                    (
                        "maximize target gain "
                        "subject to retention budget"
                    ),

                "results":
                    all_results,
            },
            f,
            indent=2,
        )

    print(
        "Report:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
