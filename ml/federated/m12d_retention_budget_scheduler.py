import json
from pathlib import Path


BUDGETS_PP = [
    0.00,
    0.10,
    0.20,
    0.50,
    1.00,
]

M11B = Path(
    "results/m11b_scheduler_benchmark.json"
)

M11C = Path(
    "results/m11c_e3_fixed_horizon.json"
)

OUTPUT = Path(
    "results/m12d_retention_budget_scheduler.json"
)


def main():
    print()
    print("FedEdgeMoE - M12D")
    print("Retention Budget Scheduler")
    print()

    with open(
        M11B,
        "r",
        encoding="utf-8",
    ) as f:
        m11b = json.load(f)

    options = []

    for item in m11b["results"]:
        options.append({
            "expert":
                f"L{item['layer']}-E{item['expert']}",

            "target_gain":
                item["target_gain"],

            "retention_change":
                item["retention_change"],
        })

    # E3 için early-stop yerine
    # fixed-horizon M11C sonucunu kullan.
    with open(
        M11C,
        "r",
        encoding="utf-8",
    ) as f:
        m11c = json.load(f)

    last = m11c["history"][-1]

    for option in options:
        if option["expert"] == "L0-E3":
            option["target_gain"] = (
                last["target_gain"]
            )

            option["retention_change"] = (
                last["retention_change"]
            )

    results = []

    print("=" * 76)
    print("RETENTION BUDGET DECISIONS")
    print("=" * 76)

    for budget_pp in BUDGETS_PP:
        budget = (
            budget_pp / 100.0
        )

        feasible = []

        for option in options:
            forgetting = max(
                0.0,
                -option[
                    "retention_change"
                ],
            )

            if forgetting <= budget:
                feasible.append(
                    option
                )

        feasible.sort(
            key=lambda x:
                x["target_gain"],
            reverse=True,
        )

        if feasible:
            winner = feasible[0]

            print(
                f"budget={budget_pp:>4.2f}pp "
                f"→ {winner['expert']} "
                f"| target="
                f"{winner['target_gain'] * 100:+.2f}pp "
                f"| retention="
                f"{winner['retention_change'] * 100:+.2f}pp"
            )

            winner_name = (
                winner["expert"]
            )

        else:
            print(
                f"budget={budget_pp:>4.2f}pp "
                f"→ no feasible expert"
            )

            winner_name = None

        results.append({
            "budget_pp":
                budget_pp,

            "winner":
                winner_name,

            "feasible":
                feasible,
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
                "policy":
                    (
                        "maximize target gain "
                        "subject to retention budget"
                    ),

                "budgets_pp":
                    BUDGETS_PP,

                "results":
                    results,
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
