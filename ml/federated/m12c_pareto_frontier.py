import json
from pathlib import Path


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
    "results/m12c_pareto_frontier.json"
)


def dominates(a, b):
    return (
        a["target_gain"]
        >= b["target_gain"]
        and
        a["retention_change"]
        >= b["retention_change"]
        and
        (
            a["target_gain"]
            > b["target_gain"]
            or
            a["retention_change"]
            > b["retention_change"]
        )
    )


def pareto_frontier(items):
    frontier = []

    for item in items:
        dominated = False

        for other in items:
            if other is item:
                continue

            if dominates(
                other,
                item,
            ):
                dominated = True
                break

        if not dominated:
            frontier.append(item)

    return frontier


def main():
    print()
    print("FedEdgeMoE - M12C")
    print("Pareto Frontier Analysis")

    with open(
        M11B,
        "r",
        encoding="utf-8",
    ) as f:
        m11b = json.load(f)

    experts = []

    for item in m11b["results"]:
        experts.append({
            "name":
                f"L{item['layer']}-E{item['expert']}",

            "target_gain":
                item["target_gain"],

            "retention_change":
                item["retention_change"],
        })

    with open(
        M11C,
        "r",
        encoding="utf-8",
    ) as f:
        m11c = json.load(f)

    last = m11c["history"][-1]

    for item in experts:
        if item["name"] == "L0-E3":
            item["target_gain"] = (
                last["target_gain"]
            )

            item["retention_change"] = (
                last["retention_change"]
            )

    frontier = pareto_frontier(
        experts
    )

    print()
    print("=" * 72)
    print("M11 EXPERT OPTIONS")
    print("=" * 72)

    for item in experts:
        status = (
            "PARETO"
            if item in frontier
            else "DOMINATED"
        )

        print(
            f"{item['name']:<8} "
            f"| target="
            f"{item['target_gain'] * 100:+.2f}pp "
            f"| retention="
            f"{item['retention_change'] * 100:+.2f}pp "
            f"| {status}"
        )

    with open(
        M11F,
        "r",
        encoding="utf-8",
    ) as f:
        m11f = json.load(f)

    print()
    print("=" * 72)
    print("M11F SCHEDULER OPTIONS")
    print("=" * 72)

    scheduler_cases = []

    for rep in m11f["replications"]:
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

        local_frontier = (
            pareto_frontier(options)
        )

        print()
        print(rep["label"])

        for item in options:
            status = (
                "PARETO"
                if item
                in local_frontier
                else "DOMINATED"
            )

            print(
                f"  {item['name']:<8} "
                f"| target="
                f"{item['target_gain'] * 100:+.2f}pp "
                f"| retention="
                f"{item['retention_change'] * 100:+.2f}pp "
                f"| {status}"
            )

        scheduler_cases.append({
            "label":
                rep["label"],

            "options":
                options,

            "frontier":
                local_frontier,
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
                "expert_options":
                    experts,

                "expert_frontier":
                    frontier,

                "scheduler_cases":
                    scheduler_cases,
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
