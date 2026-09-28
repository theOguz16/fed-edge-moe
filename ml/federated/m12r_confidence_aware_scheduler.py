import json
from pathlib import Path


INPUT = Path(
    "results/m12q_cost_registry.json"
)

OUTPUT = Path(
    "results/m12r_confidence_aware_scheduler.json"
)


def main():
    with open(
        INPUT,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    devices = report["devices"]

    devices.sort(
        key=lambda x: x["efficiency"],
        reverse=True,
    )

    winner = devices[0]
    runner_up = devices[1]

    if (
        winner["cost_source"] == "measured"
    ):
        status = "CONFIRMED"

    elif (
        winner["cost_source"] == "estimated"
        and runner_up["cost_source"]
        == "measured"
    ):
        status = "PROVISIONAL"

    else:
        status = "ESTIMATED_ONLY"

    print()
    print("FedEdgeMoE - M12R")
    print("Confidence-Aware Scheduler")
    print()

    for device in devices:
        print(
            f"{device['name']:<10} "
            f"| efficiency="
            f"{device['efficiency']:.4f} "
            f"| source="
            f"{device['cost_source']}"
        )

    print()
    print(
        "SELECTED DEVICE:",
        winner["name"],
    )

    print(
        "DECISION STATUS:",
        status,
    )

    if status == "PROVISIONAL":
        print(
            "ACTION: physical measurement required"
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
            {
                "selected_device":
                    winner["name"],
                "decision_status":
                    status,
                "winner_cost_source":
                    winner["cost_source"],
                "runner_up":
                    runner_up["name"],
                "runner_up_cost_source":
                    runner_up["cost_source"],
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
