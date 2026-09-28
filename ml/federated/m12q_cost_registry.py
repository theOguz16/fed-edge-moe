import json
from pathlib import Path


OUTPUT = Path(
    "results/m12q_cost_registry.json"
)

DEVICES = [
    {
        "name": "MSI",
        "affinity": 0.20,
        "estimated_seconds": 4.06,
        "measured_e2e_seconds": 7.712,
    },
    {
        "name": "slow-edge",
        "affinity": 0.40,
        "estimated_seconds": 11.24,
        "measured_e2e_seconds": None,
    },
]


def effective_cost(device):
    measured = device[
        "measured_e2e_seconds"
    ]

    if measured is not None:
        return measured, "measured"

    return (
        device["estimated_seconds"],
        "estimated",
    )


def main():
    print()
    print("FedEdgeMoE - M12Q")
    print("Measured-First Cost Registry")
    print()

    scored = []

    for device in DEVICES:
        seconds, source = effective_cost(
            device
        )

        efficiency = (
            device["affinity"]
            / seconds
        )

        row = {
            **device,
            "effective_seconds":
                seconds,
            "cost_source":
                source,
            "efficiency":
                efficiency,
        }

        scored.append(row)

        print(
            f"{device['name']:<10} "
            f"| cost={seconds:6.3f}s "
            f"| source={source:<9} "
            f"| affinity="
            f"{device['affinity']*100:4.1f}% "
            f"| efficiency="
            f"{efficiency:.4f}"
        )

    scored.sort(
        key=lambda x: x["efficiency"],
        reverse=True,
    )

    winner = scored[0]

    print()
    print(
        "SELECTED DEVICE:",
        winner["name"],
    )

    print(
        "COST SOURCE:",
        winner["cost_source"],
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
                "policy":
                    "measured_first",
                "devices":
                    scored,
                "selected_device":
                    winner["name"],
                "selected_cost_source":
                    winner["cost_source"],
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
