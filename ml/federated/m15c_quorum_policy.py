import json
from pathlib import Path


MIN_QUORUM = 2

SCENARIOS = [
    {
        "name": "healthy_partial",
        "valid_clients": 3,
    },
    {
        "name": "minimum_valid",
        "valid_clients": 2,
    },
    {
        "name": "too_few",
        "valid_clients": 1,
    },
    {
        "name": "none",
        "valid_clients": 0,
    },
]

OUTPUT = Path(
    "results/m15c_quorum_policy.json"
)


def main():
    print()
    print("FedEdgeMoE - M15C")
    print("Minimum Quorum Policy")
    print()

    results = []

    for scenario in SCENARIOS:
        valid = scenario[
            "valid_clients"
        ]

        commit = (
            valid >= MIN_QUORUM
        )

        status = (
            "COMMIT"
            if commit
            else "ABORT"
        )

        print(
            f"{scenario['name']:<16} "
            f"| valid={valid} "
            f"| {status}"
        )

        results.append({
            **scenario,
            "decision":
                status,
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
                "min_quorum":
                    MIN_QUORUM,
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
