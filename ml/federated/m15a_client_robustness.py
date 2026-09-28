import json
from pathlib import Path


GLOBAL_VERSION = 105

CLIENTS = [
    {
        "id": "client_A",
        "base_version": 105,
        "status": "complete",
        "weight": 120,
    },
    {
        "id": "client_B",
        "base_version": 105,
        "status": "complete",
        "weight": 120,
    },
    {
        "id": "client_C",
        "base_version": 105,
        "status": "dropout",
        "weight": 0,
    },
    {
        "id": "client_D",
        "base_version": 104,
        "status": "complete",
        "weight": 120,
    },
]

OUTPUT = Path(
    "results/m15a_client_robustness.json"
)


def main():
    accepted = []
    rejected = []

    print()
    print("FedEdgeMoE - M15A")
    print("Dropout + Staleness Robustness")
    print()

    for client in CLIENTS:
        if client["status"] == "dropout":
            reason = "DROPOUT"

        elif (
            client["base_version"]
            != GLOBAL_VERSION
        ):
            reason = "STALE_UPDATE"

        else:
            reason = None

        if reason is None:
            accepted.append(client)

            print(
                f"{client['id']:<10} "
                "| ACCEPT"
            )

        else:
            rejected.append({
                **client,
                "reason": reason,
            })

            print(
                f"{client['id']:<10} "
                f"| REJECT | {reason}"
            )

    total_weight = sum(
        c["weight"]
        for c in accepted
    )

    participation = (
        len(accepted)
        / len(CLIENTS)
    )

    print()
    print(
        "Accepted:",
        len(accepted),
    )

    print(
        "Rejected:",
        len(rejected),
    )

    print(
        "Effective participation:",
        f"{participation * 100:.1f}%",
    )

    print(
        "Aggregation weight:",
        total_weight,
    )

    round_valid = (
        len(accepted) >= 2
    )

    print()
    print(
        "ROUND VALID:",
        round_valid,
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
                "global_version":
                    GLOBAL_VERSION,
                "accepted":
                    accepted,
                "rejected":
                    rejected,
                "participation":
                    participation,
                "round_valid":
                    round_valid,
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
