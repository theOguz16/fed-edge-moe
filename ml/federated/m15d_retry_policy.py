import json
from pathlib import Path


MIN_QUORUM = 2
MAX_ATTEMPTS = 3

ATTEMPTS = [
    {
        "attempt": 1,
        "clients": {
            "A": "dropout",
            "B": "complete",
            "C": "dropout",
            "D": "stale",
        },
    },
    {
        "attempt": 2,
        "clients": {
            "A": "complete",
            "B": "complete",
            "C": "dropout",
            "D": "stale",
        },
    },
]

OUTPUT = Path(
    "results/m15d_retry_policy.json"
)


def valid(status):
    return status == "complete"


def main():
    print()
    print("FedEdgeMoE - M15D")
    print("Quorum Retry Policy")
    print()

    history = []
    final = "FAILED"

    for attempt in ATTEMPTS:
        valid_clients = [
            client
            for client, status
            in attempt["clients"].items()
            if valid(status)
        ]

        count = len(valid_clients)

        if count >= MIN_QUORUM:
            decision = "COMMIT"
            final = "SUCCESS"
        else:
            decision = "RETRY"

        print(
            f"attempt={attempt['attempt']} "
            f"| valid={count} "
            f"| clients={','.join(valid_clients) or '-'} "
            f"| {decision}"
        )

        history.append({
            "attempt":
                attempt["attempt"],
            "valid_clients":
                valid_clients,
            "decision":
                decision,
        })

        if decision == "COMMIT":
            break

        if attempt["attempt"] >= MAX_ATTEMPTS:
            final = "FAILED"
            break

    print()
    print(
        "FINAL ROUND STATUS:",
        final,
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
                "min_quorum":
                    MIN_QUORUM,
                "max_attempts":
                    MAX_ATTEMPTS,
                "history":
                    history,
                "final_status":
                    final,
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
