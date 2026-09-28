import json
from pathlib import Path


GLOBAL_VERSION = 105
MIN_QUORUM = 2
MAX_ATTEMPTS = 3

ATTEMPTS = [
    [
        {
            "client_id": "B",
            "base_version": 105,
            "status": "complete",
            "weight": 120,
        },
        {
            "client_id": "B",
            "base_version": 105,
            "status": "complete",
            "weight": 120,
        },
        {
            "client_id": "C",
            "base_version": 104,
            "status": "complete",
            "weight": 120,
        },
        {
            "client_id": "A",
            "base_version": 105,
            "status": "dropout",
            "weight": 0,
        },
    ],

    [
        {
            "client_id": "A",
            "base_version": 105,
            "status": "complete",
            "weight": 120,
        },
        {
            "client_id": "B",
            "base_version": 105,
            "status": "complete",
            "weight": 120,
        },
    ],
]

OUTPUT = Path(
    "results/m15f_round_controller.json"
)


def process_attempt(updates):
    seen = set()
    accepted = []
    rejected = []

    for update in updates:
        client = update["client_id"]

        if update["status"] == "dropout":
            reason = "DROPOUT"

        elif (
            update["base_version"]
            != GLOBAL_VERSION
        ):
            reason = "STALE_UPDATE"

        elif client in seen:
            reason = "DUPLICATE_UPDATE"

        else:
            reason = None

        if reason:
            rejected.append({
                **update,
                "reason": reason,
            })
            continue

        seen.add(client)
        accepted.append(update)

    return accepted, rejected


def main():
    print()
    print("FedEdgeMoE - M15F")
    print("Unified Robust Round Controller")
    print()

    history = []
    committed = False

    for attempt_id, updates in enumerate(
        ATTEMPTS,
        start=1,
    ):
        accepted, rejected = (
            process_attempt(updates)
        )

        valid_count = len(accepted)

        decision = (
            "COMMIT"
            if valid_count >= MIN_QUORUM
            else "RETRY"
        )

        print(
            f"Attempt {attempt_id}: "
            f"accepted={valid_count} "
            f"| rejected={len(rejected)} "
            f"| {decision}"
        )

        for item in rejected:
            print(
                f"  reject "
                f"{item['client_id']}: "
                f"{item['reason']}"
            )

        history.append({
            "attempt":
                attempt_id,
            "accepted":
                [
                    x["client_id"]
                    for x in accepted
                ],
            "rejected":
                rejected,
            "decision":
                decision,
        })

        if decision == "COMMIT":
            committed = True

            total_weight = sum(
                x["weight"]
                for x in accepted
            )

            new_version = (
                GLOBAL_VERSION + 1
            )

            print()
            print(
                "Committed clients:",
                ", ".join(
                    x["client_id"]
                    for x in accepted
                ),
            )

            print(
                "Aggregation weight:",
                total_weight,
            )

            print(
                "Global version:",
                f"V{GLOBAL_VERSION} -> "
                f"V{new_version}",
            )

            break

        if attempt_id >= MAX_ATTEMPTS:
            break

    final_status = (
        "COMMITTED"
        if committed
        else "ABORTED"
    )

    print()
    print(
        "FINAL STATUS:",
        final_status,
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
                "min_quorum":
                    MIN_QUORUM,
                "history":
                    history,
                "final_status":
                    final_status,
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
