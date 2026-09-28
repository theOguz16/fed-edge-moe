import json
from pathlib import Path


ROUND = 7
GLOBAL_VERSION = 105

UPDATES = [
    {
        "client_id": "A",
        "round": 7,
        "base_version": 105,
        "weight": 120,
    },
    {
        "client_id": "B",
        "round": 7,
        "base_version": 105,
        "weight": 120,
    },
    {
        # Network retry / duplicate POST
        "client_id": "B",
        "round": 7,
        "base_version": 105,
        "weight": 120,
    },
    {
        "client_id": "C",
        "round": 7,
        "base_version": 104,
        "weight": 120,
    },
]

OUTPUT = Path(
    "results/m15e_duplicate_update_guard.json"
)


def main():
    seen = set()
    accepted = []
    rejected = []

    print()
    print("FedEdgeMoE - M15E")
    print("Duplicate Update Guard")
    print()

    for update in UPDATES:
        key = (
            update["round"],
            update["base_version"],
            update["client_id"],
        )

        if update["round"] != ROUND:
            reason = "WRONG_ROUND"

        elif (
            update["base_version"]
            != GLOBAL_VERSION
        ):
            reason = "STALE_UPDATE"

        elif key in seen:
            reason = "DUPLICATE_UPDATE"

        else:
            reason = None

        if reason is None:
            seen.add(key)
            accepted.append(update)

            print(
                f"{update['client_id']:<8} "
                "| ACCEPT"
            )

        else:
            rejected.append({
                **update,
                "reason": reason,
            })

            print(
                f"{update['client_id']:<8} "
                f"| REJECT | {reason}"
            )

    total_weight = sum(
        item["weight"]
        for item in accepted
    )

    safe = (
        [x["client_id"] for x in accepted]
        == ["A", "B"]
        and total_weight == 240
    )

    print()
    print(
        "Accepted clients:",
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
        "DUPLICATE GUARD OK:",
        safe,
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
                "round": ROUND,
                "global_version":
                    GLOBAL_VERSION,
                "accepted":
                    accepted,
                "rejected":
                    rejected,
                "aggregation_weight":
                    total_weight,
                "safe":
                    safe,
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
