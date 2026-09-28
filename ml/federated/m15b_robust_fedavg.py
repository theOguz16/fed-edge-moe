import json
from pathlib import Path

import torch


GLOBAL_VERSION = 105

CLIENTS = [
    {
        "id": "client_A",
        "base_version": 105,
        "status": "complete",
        "weight": 120,
        "delta": torch.tensor(
            [1.0, 2.0, 3.0]
        ),
    },
    {
        "id": "client_B",
        "base_version": 105,
        "status": "complete",
        "weight": 240,
        "delta": torch.tensor(
            [3.0, 4.0, 5.0]
        ),
    },
    {
        "id": "client_C",
        "base_version": 105,
        "status": "dropout",
        "weight": 0,
        "delta": None,
    },
    {
        "id": "client_D",
        "base_version": 104,
        "status": "complete",
        "weight": 999,
        "delta": torch.tensor(
            [100.0, 100.0, 100.0]
        ),
    },
]

OUTPUT = Path(
    "results/m15b_robust_fedavg.json"
)


def valid(client):
    return (
        client["status"] == "complete"
        and client["base_version"]
        == GLOBAL_VERSION
        and client["delta"] is not None
    )


def main():
    accepted = [
        c
        for c in CLIENTS
        if valid(c)
    ]

    total_weight = sum(
        c["weight"]
        for c in accepted
    )

    aggregated = sum(
        c["delta"] * c["weight"]
        for c in accepted
    ) / total_weight

    expected = (
        CLIENTS[0]["delta"] * 120
        + CLIENTS[1]["delta"] * 240
    ) / 360

    max_diff = (
        aggregated
        - expected
    ).abs().max().item()

    stale_contaminated = (
        aggregated.max().item()
        > 10.0
    )

    print()
    print("FedEdgeMoE - M15B")
    print("Robust Partial FedAvg")
    print()

    print(
        "Accepted clients:",
        ", ".join(
            c["id"]
            for c in accepted
        ),
    )

    print(
        "Total weight:",
        total_weight,
    )

    print(
        "Aggregated delta:",
        aggregated.tolist(),
    )

    print(
        "Expected delta:",
        expected.tolist(),
    )

    print(
        "Max difference:",
        max_diff,
    )

    print(
        "Stale contamination:",
        stale_contaminated,
    )

    robust = (
        max_diff == 0.0
        and not stale_contaminated
    )

    print()
    print(
        "ROBUST FEDAVG OK:",
        robust,
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
                "accepted_clients": [
                    c["id"]
                    for c in accepted
                ],
                "total_weight":
                    total_weight,
                "aggregated_delta":
                    aggregated.tolist(),
                "max_difference":
                    max_diff,
                "stale_contamination":
                    stale_contaminated,
                "robust":
                    robust,
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
