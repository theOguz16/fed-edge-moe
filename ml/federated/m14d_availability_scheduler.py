import json
from pathlib import Path


OUTPUT = Path(
    "results/m14d_availability_scheduler.json"
)

DEVICES = [
    {
        "device_id": "mac_m4",
        "affinity": 0.214,
        "training_seconds": 3.462,
        "available": False,
    },
    {
        "device_id": "msi_rtx3050",
        "affinity": 0.191,
        "training_seconds": 3.458,
        "available": True,
    },
]


def main():
    print()
    print("FedEdgeMoE - M14D")
    print("Availability-Aware Scheduler")
    print()

    eligible = []

    for device in DEVICES:
        if not device["available"]:
            print(
                f"{device['device_id']:<15} "
                "| OFFLINE/BUSY -> PRUNE"
            )
            continue

        score = (
            device["affinity"]
            / device["training_seconds"]
        )

        eligible.append({
            **device,
            "score": score,
        })

        print(
            f"{device['device_id']:<15} "
            f"| AVAILABLE "
            f"| score={score:.4f}"
        )

    if not eligible:
        print()
        print("NO DEVICE AVAILABLE")
        return

    eligible.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    winner = eligible[0]

    print()
    print(
        "SELECTED DEVICE:",
        winner["device_id"],
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
                "expert": "L0-E7",
                "selected_device":
                    winner["device_id"],
                "eligible_devices":
                    eligible,
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
