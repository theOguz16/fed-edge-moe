import json
from pathlib import Path


OUTPUT = Path(
    "results/m14c_affinity_speed_scheduler.json"
)

CANDIDATES = [
    {
        "device_id": "mac_m4",
        "expert": "L0-E7",
        "affinity": 0.214,
        "training_seconds_120": 3.462,
    },
    {
        "device_id": "msi_rtx3050",
        "expert": "L0-E7",
        "affinity": 0.191,
        "training_seconds_120": 3.458,
    },
]


def main():
    print()
    print("FedEdgeMoE - M14C")
    print("Affinity + Speed Scheduler")
    print()

    scored = []

    for device in CANDIDATES:
        score = (
            device["affinity"]
            / device["training_seconds_120"]
        )

        row = {
            **device,
            "score": score,
        }

        scored.append(row)

        print(
            f"{device['device_id']:<15} "
            f"| affinity={device['affinity']*100:4.1f}% "
            f"| train={device['training_seconds_120']:.3f}s "
            f"| score={score:.4f}"
        )

    scored.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    winner = scored[0]

    print()
    print(
        "PREFERRED DEVICE:",
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
                "ranking": scored,
                "preferred_device":
                    winner["device_id"],
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
