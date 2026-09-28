import json
from pathlib import Path


INPUT = Path(
    "results/m14a_device_profiles.json"
)

OUTPUT = Path(
    "results/m14b_capability_filter.json"
)

TASK = {
    "task_id": "train_L0_E7",
    "target_expert": "L0-E7",
    "mode": "FEDERATED_LOCAL",
    "local_steps": 120,
}


def is_eligible(
    device,
    task,
):
    caps = device[
        "capabilities"
    ]

    if (
        task["mode"]
        == "FEDERATED_LOCAL"
    ):
        return caps[
            "federated_local_training"
        ]

    if (
        task["mode"]
        == "SPLIT"
    ):
        return caps[
            "split_execution"
        ]

    return False


def main():
    with open(
        INPUT,
        "r",
        encoding="utf-8",
    ) as f:
        profiles = json.load(f)

    eligible = []
    rejected = []

    print()
    print("FedEdgeMoE - M14B")
    print("Task-Aware Capability Filter")
    print()

    print(
        "Task:",
        TASK["task_id"],
    )

    print(
        "Mode:",
        TASK["mode"],
    )

    print()

    for device in profiles["devices"]:
        ok = is_eligible(
            device,
            TASK,
        )

        if ok:
            eligible.append(
                device["device_id"]
            )
            status = "KEEP"
        else:
            rejected.append(
                device["device_id"]
            )
            status = "PRUNE"

        print(
            f"{device['device_id']:<15} "
            f"| {status}"
        )

    print()
    print(
        "Eligible devices:",
        ", ".join(eligible),
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
                "task":
                    TASK,
                "eligible_devices":
                    eligible,
                "rejected_devices":
                    rejected,
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
