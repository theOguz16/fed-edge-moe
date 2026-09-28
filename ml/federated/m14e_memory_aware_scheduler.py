import json
from pathlib import Path


INPUT = Path(
    "results/m14a_device_profiles.json"
)

OUTPUT = Path(
    "results/m14e_memory_aware_scheduler.json"
)


TASKS = [
    {
        "name": "small_expert_job",
        "required_training_memory_gb": 1.5,
    },
    {
        "name": "medium_expert_job",
        "required_training_memory_gb": 3.5,
    },
    {
        "name": "large_expert_job",
        "required_training_memory_gb": 6.0,
    },
]


def available_training_memory(device):
    compute = device["compute"]

    if compute["backend"] == "cuda":
        return compute[
            "gpu_memory_gb"
        ]

    return compute.get(
        "memory_gb",
        0.0,
    )


def main():
    with open(
        INPUT,
        "r",
        encoding="utf-8",
    ) as f:
        profiles = json.load(f)

    results = []

    print()
    print("FedEdgeMoE - M14E")
    print("Memory-Aware Placement")
    print()

    for task in TASKS:
        required = task[
            "required_training_memory_gb"
        ]

        print(
            f"{task['name']} "
            f"| required={required:.1f} GB"
        )

        task_results = []

        for device in profiles["devices"]:
            memory = (
                available_training_memory(
                    device
                )
            )

            fl_capable = device[
                "capabilities"
            ][
                "federated_local_training"
            ]

            fits = (
                fl_capable
                and memory >= required
            )

            status = (
                "KEEP"
                if fits
                else "PRUNE"
            )

            print(
                f"  {device['device_id']:<15} "
                f"| memory={memory:4.1f} GB "
                f"| {status}"
            )

            task_results.append({
                "device_id":
                    device["device_id"],
                "available_memory_gb":
                    memory,
                "fits":
                    fits,
            })

        results.append({
            **task,
            "devices":
                task_results,
        })

        print()

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
                "results":
                    results,
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
