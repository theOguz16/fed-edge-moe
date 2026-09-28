import math
import json
from pathlib import Path


NETWORK_BUDGET_MB = 5.0

FEDERATED = {
    "steps": 120,
    "seconds": 7.712,
    "traffic_bytes": 787_000,
}

SPLIT = {
    "seconds_per_step": 1.940,

    "expert_download_bytes": 393_472,
    "expert_upload_bytes": 393_472,

    "activation_bytes_per_step": (
        102_480
        + 102_480
        + 102_480
    ),
}

OUTPUT = Path(
    "results/m13k_measured_hybrid_scheduler.json"
)


def split_traffic(steps):
    return (
        SPLIT["expert_download_bytes"]
        + steps
        * SPLIT["activation_bytes_per_step"]
        + SPLIT["expert_upload_bytes"]
    )


def main():
    budget_bytes = (
        NETWORK_BUDGET_MB
        * 1_000_000
    )

    remaining = (
        budget_bytes
        - SPLIT["expert_download_bytes"]
        - SPLIT["expert_upload_bytes"]
    )

    max_split_steps = max(
        0,
        math.floor(
            remaining
            / SPLIT["activation_bytes_per_step"]
        ),
    )

    max_split_seconds = (
        max_split_steps
        * SPLIT["seconds_per_step"]
    )

    print()
    print("FedEdgeMoE - M13K")
    print("Measured Hybrid Scheduler")
    print()

    print(
        "Network budget:",
        f"{NETWORK_BUDGET_MB:.1f} MB",
    )

    print()

    print(
        "Federated local:",
        f"{FEDERATED['steps']} steps",
        f"| {FEDERATED['seconds']:.3f}s",
        f"| {FEDERATED['traffic_bytes']/1e6:.3f} MB",
    )

    print(
        "Split physical:",
        "1 step",
        f"| {SPLIT['seconds_per_step']:.3f}s",
        f"| {split_traffic(1)/1e6:.3f} MB",
    )

    print()

    print(
        "Max split steps under budget:",
        max_split_steps,
    )

    print(
        "Estimated split time at limit:",
        f"{max_split_seconds:.3f}s",
    )

    print()

    print(
        "POLICY:"
    )

    print(
        "long local training -> FEDERATED_LOCAL"
    )

    print(
        "short remote execution -> SPLIT"
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
                "network_budget_mb":
                    NETWORK_BUDGET_MB,

                "federated":
                    FEDERATED,

                "split":
                    SPLIT,

                "max_split_steps":
                    max_split_steps,

                "max_split_seconds":
                    max_split_seconds,
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
