import math
import json
from pathlib import Path


NETWORK_BUDGET_MB = 5.0

FEDERATED_SECONDS = 7.712
FEDERATED_STEPS = 120
FEDERATED_TRAFFIC = 787_000

SPLIT_SECONDS_PER_STEP = 1.940

SPLIT_FIXED_BYTES = (
    393_472
    + 393_472
)

SPLIT_BYTES_PER_STEP = (
    102_480
    + 102_480
    + 102_480
)

OUTPUT = Path(
    "results/m13l_dual_budget_policy.json"
)


def main():
    network_budget = (
        NETWORK_BUDGET_MB
        * 1_000_000
    )

    max_by_network = math.floor(
        (
            network_budget
            - SPLIT_FIXED_BYTES
        )
        / SPLIT_BYTES_PER_STEP
    )

    max_by_latency = math.floor(
        FEDERATED_SECONDS
        / SPLIT_SECONDS_PER_STEP
    )

    safe_split_steps = min(
        max_by_network,
        max_by_latency,
    )

    limiting_factor = (
        "NETWORK"
        if max_by_network < max_by_latency
        else "LATENCY"
    )

    print()
    print("FedEdgeMoE - M13L")
    print("Dual-Budget Hybrid Policy")
    print()

    print(
        "Split max by network:",
        max_by_network,
        "steps",
    )

    print(
        "Split max by latency:",
        max_by_latency,
        "steps",
    )

    print(
        "Safe split limit:",
        safe_split_steps,
        "steps",
    )

    print(
        "Limiting factor:",
        limiting_factor,
    )

    print()
    print(
        "Federated:",
        f"{FEDERATED_STEPS} steps",
        f"| {FEDERATED_SECONDS:.3f}s",
        f"| {FEDERATED_TRAFFIC / 1e6:.3f} MB",
    )

    print()

    print(
        "POLICY:"
    )

    print(
        f"1-{safe_split_steps} steps "
        "-> SPLIT allowed"
    )

    print(
        f">{safe_split_steps} steps "
        "-> prefer FEDERATED_LOCAL"
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
                "max_split_steps_network":
                    max_by_network,
                "max_split_steps_latency":
                    max_by_latency,
                "safe_split_steps":
                    safe_split_steps,
                "limiting_factor":
                    limiting_factor,
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
