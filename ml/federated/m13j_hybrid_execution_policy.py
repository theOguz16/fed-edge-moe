import json
from pathlib import Path


INPUT = Path(
    "results/m13i_communication_accounting.json"
)

OUTPUT = Path(
    "results/m13j_hybrid_execution_policy.json"
)


def split_cost(
    payloads,
    steps,
):
    per_step = (
        payloads["job_bytes"]
        + payloads["forward_bytes"]
        + payloads["gradient_bytes"]
    )

    return (
        payloads["expert_bytes"]
        + steps * per_step
        + payloads["expert_bytes"]
    )


def choose_mode(
    local_training_capable,
    steps,
    network_budget_mb,
    payloads,
):
    budget_bytes = (
        network_budget_mb
        * 1_000_000
    )

    federated_bytes = (
        payloads["expert_bytes"]
        * 2
    )

    split_bytes = split_cost(
        payloads,
        steps,
    )

    if (
        local_training_capable
        and federated_bytes
        <= budget_bytes
    ):
        return {
            "mode":
                "FEDERATED_LOCAL",
            "traffic_bytes":
                federated_bytes,
        }

    if split_bytes <= budget_bytes:
        return {
            "mode":
                "SPLIT_TRAINING",
            "traffic_bytes":
                split_bytes,
        }

    return {
        "mode":
            "SKIP_DEVICE",
        "traffic_bytes":
            None,
    }


def main():
    with open(
        INPUT,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    payloads = report["payloads"]

    scenarios = [
        {
            "name":
                "capable_edge_120_steps",
            "local_training_capable":
                True,
            "steps":
                120,
            "network_budget_mb":
                5.0,
        },
        {
            "name":
                "thin_edge_1_step",
            "local_training_capable":
                False,
            "steps":
                1,
            "network_budget_mb":
                5.0,
        },
        {
            "name":
                "thin_edge_120_steps",
            "local_training_capable":
                False,
            "steps":
                120,
            "network_budget_mb":
                5.0,
        },
    ]

    results = []

    print()
    print("FedEdgeMoE - M13J")
    print("Hybrid Execution Policy")
    print()

    for scenario in scenarios:
        decision = choose_mode(
            scenario[
                "local_training_capable"
            ],
            scenario["steps"],
            scenario[
                "network_budget_mb"
            ],
            payloads,
        )

        traffic = (
            "-"
            if decision["traffic_bytes"]
            is None
            else (
                f"{decision['traffic_bytes'] / 1_000_000:.3f} MB"
            )
        )

        print(
            f"{scenario['name']:<24} "
            f"| mode={decision['mode']:<16} "
            f"| traffic={traffic}"
        )

        results.append({
            **scenario,
            **decision,
        })

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
                "policy":
                    "hybrid_execution",
                "results":
                    results,
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
