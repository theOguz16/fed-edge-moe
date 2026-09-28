RETENTION_BUDGET_PP = 0.20

EXPERTS = [
    {
        "name": "L1-E4",
        "target_gain": 7.06,
        "retention_change": 0.00,
    },
    {
        "name": "L0-E7",
        "target_gain": 8.96,
        "retention_change": -0.17,
    },
    {
        "name": "L0-E3",
        "target_gain": 7.35,
        "retention_change": -0.35,
    },
]

DEVICES = [
    {
        "name": "MSI",
        "round_seconds": 4.06,
        "shared_affinity": 0.20,
    },
    {
        "name": "slow-edge",
        "round_seconds": 11.24,
        "shared_affinity": 0.40,
    },
]

FAST_EFFICIENCY = 0.20 / 4.06
PARTNER_AFFINITY_LIMIT = 0.40


def main():
    print()
    print("FedEdgeMoE - M12L")
    print("Unified Scheduler")
    print()

    feasible_experts = []

    for expert in EXPERTS:
        forgetting = max(
            0.0,
            -expert["retention_change"],
        )

        if forgetting <= RETENTION_BUDGET_PP:
            feasible_experts.append(expert)

    feasible_experts.sort(
        key=lambda x: x["target_gain"],
        reverse=True,
    )

    selected_expert = feasible_experts[0]

    print(
        "Expert:",
        selected_expert["name"],
    )

    print(
        "Target gain:",
        f"{selected_expert['target_gain']:+.2f}pp",
    )

    print(
        "Retention:",
        f"{selected_expert['retention_change']:+.2f}pp",
    )

    print()

    viable_devices = []

    for device in DEVICES:
        required_affinity = (
            FAST_EFFICIENCY
            * device["round_seconds"]
        )

        viable = (
            required_affinity
            <= PARTNER_AFFINITY_LIMIT
        )

        print(
            f"{device['name']:<10} "
            f"| round={device['round_seconds']:.2f}s "
            f"| required={required_affinity*100:.1f}% "
            f"| {'KEEP' if viable else 'PRUNE'}"
        )

        if viable:
            efficiency = (
                device["shared_affinity"]
                / device["round_seconds"]
            )

            viable_devices.append({
                **device,
                "efficiency": efficiency,
            })

    viable_devices.sort(
        key=lambda x: x["efficiency"],
        reverse=True,
    )

    selected_device = viable_devices[0]

    print()
    print("FINAL SCHEDULER DECISION")
    print(
        selected_expert["name"],
        "→",
        selected_device["name"],
    )


if __name__ == "__main__":
    main()
