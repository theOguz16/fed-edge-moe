import json
from pathlib import Path


PHYSICAL_REPORT = Path(
    "results/m12n_physical_timing.json"
)

OUTPUT = Path(
    "results/m12p_e2e_cost_scheduler.json"
)

MSI_AFFINITY = 0.20
SLOW_EDGE_AFFINITY = 0.40
SLOW_EDGE_SECONDS = 11.24

OLD_MSI_ESTIMATE = 4.06


def find_round_seconds(obj):
    values = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "round_seconds":
                values.append(float(value))
            else:
                values.extend(
                    find_round_seconds(value)
                )

    elif isinstance(obj, list):
        for item in obj:
            values.extend(
                find_round_seconds(item)
            )

    return values


def efficiency(
    affinity,
    seconds,
):
    return affinity / seconds


def main():
    with open(
        PHYSICAL_REPORT,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    round_values = find_round_seconds(
        report
    )

    if not round_values:
        raise RuntimeError(
            "round_seconds not found"
        )

    measured_msi_seconds = (
        round_values[-1]
    )

    old_eff = efficiency(
        MSI_AFFINITY,
        OLD_MSI_ESTIMATE,
    )

    measured_eff = efficiency(
        MSI_AFFINITY,
        measured_msi_seconds,
    )

    slow_eff = efficiency(
        SLOW_EDGE_AFFINITY,
        SLOW_EDGE_SECONDS,
    )

    required_slow_affinity = (
        measured_eff
        * SLOW_EDGE_SECONDS
    )

    winner = (
        "MSI"
        if measured_eff >= slow_eff
        else "slow-edge"
    )

    print()
    print("FedEdgeMoE - M12P")
    print("End-to-End Cost-Aware Scheduler")
    print()

    print(
        f"Old MSI estimate: "
        f"{OLD_MSI_ESTIMATE:.3f}s "
        f"| efficiency={old_eff:.4f}"
    )

    print(
        f"Measured MSI E2E: "
        f"{measured_msi_seconds:.3f}s "
        f"| efficiency={measured_eff:.4f}"
    )

    print(
        f"Slow-edge: "
        f"{SLOW_EDGE_SECONDS:.3f}s "
        f"| efficiency={slow_eff:.4f}"
    )

    print()

    print(
        "Slow-edge required affinity:",
        f"{required_slow_affinity * 100:.1f}%"
    )

    print(
        "COST-AWARE WINNER:",
        winner,
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
                "old_msi_estimate_seconds":
                    OLD_MSI_ESTIMATE,
                "measured_msi_e2e_seconds":
                    measured_msi_seconds,
                "msi_affinity":
                    MSI_AFFINITY,
                "msi_efficiency":
                    measured_eff,
                "slow_edge_seconds":
                    SLOW_EDGE_SECONDS,
                "slow_edge_affinity":
                    SLOW_EDGE_AFFINITY,
                "slow_edge_efficiency":
                    slow_eff,
                "required_slow_affinity":
                    required_slow_affinity,
                "winner":
                    winner,
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
