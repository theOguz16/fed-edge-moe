import json
import shutil
from pathlib import Path


INPUT = Path(
    "results/m19e_physical_aggregation.json"
)

SOURCE = Path(
    "checkpoints/m19e/"
    "global_L0_E0_physical.safetensors"
)

OUTPUT = Path(
    "checkpoints/m19f/global_v001/"
    "L0_E0.safetensors"
)

REPORT = Path(
    "results/m19f_physical_commit.json"
)

LAMBDA = 2.0


def main():
    with open(INPUT, "r") as f:
        data = json.load(f)

    before = data["baseline"]
    after = data["final"]

    gain = [
        before[i] - after[i]
        for i in range(4)
    ]

    target_gain = (
        gain[0] + gain[1]
    ) / 2.0

    retention_damage = (
        max(0.0, -gain[2])
        + max(0.0, -gain[3])
    ) / 2.0

    utility = (
        target_gain
        - LAMBDA * retention_damage
    )

    decision = (
        "ACCEPT"
        if utility > 0
        else "REJECT"
    )

    committed = False

    if decision == "ACCEPT":
        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            SOURCE,
            OUTPUT,
        )

        committed = True

    print()
    print("FedEdgeMoE - M19F")
    print("Physical Round Commit")
    print()

    print(
        "Target gain:",
        f"{target_gain:+.6f}",
    )

    print(
        "Retention damage:",
        f"{retention_damage:.6f}",
    )

    print(
        "Utility:",
        f"{utility:+.6f}",
    )

    print(
        "Decision:",
        decision,
    )

    print(
        "Version:",
        "V000 -> V001"
        if committed
        else "V000 unchanged",
    )

    print()

    print(
        "PHYSICAL ROUND COMMITTED:",
        committed,
    )

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(REPORT, "w") as f:
        json.dump(
            {
                "target_gain":
                    target_gain,
                "retention_damage":
                    retention_damage,
                "lambda":
                    LAMBDA,
                "utility":
                    utility,
                "decision":
                    decision,
                "committed":
                    committed,
                "version":
                    1 if committed else 0,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
