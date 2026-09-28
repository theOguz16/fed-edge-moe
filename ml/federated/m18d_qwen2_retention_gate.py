import json
from pathlib import Path


INPUT = Path(
    "results/m18c_qwen2moe_federated_round.json"
)

OUTPUT = Path(
    "results/m18d_qwen2_retention_gate.json"
)

LAMBDA = 2.0


def main():
    with open(
        INPUT,
        "r",
        encoding="utf-8",
    ) as f:
        report = json.load(f)

    before = report["baseline"]
    after = report["final"]

    improvements = [
        before[i] - after[i]
        for i in range(4)
    ]

    target_gain = (
        improvements[0]
        + improvements[1]
    ) / 2.0

    forgetting = (
        max(0.0, -improvements[2])
        + max(0.0, -improvements[3])
    ) / 2.0

    utility = (
        target_gain
        - LAMBDA * forgetting
    )

    decision = (
        "ACCEPT"
        if utility > 0
        else "REJECT"
    )

    print()
    print("FedEdgeMoE - M18D")
    print("Qwen2 Retention-Aware Gate")
    print()

    print(
        "Target gain:",
        f"{target_gain:+.6f}",
    )

    print(
        "Retention damage:",
        f"{forgetting:.6f}",
    )

    print(
        "Lambda:",
        LAMBDA,
    )

    print(
        "Utility:",
        f"{utility:+.6f}",
    )

    print()

    print(
        "GLOBAL UPDATE:",
        decision,
    )

    result = {
        "target_gain":
            target_gain,
        "retention_damage":
            forgetting,
        "lambda":
            LAMBDA,
        "utility":
            utility,
        "decision":
            decision,
    }

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
            result,
            f,
            indent=2,
        )

    print(
        "Report:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
