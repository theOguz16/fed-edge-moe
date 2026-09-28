import json
from pathlib import Path


OUTPUT = Path(
    "results/m14f_unified_heterogeneous_scheduler.json"
)

TASK = {
    "expert": "L0-E7",
    "mode": "FEDERATED_LOCAL",
    "required_memory_gb": 3.5,
}

DEVICES = [
    {
        "id": "mac_m4",
        "available": True,
        "fl_capable": True,
        "memory_gb": 16.0,
        "affinity": 0.214,
        "train_seconds": 3.462,
    },
    {
        "id": "msi_rtx3050",
        "available": True,
        "fl_capable": True,
        "memory_gb": 4.0,
        "affinity": 0.191,
        "train_seconds": 3.458,
    },
    {
        "id": "slow_edge_sim",
        "available": True,
        "fl_capable": False,
        "memory_gb": 2.0,
        "affinity": 0.40,
        "train_seconds": 8.0,
    },
]


def main():
    print()
    print("FedEdgeMoE - M14F")
    print("Unified Heterogeneous Scheduler")
    print()

    candidates = []

    for d in DEVICES:
        reason = None

        if not d["available"]:
            reason = "UNAVAILABLE"

        elif not d["fl_capable"]:
            reason = "NO_FL_CAPABILITY"

        elif (
            d["memory_gb"]
            < TASK["required_memory_gb"]
        ):
            reason = "INSUFFICIENT_MEMORY"

        if reason:
            print(
                f"{d['id']:<15} "
                f"| PRUNE | {reason}"
            )
            continue

        score = (
            d["affinity"]
            / d["train_seconds"]
        )

        candidates.append({
            **d,
            "score": score,
        })

        print(
            f"{d['id']:<15} "
            f"| KEEP "
            f"| memory={d['memory_gb']:.1f}GB "
            f"| affinity={d['affinity']*100:.1f}% "
            f"| train={d['train_seconds']:.3f}s "
            f"| score={score:.4f}"
        )

    if not candidates:
        print()
        print("DECISION: DEFER_TASK")
        return

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    winner = candidates[0]

    print()
    print(
        "SELECTED DEVICE:",
        winner["id"],
    )

    print(
        "SELECTED EXPERT:",
        TASK["expert"],
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
                "task": TASK,
                "candidates": candidates,
                "selected_device":
                    winner["id"],
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
