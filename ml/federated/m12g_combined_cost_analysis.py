import json
from pathlib import Path

M11B = Path("results/m11b_scheduler_benchmark.json")
M11C = Path("results/m11c_e3_fixed_horizon.json")
COST = Path("results/m12f_device_cost_profile.json")

LAMBDA_RETENTION = 2.0


def utility(target_gain, retention_change):
    forgetting = max(0.0, -retention_change)
    return target_gain - LAMBDA_RETENTION * forgetting


def main():
    with open(M11B) as f:
        m11b = json.load(f)

    with open(M11C) as f:
        m11c = json.load(f)

    with open(COST) as f:
        cost = json.load(f)

    options = []

    for item in m11b["results"]:
        options.append({
            "expert": f"L{item['layer']}-E{item['expert']}",
            "target_gain": item["target_gain"],
            "retention_change": item["retention_change"],
        })

    e3 = m11c["history"][-1]

    for item in options:
        if item["expert"] == "L0-E3":
            item["target_gain"] = e3["target_gain"]
            item["retention_change"] = e3["retention_change"]

    round_seconds = cost["steady_round_seconds"]
    communication_mb = cost["msi_cuda"]["communication_mb"]

    print()
    print("FedEdgeMoE - M12G")
    print("Combined ML + System Cost Analysis")
    print()

    print(f"Shared round cost: {round_seconds:.3f}s")
    print(f"Shared communication: {communication_mb:.3f} MB")
    print()

    options.sort(
        key=lambda x: utility(
            x["target_gain"],
            x["retention_change"],
        ),
        reverse=True,
    )

    for rank, item in enumerate(options, 1):
        score = utility(
            item["target_gain"],
            item["retention_change"],
        )

        print(
            f"{rank}. {item['expert']} "
            f"| target={item['target_gain']*100:+.2f}pp "
            f"| retention={item['retention_change']*100:+.2f}pp "
            f"| ML utility={score*100:+.2f} "
            f"| cost={round_seconds:.2f}s / {communication_mb:.2f}MB"
        )

    print()
    print("SYSTEM-COST EFFECT ON RANKING: NONE")
    print("Reason: all candidates use the same Mac+MSI assignment.")


if __name__ == "__main__":
    main()
