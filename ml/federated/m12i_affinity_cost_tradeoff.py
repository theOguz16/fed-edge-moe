import json
from pathlib import Path


OUTPUT = Path(
    "results/m12i_affinity_cost_tradeoff.json"
)

A_AFFINITY = 0.40

B_AFFINITY = 0.20
B_ROUND_SECONDS = 4.06

C_AFFINITIES = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.80,
]

C_ROUND_SECONDS = 11.24


def efficiency(
    a_affinity,
    edge_affinity,
    seconds,
):
    shared = min(
        a_affinity,
        edge_affinity,
    )

    return (
        shared,
        shared / seconds,
    )


def main():
    print()
    print("FedEdgeMoE - M12I")
    print("Affinity vs System-Cost Trade-off")
    print()

    b_shared, b_eff = efficiency(
        A_AFFINITY,
        B_AFFINITY,
        B_ROUND_SECONDS,
    )

    print(
        f"MSI baseline: "
        f"affinity={b_shared*100:.1f}% "
        f"| round={B_ROUND_SECONDS:.2f}s "
        f"| efficiency={b_eff:.4f}"
    )

    print()
    print("Slow-edge scenarios:")

    results = []

    for c_affinity in C_AFFINITIES:
        c_shared, c_eff = efficiency(
            A_AFFINITY,
            c_affinity,
            C_ROUND_SECONDS,
        )

        winner = (
            "slow-edge"
            if c_eff > b_eff
            else "MSI"
        )

        print(
            f"C affinity={c_affinity*100:5.1f}% "
            f"| shared={c_shared*100:5.1f}% "
            f"| efficiency={c_eff:.4f} "
            f"| winner={winner}"
        )

        results.append({
            "c_affinity":
                c_affinity,

            "c_shared_affinity":
                c_shared,

            "c_efficiency":
                c_eff,

            "winner":
                winner,
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
                "a_affinity":
                    A_AFFINITY,

                "b_affinity":
                    B_AFFINITY,

                "b_round_seconds":
                    B_ROUND_SECONDS,

                "b_efficiency":
                    b_eff,

                "c_round_seconds":
                    C_ROUND_SECONDS,

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
