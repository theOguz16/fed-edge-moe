from pathlib import Path
import json


OUTPUT = Path(
    "results/m12j_crossover_threshold.json"
)


def required_affinity(
    fast_affinity,
    fast_seconds,
    slow_seconds,
):
    fast_efficiency = (
        fast_affinity
        / fast_seconds
    )

    required = (
        fast_efficiency
        * slow_seconds
    )

    return (
        fast_efficiency,
        required,
    )


def main():
    print()
    print("FedEdgeMoE - M12J")
    print("Device Crossover Threshold")
    print()

    fast_affinity = 0.20
    fast_seconds = 4.06

    slow_seconds_list = [
        5.0,
        6.0,
        8.0,
        10.0,
        11.24,
        14.0,
    ]

    partner_affinity_limit = 0.40

    results = []

    for slow_seconds in slow_seconds_list:
        (
            fast_eff,
            required,
        ) = required_affinity(
            fast_affinity,
            fast_seconds,
            slow_seconds,
        )

        possible = (
            required
            <= partner_affinity_limit
        )

        print(
            f"slow_round={slow_seconds:5.2f}s "
            f"| required_shared="
            f"{required*100:5.1f}% "
            f"| possible={possible}"
        )

        results.append({
            "slow_round_seconds":
                slow_seconds,

            "required_shared_affinity":
                required,

            "partner_affinity_limit":
                partner_affinity_limit,

            "possible":
                possible,
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
                "fast_affinity":
                    fast_affinity,

                "fast_round_seconds":
                    fast_seconds,

                "partner_affinity_limit":
                    partner_affinity_limit,

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
