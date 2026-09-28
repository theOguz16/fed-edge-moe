import json
from pathlib import Path


ROUTING_REPORT = Path(
    "results/m11a_affinity_scheduler.json"
)

COST_REPORT = Path(
    "results/m12f_device_cost_profile.json"
)

OUTPUT = Path(
    "results/m12h_three_client_testbed.json"
)

TARGET_LAYER = 0
TARGET_EXPERT = 7


def main():
    with open(ROUTING_REPORT) as f:
        routing = json.load(f)

    with open(COST_REPORT) as f:
        cost = json.load(f)

    shares_a = routing["clients"]["client_A"]["routing_shares"]
    shares_b = routing["clients"]["client_B"]["routing_shares"]

    affinity_a = shares_a[TARGET_LAYER][TARGET_EXPERT]
    affinity_b = shares_b[TARGET_LAYER][TARGET_EXPERT]

    # C intentionally has same data/routing profile as B.
    # Only system cost differs.
    affinity_c = affinity_b

    clients = {
        "A_mac": {
            "affinity": affinity_a,
            "training_seconds":
                cost["mac_mps"]["training_seconds"],
            "download_seconds": 0.0,
        },

        "B_msi": {
            "affinity": affinity_b,
            "training_seconds":
                cost["msi_cuda"]["training_seconds"],
            "download_seconds":
                cost["msi_cuda"]["download_seconds"],
        },

        "C_slow_edge": {
            "affinity": affinity_c,
            "training_seconds": 8.0,
            "download_seconds": 3.0,
        },
    }

    aggregation = cost["aggregation_seconds"]

    candidates = []

    for remote in ["B_msi", "C_slow_edge"]:
        local = clients["A_mac"]
        edge = clients[remote]

        shared_affinity = min(
            local["affinity"],
            edge["affinity"],
        )

        local_time = (
            local["training_seconds"]
            + local["download_seconds"]
        )

        remote_time = (
            edge["training_seconds"]
            + edge["download_seconds"]
        )

        estimated_round_seconds = (
            max(
                local_time,
                remote_time,
            )
            + aggregation
        )

        efficiency = (
            shared_affinity
            / estimated_round_seconds
        )

        candidates.append({
            "pair":
                f"A_mac + {remote}",

            "shared_affinity":
                shared_affinity,

            "estimated_round_seconds":
                estimated_round_seconds,

            "affinity_per_second":
                efficiency,
        })

    candidates.sort(
        key=lambda x:
            x["affinity_per_second"],
        reverse=True,
    )

    print()
    print("FedEdgeMoE - M12H")
    print("Three-Client Heterogeneous Testbed")
    print()

    print(
        f"Target expert: "
        f"L{TARGET_LAYER}-E{TARGET_EXPERT}"
    )

    print()

    for item in candidates:
        print(
            f"{item['pair']:<24} "
            f"| affinity="
            f"{item['shared_affinity'] * 100:5.2f}% "
            f"| round="
            f"{item['estimated_round_seconds']:5.2f}s "
            f"| efficiency="
            f"{item['affinity_per_second']:.4f}"
        )

    winner = candidates[0]

    print()
    print(
        "COST-AWARE DEVICE CHOICE:",
        winner["pair"],
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
                "target_layer":
                    TARGET_LAYER,

                "target_expert":
                    TARGET_EXPERT,

                "client_C":
                    "simulated slow edge",

                "candidates":
                    candidates,

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
