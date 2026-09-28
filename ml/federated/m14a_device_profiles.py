import json
from pathlib import Path


OUTPUT = Path(
    "results/m14a_device_profiles.json"
)


DEVICES = [
    {
        "device_id": "mac_m4",
        "role": "server_edge",
        "compute": {
            "backend": "mps",
            "memory_gb": 16.0,
        },
        "network": {
            "type": "lan",
        },
        "capabilities": {
            "federated_local_training": True,
            "split_execution": True,
        },
        "measured": {
            "local_train_seconds_120":
                3.462,
        },
    },

    {
        "device_id": "msi_rtx3050",
        "role": "edge",
        "compute": {
            "backend": "cuda",
            "gpu": "RTX 3050 Laptop",
            "gpu_memory_gb": 4.0,
            "system_memory_gb": 8.0,
        },
        "network": {
            "type": "lan",
        },
        "capabilities": {
            "federated_local_training": True,
            "split_execution": True,
        },
        "measured": {
            "federated_e2e_seconds_120":
                7.712,

            "split_e2e_seconds_1":
                1.940,

            "snapshot_download_seconds":
                0.295,
        },
    },

    {
        "device_id": "slow_edge_sim",
        "role": "edge",
        "compute": {
            "backend": "simulated",
            "memory_gb": 2.0,
        },
        "network": {
            "type": "simulated_slow",
        },
        "capabilities": {
            "federated_local_training":
                False,

            "split_execution":
                True,
        },
        "measured": {},
        "estimated": {
            "round_seconds":
                11.24,
        },
    },
]


def main():
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
                "schema_version": 1,
                "devices": DEVICES,
            },
            f,
            indent=2,
        )

    print()
    print("FedEdgeMoE - M14A")
    print("Device Capability Profiles")
    print()

    for device in DEVICES:
        caps = device[
            "capabilities"
        ]

        print(
            f"{device['device_id']:<15} "
            f"| FL={str(caps['federated_local_training']):<5} "
            f"| SPLIT={str(caps['split_execution']):<5}"
        )

    print()
    print(
        "Profiles:",
        len(DEVICES),
    )

    print(
        "Report:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
