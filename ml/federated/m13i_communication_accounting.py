import json
from pathlib import Path


LOCAL_STEPS = 120

M13H = Path(
    "checkpoints/m13h"
)

EXPERT = Path(
    "checkpoints/m13a/v0100_L0_E7/"
    "experts/layer_00/"
    "expert_07_v0100.safetensors"
)

BUNDLE = Path(
    "checkpoints/m13a/"
    "v0100_L0_E7.tar.gz"
)

FULL = Path(
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

OUTPUT = Path(
    "results/m13i_communication_accounting.json"
)


def size(path):
    return path.stat().st_size


def directory_size(path):
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file()
    )


def mb(value):
    return value / 1_000_000


def main():
    job_bytes = size(
        M13H / "job.safetensors"
    )

    forward_bytes = size(
        M13H / "forward.safetensors"
    )

    gradient_bytes = size(
        M13H / "gradient.safetensors"
    )

    updated_bytes = size(
        M13H / "updated.safetensors"
    )

    expert_bytes = size(EXPERT)

    bundle_bytes = size(BUNDLE)

    full_bytes = directory_size(
        FULL
    )

    split_per_step = (
        job_bytes
        + forward_bytes
        + gradient_bytes
    )

    split_120 = (
        expert_bytes
        + LOCAL_STEPS
        * split_per_step
        + updated_bytes
    )

    federated_expert_only = (
        expert_bytes
        + updated_bytes
    )

    federated_bundle = (
        bundle_bytes
        + updated_bytes
    )

    old_full_snapshot = (
        full_bytes
        + updated_bytes
    )

    print()
    print("FedEdgeMoE - M13I")
    print("Communication Accounting")
    print()

    print("Measured M13H payloads:")
    print(
        f"job hidden-state : "
        f"{job_bytes / 1000:.1f} KB"
    )
    print(
        f"expert output    : "
        f"{forward_bytes / 1000:.1f} KB"
    )
    print(
        f"gradient         : "
        f"{gradient_bytes / 1000:.1f} KB"
    )
    print(
        f"expert shard     : "
        f"{expert_bytes / 1000:.1f} KB"
    )

    print()
    print(
        f"Split training "
        f"({LOCAL_STEPS} steps): "
        f"{mb(split_120):.3f} MB"
    )

    print(
        "Federated expert-only: "
        f"{mb(federated_expert_only):.3f} MB"
    )

    print(
        "Federated shard bundle: "
        f"{mb(federated_bundle):.3f} MB"
    )

    print(
        "Old full snapshot FL: "
        f"{mb(old_full_snapshot):.3f} MB"
    )

    print()
    print(
        "Split / expert-only ratio:",
        f"{split_120 / federated_expert_only:.1f}x",
    )

    print(
        "Split / bundle-FL ratio:",
        f"{split_120 / federated_bundle:.1f}x",
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
                "local_steps":
                    LOCAL_STEPS,
                "payloads": {
                    "job_bytes":
                        job_bytes,
                    "forward_bytes":
                        forward_bytes,
                    "gradient_bytes":
                        gradient_bytes,
                    "expert_bytes":
                        expert_bytes,
                },
                "totals": {
                    "split_training_bytes":
                        split_120,
                    "federated_expert_only_bytes":
                        federated_expert_only,
                    "federated_bundle_bytes":
                        federated_bundle,
                    "old_full_snapshot_bytes":
                        old_full_snapshot,
                },
                "ratios": {
                    "split_vs_expert_only":
                        split_120
                        / federated_expert_only,
                    "split_vs_bundle_fl":
                        split_120
                        / federated_bundle,
                },
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
