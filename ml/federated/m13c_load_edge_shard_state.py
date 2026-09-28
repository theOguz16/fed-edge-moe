import json
from pathlib import Path

import torch
from safetensors.torch import load_file


BUNDLE = Path(
    "checkpoints/m13a/v0100_L0_E7"
)

MANIFEST = BUNDLE / "manifest.json"


def count_parameters(state):
    return sum(
        tensor.numel()
        for tensor in state.values()
    )


def main():
    print()
    print("FedEdgeMoE - M13C")
    print("Edge Shard State Loader")
    print()

    with open(
        MANIFEST,
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    shared = load_file(
        str(BUNDLE / "shared.safetensors")
    )

    routers = {}

    for path in sorted(
        (BUNDLE / "routers").glob(
            "*.safetensors"
        )
    ):
        routers[path.name] = (
            load_file(str(path))
        )

    expert_path = (
        BUNDLE
        / manifest["files"]["expert"]
    )

    expert = load_file(
        str(expert_path)
    )

    shared_params = count_parameters(
        shared
    )

    router_params = sum(
        count_parameters(state)
        for state in routers.values()
    )

    expert_params = count_parameters(
        expert
    )

    total_params = (
        shared_params
        + router_params
        + expert_params
    )

    trainable_expert = {
        key:
            value.clone()
            .detach()
            .requires_grad_(True)

        for key, value
        in expert.items()
    }

    trainable_params = sum(
        tensor.numel()
        for tensor
        in trainable_expert.values()
        if tensor.requires_grad
    )

    print(
        "Global version:",
        manifest["global_version"],
    )

    print(
        "Target:",
        f"L{manifest['target_layer']}"
        f"-E{manifest['target_expert']}",
    )

    print()
    print(
        "Shared params:",
        f"{shared_params:,}",
    )

    print(
        "Router params:",
        f"{router_params:,}",
    )

    print(
        "Expert params:",
        f"{expert_params:,}",
    )

    print(
        "Loaded shard params:",
        f"{total_params:,}",
    )

    print()
    print(
        "Trainable params:",
        f"{trainable_params:,}",
    )

    print(
        "Only expert trainable:",
        trainable_params == expert_params,
    )

    print(
        "EDGE SHARD LOAD OK:",
        (
            trainable_params
            == expert_params
            and expert_params > 0
        ),
    )


if __name__ == "__main__":
    main()
