from pathlib import Path

import torch
from safetensors.torch import load_file


SOURCE = Path(
    "checkpoints/m10_candidates/global_step_0100"
)

BUNDLE = Path(
    "checkpoints/m13a/v0100_L0_E7"
)

TARGET_LAYER = 0
TARGET_EXPERT = 7
VERSION = 100


def compare_files(source, bundled):
    a = load_file(str(source))
    b = load_file(str(bundled))

    if set(a.keys()) != set(b.keys()):
        return False, None

    max_diff = 0.0

    for key in a:
        diff = (
            a[key] - b[key]
        ).abs().max().item()

        max_diff = max(
            max_diff,
            diff,
        )

    return True, max_diff


def main():
    print()
    print("FedEdgeMoE - M13B")
    print("Shard Bundle Integrity Check")
    print()

    checks = []

    shared_ok, shared_diff = compare_files(
        SOURCE / "shared.safetensors",
        BUNDLE / "shared.safetensors",
    )

    checks.append(
        ("shared", shared_ok, shared_diff)
    )

    for layer in range(3):
        name = (
            f"layer_{layer:02d}_router.safetensors"
        )

        ok, diff = compare_files(
            SOURCE / "routers" / name,
            BUNDLE / "routers" / name,
        )

        checks.append(
            (f"router_L{layer}", ok, diff)
        )

    expert_name = (
        f"expert_{TARGET_EXPERT:02d}"
        f"_v{VERSION:04d}.safetensors"
    )

    expert_source = (
        SOURCE
        / "experts"
        / f"layer_{TARGET_LAYER:02d}"
        / expert_name
    )

    expert_bundle = (
        BUNDLE
        / "experts"
        / f"layer_{TARGET_LAYER:02d}"
        / expert_name
    )

    expert_ok, expert_diff = compare_files(
        expert_source,
        expert_bundle,
    )

    checks.append(
        ("target_expert", expert_ok, expert_diff)
    )

    bundled_experts = list(
        (BUNDLE / "experts").rglob(
            "*.safetensors"
        )
    )

    print(
        "Bundled expert count:",
        len(bundled_experts),
    )

    for name, ok, diff in checks:
        print(
            f"{name:<14} "
            f"| exact_keys={ok} "
            f"| max_diff={diff}"
        )

    all_exact = (
        len(bundled_experts) == 1
        and all(
            ok and diff == 0.0
            for _, ok, diff in checks
        )
    )

    print()
    print(
        "BUNDLE EXACT:",
        all_exact,
    )


if __name__ == "__main__":
    main()
