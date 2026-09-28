from pathlib import Path

import torch

from ml.model.transformer import MiniMoELM

from ml.federated.checkpoint_manager import (
    export_global_snapshot,
    load_global_snapshot,
    export_single_expert,
    load_single_expert,
)


def build_model():
    return MiniMoELM(
        vocab_size=64,
        max_seq_len=32,

        d_model=128,
        num_heads=4,
        num_layers=3,

        num_experts=8,
        top_k=2,

        expert_hidden_dim=256,

        router_aux_loss_weight=0.01,
    )


def compare_models(model_a, model_b):
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    max_difference = 0.0
    different_keys = []

    for key in state_a:

        a = state_a[key].detach().cpu()
        b = state_b[key].detach().cpu()

        difference = (
            a - b
        ).abs().max().item()

        max_difference = max(
            max_difference,
            difference,
        )

        if difference != 0:
            different_keys.append(
                (key, difference)
            )

    return (
        max_difference,
        different_keys,
    )


def main():

    print()
    print("M2 - Expert Isolation Test")
    print()

    # ========================================================
    # 1. Original trained M1 model
    # ========================================================

    original = build_model()

    checkpoint = torch.load(
        "checkpoints/m1/minimoe_m1.pt",
        map_location="cpu",
        weights_only=True,
    )

    original.load_state_dict(
        checkpoint
    )

    original.eval()

    # ========================================================
    # 2. Export global snapshot
    # ========================================================

    snapshot_dir = Path(
        "checkpoints/m2/global_v0000"
    )

    export_global_snapshot(
        model=original,
        output_dir=snapshot_dir,
        global_version=0,
    )

    print(
        "Global snapshot exported:"
    )

    print(snapshot_dir)

    # ========================================================
    # 3. Fresh model
    # ========================================================

    restored = build_model()

    load_global_snapshot(
        model=restored,
        snapshot_dir=snapshot_dir,
    )

    restored.eval()

    # ========================================================
    # 4. Verify perfect reconstruction
    # ========================================================

    max_diff, different = compare_models(
        original,
        restored,
    )

    print()
    print(
        "Max parameter difference "
        "after reconstruction:"
    )

    print(max_diff)

    assert max_diff == 0.0

    print()
    print(
        "Global reconstruction OK"
    )

    # ========================================================
    # 5. Modify only one expert
    #
    # Target:
    # Layer 0
    # Expert 6
    # ========================================================

    layer_idx = 0
    expert_idx = 6

    target_expert = (
        restored
        .blocks[layer_idx]
        .moe
        .experts[expert_idx]
    )

    with torch.no_grad():

        for parameter in (
            target_expert.parameters()
        ):
            parameter.add_(0.001)

    # ========================================================
    # 6. Detect exactly what changed
    # ========================================================

    _, changed = compare_models(
        original,
        restored,
    )

    print()
    print("Changed parameters:")

    for key, difference in changed:
        print(
            f"{key}: "
            f"{difference:.6f}"
        )

    expected_prefix = (
        "blocks.0.moe.experts.6."
    )

    assert len(changed) > 0

    assert all(
        key.startswith(
            expected_prefix
        )
        for key, _ in changed
    )

    print()
    print(
        "Expert isolation verified:"
    )

    print(
        "Only L0-E6 changed."
    )

    # ========================================================
    # 7. Export modified expert separately
    # ========================================================

    modified_file = Path(
        "checkpoints/m2/client_sim/"
        "layer_00_expert_06_v0001.safetensors"
    )

    export_single_expert(
        model=restored,
        layer_idx=0,
        expert_idx=6,
        output_file=modified_file,
    )

    print()
    print(
        "Modified expert exported:"
    )

    print(modified_file)

    # ========================================================
    # 8. Restore original L0-E6 from global snapshot
    # ========================================================

    original_expert_file = (
        snapshot_dir
        / "experts"
        / "layer_00"
        / "expert_06_v0000.safetensors"
    )

    load_single_expert(
        model=restored,
        layer_idx=0,
        expert_idx=6,
        expert_file=original_expert_file,
    )

    final_diff, final_changed = (
        compare_models(
            original,
            restored,
        )
    )

    print()
    print(
        "Difference after expert restore:"
    )

    print(final_diff)

    assert final_diff == 0.0

    assert len(final_changed) == 0

    print()
    print(
        "Single expert restore OK"
    )

    print()
    print("=" * 60)

    print(
        "M2 EXPERT ISOLATION SUCCESS"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
