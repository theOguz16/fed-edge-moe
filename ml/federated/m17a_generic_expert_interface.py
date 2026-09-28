import torch

from ml.model.expert import ExpertMLP
from ml.adapters.expert_adapter import (
    MiniMoEExpertAdapter,
)


def main():
    torch.manual_seed(20260927)

    expert = ExpertMLP(
        d_model=128,
        hidden_dim=256,
    )

    adapter = MiniMoEExpertAdapter(
        expert
    )

    layers = (
        adapter.projection_layers()
    )

    params = (
        adapter.parameter_count()
    )

    before = adapter.state_dict()

    adapter.freeze()

    frozen = all(
        not p.requires_grad
        for p in expert.parameters()
    )

    adapter.unfreeze()

    unfrozen = all(
        p.requires_grad
        for p in expert.parameters()
    )

    with torch.no_grad():
        expert.gate_proj.weight.add_(
            0.001
        )

    after = adapter.state_dict()

    delta = adapter.delta(
        before,
        after,
    )

    delta_norm = (
        adapter.delta_l2_norm(
            delta
        )
    )

    print()
    print("FedEdgeMoE - M17A")
    print("Generic Expert Interface")
    print()

    print(
        "Detected projections:",
        ", ".join(layers.keys()),
    )

    print(
        "Expert params:",
        f"{params:,}",
    )

    print(
        "Freeze works:",
        frozen,
    )

    print(
        "Unfreeze works:",
        unfrozen,
    )

    print(
        "Delta L2 norm:",
        f"{delta_norm:.6f}",
    )

    ok = (
        list(layers.keys())
        == [
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        and params == 98_304
        and frozen
        and unfrozen
        and delta_norm > 0
    )

    print()
    print(
        "GENERIC INTERFACE OK:",
        ok,
    )


if __name__ == "__main__":
    main()
