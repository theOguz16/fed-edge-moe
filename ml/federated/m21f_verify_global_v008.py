import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen2MoeForCausalLM

from ml.training.synthetic_v2 import build_split_tensor


BASE_MODEL = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

REGISTRY_PATH = Path(
    "checkpoints/m21/global_v008/registry.json"
)

SHARDS = [
    {
        "layer": 0,
        "expert": 0,
        "path": Path(
            "checkpoints/m20/global_v006/"
            "L0_E0.safetensors"
        ),
    },
    {
        "layer": 0,
        "expert": 3,
        "path": Path(
            "checkpoints/m21/global_v007/"
            "L0_E3.safetensors"
        ),
    },
    {
        "layer": 1,
        "expert": 2,
        "path": Path(
            "checkpoints/m21/global_v008/"
            "L1_E2.safetensors"
        ),
    },
]

RESULT = Path(
    "results/m21f_global_v008_verification.json"
)


def max_diff(a, b):
    return float(
        (
            a.detach().cpu()
            - b.detach().cpu()
        )
        .abs()
        .max()
    )


def extract_expert(model, layer_idx, expert_idx):
    experts = (
        model.model.layers[layer_idx]
        .mlp.experts
    )

    gate_up = (
        experts.gate_up_proj[expert_idx]
        .detach()
        .cpu()
        .clone()
    )

    split = gate_up.shape[0] // 2

    return {
        "gate_proj":
            gate_up[:split].clone(),

        "up_proj":
            gate_up[split:].clone(),

        "down_proj":
            experts.down_proj[expert_idx]
            .detach()
            .cpu()
            .clone(),
    }


def apply_shard(model, layer_idx, expert_idx, shard):
    experts = (
        model.model.layers[layer_idx]
        .mlp.experts
    )

    gate_up = torch.cat(
        [
            shard["gate_proj"],
            shard["up_proj"],
        ],
        dim=0,
    )

    with torch.no_grad():
        experts.gate_up_proj[
            expert_idx
        ].copy_(gate_up)

        experts.down_proj[
            expert_idx
        ].copy_(
            shard["down_proj"]
        )


@torch.no_grad()
def validation_loss(model, domain):
    model.eval()

    seq = build_split_tensor(
        domain,
        "validation",
        16,
    )[:64]

    labels = seq.clone()
    labels[:, :2] = -100

    output = model(
        input_ids=seq,
        labels=labels,
    )

    return float(
        output.loss.cpu()
    )


def main():
    print()
    print("FedEdgeMoE - M21F")
    print("Global V008 Integrity Verification")
    print()

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Registry missing: {REGISTRY_PATH}"
        )

    registry = json.loads(
        REGISTRY_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        "Registry version:",
        registry.get("version"),
    )

    print(
        "Registry inherits:",
        registry.get("inherits"),
    )

    print()

    # Immutable original model.
    torch.manual_seed(20260927)

    base_model = (
        Qwen2MoeForCausalLM
        .from_pretrained(BASE_MODEL)
    )

    # Reconstructed final V008.
    torch.manual_seed(20260927)

    final_model = (
        Qwen2MoeForCausalLM
        .from_pretrained(BASE_MODEL)
    )

    accepted = set()

    shard_checks = []

    for item in SHARDS:
        layer_idx = item["layer"]
        expert_idx = item["expert"]
        path = item["path"]

        if not path.exists():
            raise FileNotFoundError(
                f"Missing shard: {path}"
            )

        shard = load_file(
            str(path)
        )

        apply_shard(
            final_model,
            layer_idx,
            expert_idx,
            shard,
        )

        reconstructed = extract_expert(
            final_model,
            layer_idx,
            expert_idx,
        )

        diff = max(
            max_diff(
                reconstructed[key],
                shard[key],
            )
            for key in shard
        )

        accepted.add(
            (layer_idx, expert_idx)
        )

        shard_checks.append(
            {
                "layer": layer_idx,
                "expert": expert_idx,
                "path": str(path),
                "max_diff": diff,
            }
        )

        print(
            f"L{layer_idx}-E{expert_idx} "
            f"shard diff: {diff:.3e}"
        )

    print()

    # Verify all unrelated experts still match V000.
    unrelated_max_diff = 0.0
    unrelated_results = []

    num_layers = len(
        base_model.model.layers
    )

    num_experts = (
        base_model.config.num_experts
    )

    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):

            if (
                layer_idx,
                expert_idx,
            ) in accepted:
                continue

            base = extract_expert(
                base_model,
                layer_idx,
                expert_idx,
            )

            final = extract_expert(
                final_model,
                layer_idx,
                expert_idx,
            )

            diff = max(
                max_diff(
                    base[key],
                    final[key],
                )
                for key in base
            )

            unrelated_max_diff = max(
                unrelated_max_diff,
                diff,
            )

            unrelated_results.append(
                {
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "max_diff": diff,
                }
            )

    print(
        "Unrelated expert max diff:",
        f"{unrelated_max_diff:.3e}",
    )

    # Verify routers unchanged.
    router_max_diff = 0.0
    router_results = []

    for layer_idx in range(num_layers):
        base_router = (
            base_model.model.layers[
                layer_idx
            ].mlp.gate
        )

        final_router = (
            final_model.model.layers[
                layer_idx
            ].mlp.gate
        )

        layer_router_diff = 0.0

        base_state = (
            base_router.state_dict()
        )

        final_state = (
            final_router.state_dict()
        )

        for name in base_state:
            diff = max_diff(
                base_state[name],
                final_state[name],
            )

            layer_router_diff = max(
                layer_router_diff,
                diff,
            )

        router_max_diff = max(
            router_max_diff,
            layer_router_diff,
        )

        router_results.append(
            {
                "layer": layer_idx,
                "max_diff":
                    layer_router_diff,
            }
        )

    print(
        "Router max diff:",
        f"{router_max_diff:.3e}",
    )

    print()
    print("Validation losses:")

    losses = {}

    for domain in range(4):
        loss = validation_loss(
            final_model,
            domain,
        )

        losses[f"D{domain}"] = loss

        print(
            f"  D{domain}: "
            f"{loss:.6f}"
        )

    shard_max_diff = max(
        item["max_diff"]
        for item in shard_checks
    )

    registry_ok = (
        registry.get("version") == 8
        and registry.get("inherits") == 7
    )

    integrity_ok = (
        registry_ok
        and shard_max_diff == 0.0
        and unrelated_max_diff == 0.0
        and router_max_diff == 0.0
    )

    report = {
        "version": 8,
        "registry_ok": registry_ok,
        "accepted_shard_checks":
            shard_checks,
        "accepted_shard_max_diff":
            shard_max_diff,
        "unrelated_expert_max_diff":
            unrelated_max_diff,
        "router_max_diff":
            router_max_diff,
        "validation_losses":
            losses,
        "integrity_ok":
            integrity_ok,
    }

    RESULT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "GLOBAL V008 INTEGRITY:",
        integrity_ok,
    )

    print(
        "Saved:",
        RESULT,
    )


if __name__ == "__main__":
    main()
