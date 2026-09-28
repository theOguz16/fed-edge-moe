import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2MoeForCausalLM

from ml.adapters.expert_adapter import MiniMoEExpertAdapter
from ml.adapters.generic_lora import inject_lora
from ml.adapters.qwen2moe_tensor_expert import (
    Qwen2MoeTensorExpertAdapter,
)
from ml.federated.m18b_qwen2moe_full_forward_lora import (
    HybridExperts,
)
from ml.training.synthetic_v2 import build_split_tensor


BASE_CHECKPOINT = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

GLOBAL_E0_V006 = Path(
    "checkpoints/m20/global_v006/L0_E0.safetensors"
)

REGISTRY_SHARDS = [
    (
        6,
        0,
        0,
        Path(
            "checkpoints/m20/global_v006/"
            "L0_E0.safetensors"
        ),
    ),
    (
        7,
        0,
        3,
        Path(
            "checkpoints/m21/global_v007/"
            "L0_E3.safetensors"
        ),
    ),
]

RANK = 4
ALPHA = 4.0
LR = 1e-3
BATCH_SIZE = 32

BASE_SEED = 20260927
LORA_SEED = 424242


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


def apply_expert_shard(
    model,
    layer_idx,
    expert_idx,
    path,
):
    shard = load_file(str(path))

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
        ].copy_(
            gate_up.to(
                experts.gate_up_proj.device
            )
        )

        experts.down_proj[
            expert_idx
        ].copy_(
            shard["down_proj"].to(
                experts.down_proj.device
            )
        )


def apply_global_state(model, version):
    applied = []

    for (
        introduced_version,
        layer_idx,
        expert_idx,
        path,
    ) in REGISTRY_SHARDS:

        if introduced_version > version:
            continue

        if not path.exists():
            raise FileNotFoundError(
                f"Required global shard missing: {path}"
            )

        apply_expert_shard(
            model,
            layer_idx,
            expert_idx,
            path,
        )

        applied.append(
            f"L{layer_idx}-E{expert_idx}"
        )

    return applied


@torch.no_grad()
def validation_loss(
    model,
    domain,
    device,
):
    model.eval()

    seq = build_split_tensor(
        domain,
        "validation",
        16,
    )[:64].to(device)

    labels = seq.clone()
    labels[:, :2] = -100

    output = model(
        input_ids=seq,
        labels=labels,
    )

    return float(
        output.loss.cpu()
    )


def effective_delta(layer):
    return (
        layer.scaling
        * (
            layer.lora_B
            @ layer.lora_A
        )
    ).detach().float().cpu()


def max_diff(a, b):
    return float(
        (
            a.detach().cpu()
            - b.detach().cpu()
        )
        .abs()
        .max()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client-id",
        required=True,
    )

    parser.add_argument(
        "--domain",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--layer",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--expert",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--base-version",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    client_id = args.client_id
    domain = args.domain
    layer_idx = args.layer
    expert_idx = args.expert
    steps = args.steps
    base_version = args.base_version

    device = select_device()

    torch.manual_seed(
        BASE_SEED
    )

    model = (
        Qwen2MoeForCausalLM
        .from_pretrained(
            BASE_CHECKPOINT
        )
    )

    # Reconstruct the exact accepted
    # global state for this round.
    applied_shards = apply_global_state(
        model,
        base_version,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    sparse = (
        model.model
        .layers[layer_idx]
        .mlp
    )

    base_experts = sparse.experts

    # Isolation snapshots.
    gate_up_before = (
        base_experts.gate_up_proj
        .detach()
        .cpu()
        .clone()
    )

    down_before = (
        base_experts.down_proj
        .detach()
        .cpu()
        .clone()
    )

    router_before = {
        name:
            parameter.detach()
            .cpu()
            .clone()
        for name, parameter
        in sparse.gate.named_parameters()
    }

    bridge = (
        Qwen2MoeTensorExpertAdapter(
            base_experts,
            expert_index=expert_idx,
        )
    )

    expert = bridge.materialize()

    adapter = MiniMoEExpertAdapter(
        expert
    )

    # Same LoRA initialization for
    # the same expert across clients.
    torch.manual_seed(
        LORA_SEED
        + layer_idx * 100
        + expert_idx
    )

    inject_lora(
        adapter,
        rank=RANK,
        alpha=ALPHA,
    )

    hybrid = HybridExperts(
        base_experts,
        expert,
        target_expert=expert_idx,
    )

    sparse.experts = hybrid

    model = model.to(device)

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    trainable_params = sum(
        p.numel()
        for p in trainable
    )

    before = validation_loss(
        model,
        domain,
        device,
    )

    pool = build_split_tensor(
        domain,
        "train",
        16,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=0.0,
    )

    # Different data stream by
    # domain / expert.
    torch.manual_seed(
        BASE_SEED
        + domain * 1000
        + layer_idx * 100
        + expert_idx
    )

    first_loss = None
    last_loss = None

    synchronize(device)

    start = time.perf_counter()

    model.train()

    for _ in range(steps):
        indices = torch.randint(
            0,
            pool.shape[0],
            (BATCH_SIZE,),
        )

        seq = pool[
            indices
        ].to(device)

        labels = seq.clone()
        labels[:, :2] = -100

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            input_ids=seq,
            labels=labels,
        )

        loss = output.loss

        loss.backward()
        optimizer.step()

        value = float(
            loss.detach().cpu()
        )

        if first_loss is None:
            first_loss = value

        last_loss = value

    synchronize(device)

    elapsed = (
        time.perf_counter()
        - start
    )

    after = validation_loss(
        model,
        domain,
        device,
    )

    lora_state = {
        name:
            tensor.detach()
            .cpu()
            .clone()
        for name, tensor
        in expert.state_dict().items()
        if "lora_" in name
    }

    output_dir = Path(
        "checkpoints/m21/"
        f"round_v{base_version:03d}_multi_expert"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tag = (
        f"{client_id}_"
        f"L{layer_idx}_E{expert_idx}"
    )

    adapter_path = (
        output_dir
        / f"{tag}_lora.safetensors"
    )

    save_file(
        lora_state,
        str(adapter_path),
    )

    deltas = {
        "gate_proj":
            effective_delta(
                expert.gate_proj
            ),

        "up_proj":
            effective_delta(
                expert.up_proj
            ),

        "down_proj":
            effective_delta(
                expert.down_proj
            ),
    }

    delta_norm = torch.sqrt(
        sum(
            x.pow(2).sum()
            for x in deltas.values()
        )
    ).item()

    adapter_bytes = (
        adapter_path
        .stat()
        .st_size
    )

    # Confirm tensor-bank base weights
    # and router were not changed.
    bank_gate_up_diff = max_diff(
        base_experts.gate_up_proj,
        gate_up_before,
    )

    bank_down_diff = max_diff(
        base_experts.down_proj,
        down_before,
    )

    router_diff = 0.0

    for name, parameter in (
        sparse.gate.named_parameters()
    ):
        router_diff = max(
            router_diff,
            max_diff(
                parameter,
                router_before[name],
            ),
        )

    metadata = {
        "client_id":
            client_id,

        "domain":
            domain,

        "global_version":
            base_version,

        "applied_global_shards":
            applied_shards,

        "target_layer":
            layer_idx,

        "target_expert":
            expert_idx,

        "device":
            str(device),

        "steps":
            steps,

        "trainable_params":
            trainable_params,

        "routed_tokens":
            hybrid.routed_tokens,

        "first_train_loss":
            first_loss,

        "last_train_loss":
            last_loss,

        "validation_before":
            before,

        "validation_after":
            after,

        "local_improvement":
            before - after,

        "delta_norm":
            delta_norm,

        "training_seconds":
            elapsed,

        "adapter_bytes":
            adapter_bytes,

        "base_gate_up_max_diff":
            bank_gate_up_diff,

        "base_down_max_diff":
            bank_down_diff,

        "router_max_diff":
            router_diff,
    }

    metadata_path = (
        output_dir
        / f"{tag}_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    mechanics_ok = (
        trainable_params == 1152
        and hybrid.routed_tokens > 0
        and delta_norm > 0.0
        and adapter_bytes < 10_000
        and bank_gate_up_diff == 0.0
        and bank_down_diff == 0.0
        and router_diff == 0.0
    )

    print()
    print("FedEdgeMoE - M21C")
    print("Generic Multi-Expert Client")
    print()

    print(
        "Client:",
        client_id,
    )

    print(
        "Domain:",
        f"D{domain}",
    )

    print(
        "Target:",
        f"L{layer_idx}-E{expert_idx}",
    )

    print(
        "Global base:",
        f"V{base_version:03d}",
    )

    print(
        "Device:",
        device,
    )

    print()

    print(
        "Local steps:",
        steps,
    )

    print(
        "Train loss:",
        f"{first_loss:.6f}",
        "->",
        f"{last_loss:.6f}",
    )

    print(
        "Validation loss:",
        f"{before:.6f}",
        "->",
        f"{after:.6f}",
    )

    print(
        "Local improvement:",
        f"{before-after:+.6f}",
    )

    print(
        "Delta norm:",
        f"{delta_norm:.6f}",
    )

    print(
        "Routed tokens:",
        hybrid.routed_tokens,
    )

    print(
        "Trainable params:",
        f"{trainable_params:,}",
    )

    print(
        "Adapter bytes:",
        f"{adapter_bytes:,}",
    )

    print(
        "Training seconds:",
        f"{elapsed:.3f}",
    )

    print()

    print(
        "Base expert bank diff:",
        f"{max(bank_gate_up_diff, bank_down_diff):.3e}",
    )

    print(
        "Router diff:",
        f"{router_diff:.3e}",
    )

    print()

    print(
        "CLIENT UPDATE VALID:",
        mechanics_ok,
    )

    print(
        "Adapter:",
        adapter_path,
    )

    print(
        "Metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    main()
