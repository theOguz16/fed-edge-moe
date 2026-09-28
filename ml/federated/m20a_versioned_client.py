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

RANK = 4
ALPHA = 4.0
LR = 1e-3

STEPS = 80
BATCH_SIZE = 32

TARGET_LAYER = 0
TARGET_EXPERT = 0

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


def global_shard_path(version):
    if version == 0:
        return None

    return Path(
        f"checkpoints/m20/global_v{version:03d}/"
        "L0_E0.safetensors"
    )


def legacy_v001_path():
    return Path(
        "checkpoints/m19f/global_v001/"
        "L0_E0.safetensors"
    )


def resolve_shard(version):
    if version == 0:
        return None

    path = global_shard_path(
        version
    )

    if path.exists():
        return path

    if version == 1:
        legacy = legacy_v001_path()

        if legacy.exists():
            return legacy

    raise FileNotFoundError(
        f"Global expert shard for "
        f"V{version:03d} not found"
    )


def apply_global_shard(
    model,
    shard_path,
):
    if shard_path is None:
        return

    shard = load_file(
        str(shard_path)
    )

    experts = (
        model.model
        .layers[TARGET_LAYER]
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
            TARGET_EXPERT
        ].copy_(
            gate_up.to(
                experts.gate_up_proj.device
            )
        )

        experts.down_proj[
            TARGET_EXPERT
        ].copy_(
            shard["down_proj"].to(
                experts.down_proj.device
            )
        )


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
        "--base-version",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=80,
    )

    args = parser.parse_args()

    client_id = args.client_id
    domain = args.domain
    base_version = args.base_version
    steps = args.steps

    proposed_version = (
        base_version + 1
    )

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

    shard_path = resolve_shard(
        base_version
    )

    apply_global_shard(
        model,
        shard_path,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    sparse = (
        model.model
        .layers[TARGET_LAYER]
        .mlp
    )

    base_experts = (
        sparse.experts
    )

    bridge = (
        Qwen2MoeTensorExpertAdapter(
            base_experts,
            expert_index=TARGET_EXPERT,
        )
    )

    expert = bridge.materialize()

    adapter = MiniMoEExpertAdapter(
        expert
    )

    # Same LoRA initialization for all
    # clients participating in this round.
    torch.manual_seed(
        LORA_SEED
        + base_version
    )

    inject_lora(
        adapter,
        rank=RANK,
        alpha=ALPHA,
    )

    hybrid = HybridExperts(
        base_experts,
        expert,
        target_expert=TARGET_EXPERT,
    )

    sparse.experts = hybrid

    model = model.to(device)

    trainable = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    trainable_params = sum(
        parameter.numel()
        for parameter
        in trainable
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

    # Deterministic but different
    # batches on each round/domain.
    torch.manual_seed(
        BASE_SEED
        + base_version * 1000
        + domain
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
        f"checkpoints/m20/"
        f"round_v{base_version:03d}_"
        f"to_v{proposed_version:03d}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    adapter_path = (
        output_dir
        / f"{client_id}_E0_lora.safetensors"
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
            tensor.pow(2).sum()
            for tensor
            in deltas.values()
        )
    ).item()

    adapter_bytes = (
        adapter_path
        .stat()
        .st_size
    )

    metadata = {
        "client_id":
            client_id,

        "domain":
            domain,

        "base_version":
            base_version,

        "proposed_version":
            proposed_version,

        "device":
            str(device),

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

        "steps":
            steps,

        "adapter_bytes":
            adapter_bytes,
    }

    metadata_path = (
        output_dir
        / f"{client_id}_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    mechanics_ok = (
        trainable_params == 1152
        and hybrid.routed_tokens > 0
        and delta_norm > 0.0
        and adapter_bytes < 10_000
    )

    print()
    print("FedEdgeMoE - M20A")
    print("Version-Aware Physical Client")
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
        "Device:",
        device,
    )

    print(
        "Base version:",
        f"V{base_version:03d}",
    )

    print(
        "Proposed version:",
        f"V{proposed_version:03d}",
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
        "Training seconds:",
        f"{elapsed:.3f}",
    )

    print(
        "Adapter bytes:",
        f"{adapter_bytes:,}",
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
