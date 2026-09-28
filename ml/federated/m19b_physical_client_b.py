import time
from pathlib import Path

import torch
from safetensors.torch import save_file
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


CHECKPOINT = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

OUTPUT = Path(
    "checkpoints/m19b/clientB_E0_lora.safetensors"
)

DOMAIN = 1
RANK = 4
ALPHA = 4.0
LR = 1e-3
STEPS = 80
BATCH_SIZE = 32


@torch.no_grad()
def val_loss(model, device):
    model.eval()

    seq = build_split_tensor(
        DOMAIN,
        "validation",
        16,
    )[:64].to(device)

    labels = seq.clone()
    labels[:, :2] = -100

    out = model(
        input_ids=seq,
        labels=labels,
    )

    return float(out.loss.cpu())


def main():
    torch.manual_seed(20260927)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device("cuda")

    model = (
        Qwen2MoeForCausalLM
        .from_pretrained(CHECKPOINT)
    )

    for p in model.parameters():
        p.requires_grad = False

    sparse = model.model.layers[0].mlp
    base_experts = sparse.experts

    bridge = Qwen2MoeTensorExpertAdapter(
        base_experts,
        expert_index=0,
    )

    expert = bridge.materialize()

    adapter = MiniMoEExpertAdapter(
        expert
    )

    torch.manual_seed(424242)

    inject_lora(
        adapter,
        rank=RANK,
        alpha=ALPHA,
    )

    hybrid = HybridExperts(
        base_experts,
        expert,
        target_expert=0,
    )

    sparse.experts = hybrid

    model = model.to(device)

    trainable = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    trainable_params = sum(
        p.numel()
        for p in trainable
    )

    before = val_loss(
        model,
        device,
    )

    pool = build_split_tensor(
        DOMAIN,
        "train",
        16,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=0.0,
    )

    torch.manual_seed(20261028)

    torch.cuda.synchronize()
    start = time.perf_counter()

    first_loss = None
    last_loss = None

    model.train()

    for _ in range(STEPS):
        idx = torch.randint(
            0,
            pool.shape[0],
            (BATCH_SIZE,),
        )

        seq = pool[idx].to(device)

        labels = seq.clone()
        labels[:, :2] = -100

        optimizer.zero_grad(
            set_to_none=True
        )

        out = model(
            input_ids=seq,
            labels=labels,
        )

        loss = out.loss
        loss.backward()
        optimizer.step()

        value = float(
            loss.detach().cpu()
        )

        if first_loss is None:
            first_loss = value

        last_loss = value

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    after = val_loss(
        model,
        device,
    )

    lora_state = {
        name: tensor.detach().cpu()
        for name, tensor
        in expert.state_dict().items()
        if "lora_" in name
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        lora_state,
        str(OUTPUT),
    )

    effective = []

    for layer in [
        expert.gate_proj,
        expert.up_proj,
        expert.down_proj,
    ]:
        delta = (
            layer.scaling
            * layer.lora_B
            @ layer.lora_A
        )

        effective.append(
            delta.detach().float().cpu()
        )

    delta_norm = torch.sqrt(
        sum(
            d.pow(2).sum()
            for d in effective
        )
    ).item()

    size = OUTPUT.stat().st_size

    print()
    print("FedEdgeMoE - M19B")
    print("Physical MSI Client B")
    print()

    print(
        "Device:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Trainable params:",
        f"{trainable_params:,}",
    )

    print(
        "Routed E0 tokens:",
        hybrid.routed_tokens,
    )

    print(
        "Train loss:",
        f"{first_loss:.6f}",
        "->",
        f"{last_loss:.6f}",
    )

    print(
        "D1 validation loss:",
        f"{before:.6f}",
        "->",
        f"{after:.6f}",
    )

    print(
        "Effective delta norm:",
        f"{delta_norm:.6f}",
    )

    print(
        "Training seconds:",
        f"{elapsed:.3f}",
    )

    print(
        "Upload bytes:",
        f"{size:,}",
    )

    print()

    print(
        "PHYSICAL CLIENT B OK:",
        (
            trainable_params == 1152
            and hybrid.routed_tokens > 0
            and delta_norm > 0
            and size < 10_000
        ),
    )

    print(
        "Saved:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
