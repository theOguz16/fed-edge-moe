import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.model.lora_expert import LoRAExpertMLP
from ml.training.synthetic_v2 import build_split_tensor
from ml.federated.checkpoint_manager import load_global_snapshot


CHECKPOINT = (
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

TARGET_LAYER = 0
TARGET_EXPERT = 7

STEPS = 60
BATCH_SIZE = 32
SEQ_LEN = 16
LR = 5e-4


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


def clone_state(module):
    return {
        key: value.detach().cpu().clone()
        for key, value
        in module.state_dict().items()
    }


def max_diff(a, b):
    return max(
        (
            a[key] - b[key]
        ).abs().max().item()
        for key in a
    )


def generate_batch(device):
    pool = build_split_tensor(
        domain=1,
        split="train",
        seq_len=SEQ_LEN,
    )

    indices = torch.randint(
        0,
        pool.shape[0],
        (BATCH_SIZE,),
    )

    seq = pool[indices].to(device)

    x = seq[:, :-1]
    y = seq[:, 1:].clone()

    y[:, 0] = -100

    return x, y


def main():
    torch.manual_seed(20260927)

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model = build_model()

    load_global_snapshot(
        model,
        CHECKPOINT,
    )

    for p in model.parameters():
        p.requires_grad = False

    base_expert = (
        model.blocks[TARGET_LAYER]
        .moe.experts[TARGET_EXPERT]
    )

    lora_expert = LoRAExpertMLP(
        base_expert,
        rank=8,
        alpha=8.0,
    )

    model.blocks[TARGET_LAYER].moe.experts[
        TARGET_EXPERT
    ] = lora_expert

    model = model.to(device)

    base_before = clone_state(
        lora_expert.gate_proj.base
    )
    base_before.update({
        "up." + k: v
        for k, v in clone_state(
            lora_expert.up_proj.base
        ).items()
    })
    base_before.update({
        "down." + k: v
        for k, v in clone_state(
            lora_expert.down_proj.base
        ).items()
    })

    adapter_before = {
        k: v.detach().cpu().clone()
        for k, v
        in lora_expert.adapter_state_dict().items()
    }

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LR,
        weight_decay=0.0,
    )

    first_loss = None
    last_loss = None

    model.train()

    for step in range(
        1,
        STEPS + 1,
    ):
        x, y = generate_batch(
            device
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            input_ids=x,
        )

        loss = F.cross_entropy(
            output.logits.reshape(
                -1,
                64,
            ),
            y.reshape(-1),
        )

        loss.backward()

        optimizer.step()

        if first_loss is None:
            first_loss = float(
                loss.detach().cpu()
            )

        last_loss = float(
            loss.detach().cpu()
        )

    base_after = clone_state(
        lora_expert.gate_proj.base
    )
    base_after.update({
        "up." + k: v
        for k, v in clone_state(
            lora_expert.up_proj.base
        ).items()
    })
    base_after.update({
        "down." + k: v
        for k, v in clone_state(
            lora_expert.down_proj.base
        ).items()
    })

    adapter_after = {
        k: v.detach().cpu().clone()
        for k, v
        in lora_expert.adapter_state_dict().items()
    }

    base_change = max_diff(
        base_before,
        base_after,
    )

    adapter_change = max_diff(
        adapter_before,
        adapter_after,
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("FedEdgeMoE - M16C")
    print("LoRA Training Isolation")
    print()

    print(
        "Device:",
        device,
    )

    print(
        "Trainable params:",
        f"{trainable_params:,}",
    )

    print(
        "First loss:",
        f"{first_loss:.6f}",
    )

    print(
        "Last loss:",
        f"{last_loss:.6f}",
    )

    print(
        "Base expert max change:",
        base_change,
    )

    print(
        "LoRA adapter max change:",
        adapter_change,
    )

    print()
    print(
        "BASE FROZEN:",
        base_change == 0.0,
    )

    print(
        "ADAPTER LEARNED:",
        adapter_change > 0.0,
    )

    print(
        "LORA ISOLATION OK:",
        (
            base_change == 0.0
            and adapter_change > 0.0
            and trainable_params == 9216
        ),
    )


if __name__ == "__main__":
    main()
