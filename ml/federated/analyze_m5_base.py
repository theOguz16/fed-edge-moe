import torch

from ml.model.transformer import MiniMoELM
from ml.federated.checkpoint_manager import load_global_snapshot


SNAPSHOT = "checkpoints/m5_candidates/global_step_0030"

NUM_LAYERS = 3
NUM_EXPERTS = 8

SEQ_LEN = 16

DOMAIN_NAMES = [
    "D0 (+1)",
    "D1 (+2)",
]

DOMAIN_STEPS = [
    1,
    2,
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


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


def build_domain_set(domain, device):

    starts = torch.arange(
        0,
        16,
        device=device,
    )

    positions = torch.arange(
        SEQ_LEN + 1,
        device=device,
    )

    base = domain * 16
    step_size = DOMAIN_STEPS[domain]

    sequences = (
        starts[:, None]
        + step_size * positions[None, :]
    ) % 16

    sequences = sequences + base

    return (
        sequences[:, :-1].long(),
        sequences[:, 1:].long(),
    )


@torch.no_grad()
def evaluate(model, domain, device):

    x, y = build_domain_set(
        domain,
        device,
    )

    output = model(
        input_ids=x,
        labels=y,
    )

    pred = output.logits.argmax(
        dim=-1
    )

    return {
        "loss": float(
            output.lm_loss.cpu()
        ),

        "accuracy": float(
            (pred == y)
            .float()
            .mean()
            .cpu()
        ),
    }


def main():

    device = get_device()

    print()
    print("FedEdgeMoE - M5 Base Analysis")
    print("Snapshot:", SNAPSHOT)
    print("Device:", device)

    model = build_model()

    load_global_snapshot(
        model,
        SNAPSHOT,
    )

    model = model.to(device)
    model.eval()

    # ========================================================
    # BASELINE
    # ========================================================

    baseline = [
        evaluate(
            model,
            domain,
            device,
        )
        for domain in range(2)
    ]

    print()
    print("=" * 76)
    print("BASELINE")
    print("=" * 76)

    for domain in range(2):

        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss={baseline[domain]['loss']:.6f} "
            f"acc="
            f"{baseline[domain]['accuracy'] * 100:.2f}%"
        )

    # ========================================================
    # ROUTING ANALYSIS
    # ========================================================

    routing = torch.zeros(
        NUM_LAYERS,
        2,
        NUM_EXPERTS,
        dtype=torch.long,
    )

    captured = [None] * NUM_LAYERS
    hooks = []

    for layer_idx, block in enumerate(
        model.blocks
    ):

        def make_hook(idx):

            def hook(
                module,
                inputs,
                output,
            ):
                captured[idx] = (
                    output.topk_indices
                    .detach()
                    .cpu()
                )

            return hook

        hooks.append(
            block.moe.router.register_forward_hook(
                make_hook(layer_idx)
            )
        )

    with torch.no_grad():

        for domain in range(2):

            x, y = build_domain_set(
                domain,
                device,
            )

            _ = model(
                input_ids=x,
                labels=y,
            )

            for layer in range(
                NUM_LAYERS
            ):

                flat = (
                    captured[layer]
                    .reshape(-1)
                )

                routing[
                    layer,
                    domain,
                ] = torch.bincount(
                    flat,
                    minlength=NUM_EXPERTS,
                )

    for hook in hooks:
        hook.remove()

    print()
    print("=" * 76)
    print("ROUTING D0 / D1")
    print("=" * 76)

    for layer in range(NUM_LAYERS):

        print()
        print(f"Layer {layer}")

        for domain in range(2):

            row = routing[
                layer,
                domain,
            ].float()

            percentages = (
                row / row.sum()
            ) * 100

            print(
                f"{DOMAIN_NAMES[domain]:<8}",
                " ".join(
                    f"E{i}:{p:5.1f}%"
                    for i, p in enumerate(
                        percentages.tolist()
                    )
                ),
            )

    # ========================================================
    # FUNCTIONAL ABLATION
    # ========================================================

    results = []

    print()
    print("=" * 76)
    print("SINGLE EXPERT ZERO-ABLATION")
    print("=" * 76)

    for layer in range(NUM_LAYERS):

        for expert_idx in range(
            NUM_EXPERTS
        ):

            expert = (
                model
                .blocks[layer]
                .moe
                .experts[expert_idx]
            )

            def zero_output(
                module,
                inputs,
                output,
            ):
                return torch.zeros_like(
                    output
                )

            handle = (
                expert
                .register_forward_hook(
                    zero_output
                )
            )

            current = [
                evaluate(
                    model,
                    domain,
                    device,
                )
                for domain in range(2)
            ]

            handle.remove()

            delta_d0 = (
                current[0]["loss"]
                - baseline[0]["loss"]
            )

            delta_d1 = (
                current[1]["loss"]
                - baseline[1]["loss"]
            )

            drop_d0 = (
                baseline[0]["accuracy"]
                - current[0]["accuracy"]
            )

            drop_d1 = (
                baseline[1]["accuracy"]
                - current[1]["accuracy"]
            )

            # Ortak expert arıyoruz.
            # İki domain'den yalnızca biri çok güçlü ise
            # federated shared-expert testi için ideal değil.
            shared_score = min(
                delta_d0,
                delta_d1,
            )

            results.append({
                "layer": layer,
                "expert": expert_idx,

                "delta_d0": delta_d0,
                "delta_d1": delta_d1,

                "drop_d0": drop_d0,
                "drop_d1": drop_d1,

                "shared_score":
                    shared_score,
            })

            print(
                f"L{layer} E{expert_idx}: "
                f"D0_loss={delta_d0:+.4f} "
                f"D1_loss={delta_d1:+.4f} "
                f"D0_acc={drop_d0 * 100:+.2f}pp "
                f"D1_acc={drop_d1 * 100:+.2f}pp"
            )

    # ========================================================
    # SHARED EXPERT CANDIDATES
    # ========================================================

    candidates = sorted(
        results,
        key=lambda x: x[
            "shared_score"
        ],
        reverse=True,
    )

    print()
    print("=" * 76)
    print("BEST SHARED EXPERT CANDIDATES")
    print("=" * 76)

    for rank, item in enumerate(
        candidates[:8],
        start=1,
    ):

        print(
            f"{rank}. "
            f"L{item['layer']} "
            f"E{item['expert']} "
            f"shared_score="
            f"{item['shared_score']:+.6f} "
            f"| D0={item['delta_d0']:+.6f} "
            f"D1={item['delta_d1']:+.6f}"
        )


if __name__ == "__main__":
    main()
