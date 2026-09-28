import torch
import torch.nn.functional as F

from ml.model.transformer import MiniMoELM
from ml.training.synthetic_v2 import (
    VOCAB_SIZE,
    build_split_tensor,
)
from ml.federated.checkpoint_manager import (
    load_global_snapshot,
)


SNAPSHOT = (
    "checkpoints/m10_candidates/"
    "global_step_0100"
)

SPLIT = "validation"

SEQ_LEN = 16

NUM_LAYERS = 3
NUM_EXPERTS = 8

TARGET_DOMAINS = [0, 1]

DOMAIN_NAMES = [
    "D0",
    "D1",
    "D2",
    "D3",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model():
    return MiniMoELM(
        vocab_size=VOCAB_SIZE,
        max_seq_len=32,

        d_model=128,
        num_heads=4,
        num_layers=3,

        num_experts=8,
        top_k=2,

        expert_hidden_dim=256,

        router_aux_loss_weight=0.01,
    )


@torch.no_grad()
def evaluate_domain(
    model,
    domain,
    device,
):
    model.eval()

    sequences = (
        build_split_tensor(
            domain,
            SPLIT,
            SEQ_LEN,
        )
        .to(device)
    )

    x = sequences[:, :-1]
    y = sequences[:, 1:]

    output = model(
        input_ids=x,
    )

    # Recurrence iki başlangıç token'ına
    # ihtiyaç duyduğu için ilk prediction
    # position'ını değerlendirmiyoruz.
    logits = (
        output.logits[:, 1:, :]
    )

    labels = y[:, 1:]

    loss = F.cross_entropy(
        logits.reshape(
            -1,
            logits.shape[-1],
        ),
        labels.reshape(-1),
    )

    predictions = (
        logits.argmax(dim=-1)
    )

    accuracy = (
        predictions == labels
    ).float().mean()

    return {
        "loss": float(
            loss.detach().cpu()
        ),
        "accuracy": float(
            accuracy.detach().cpu()
        ),
    }


def main():
    device = get_device()

    print()
    print("FedEdgeMoE - M10B")
    print(
        "Validation-Only Shared Expert Analysis"
    )

    print()
    print("Snapshot:", SNAPSHOT)
    print("Split:", SPLIT)
    print("Device:", device)

    print()
    print(
        "SEALED TEST SPLIT IS NOT USED."
    )

    model = build_model()

    load_global_snapshot(
        model,
        SNAPSHOT,
    )

    model = model.to(device)
    model.eval()

    baseline = {}

    print()
    print("=" * 80)
    print("VALIDATION BASELINE")
    print("=" * 80)

    for domain in TARGET_DOMAINS:
        result = evaluate_domain(
            model,
            domain,
            device,
        )

        baseline[domain] = result

        print(
            f"{DOMAIN_NAMES[domain]} "
            f"loss={result['loss']:.6f} "
            f"acc={result['accuracy'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    routing = torch.zeros(
        NUM_LAYERS,
        len(TARGET_DOMAINS),
        NUM_EXPERTS,
        dtype=torch.long,
    )

    captured = [
        None
        for _ in range(NUM_LAYERS)
    ]

    hooks = []

    for layer_idx, block in enumerate(
        model.blocks
    ):
        def make_hook(index):
            def hook(
                module,
                inputs,
                output,
            ):
                captured[index] = (
                    output.topk_indices
                    .detach()
                    .cpu()
                )

            return hook

        hooks.append(
            block.moe.router
            .register_forward_hook(
                make_hook(layer_idx)
            )
        )

    with torch.no_grad():
        for domain_index, domain in enumerate(
            TARGET_DOMAINS
        ):
            sequences = (
                build_split_tensor(
                    domain,
                    SPLIT,
                    SEQ_LEN,
                )
                .to(device)
            )

            x = sequences[:, :-1]

            _ = model(
                input_ids=x,
            )

            for layer in range(
                NUM_LAYERS
            ):
                # Prediction değerlendirmesinde
                # kullanılan positions ile aynı
                # kısmın routing'ini say.
                selected = (
                    captured[layer][:, 1:, :]
                    .reshape(-1)
                )

                routing[
                    layer,
                    domain_index,
                ] = torch.bincount(
                    selected,
                    minlength=NUM_EXPERTS,
                )

    for handle in hooks:
        handle.remove()

    print()
    print("=" * 80)
    print("VALIDATION ROUTING")
    print("=" * 80)

    for layer in range(
        NUM_LAYERS
    ):
        print()
        print(f"Layer {layer}")

        for domain_index, domain in enumerate(
            TARGET_DOMAINS
        ):
            counts = routing[
                layer,
                domain_index,
            ].float()

            percentages = (
                counts
                / counts.sum()
                * 100
            )

            print(
                f"{DOMAIN_NAMES[domain]}  "
                + " ".join(
                    f"E{expert}:"
                    f"{percentage:5.1f}%"
                    for expert, percentage
                    in enumerate(
                        percentages.tolist()
                    )
                )
            )

    # --------------------------------------------------------
    # ABLATION
    # --------------------------------------------------------

    results = []

    print()
    print("=" * 80)
    print(
        "VALIDATION SINGLE-EXPERT "
        "ZERO-ABLATION"
    )
    print("=" * 80)

    for layer in range(
        NUM_LAYERS
    ):
        for expert_index in range(
            NUM_EXPERTS
        ):
            expert = (
                model
                .blocks[layer]
                .moe
                .experts[expert_index]
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

            current = {}

            for domain in TARGET_DOMAINS:
                current[domain] = (
                    evaluate_domain(
                        model,
                        domain,
                        device,
                    )
                )

            handle.remove()

            delta_loss_d0 = (
                current[0]["loss"]
                - baseline[0]["loss"]
            )

            delta_loss_d1 = (
                current[1]["loss"]
                - baseline[1]["loss"]
            )

            acc_drop_d0 = (
                baseline[0]["accuracy"]
                - current[0]["accuracy"]
            )

            acc_drop_d1 = (
                baseline[1]["accuracy"]
                - current[1]["accuracy"]
            )

            shared_loss_score = min(
                max(
                    delta_loss_d0,
                    0.0,
                ),
                max(
                    delta_loss_d1,
                    0.0,
                ),
            )

            shared_acc_score = min(
                max(
                    acc_drop_d0,
                    0.0,
                ),
                max(
                    acc_drop_d1,
                    0.0,
                ),
            )

            routing_d0 = float(
                routing[
                    layer,
                    0,
                    expert_index,
                ]
            )

            routing_d1 = float(
                routing[
                    layer,
                    1,
                    expert_index,
                ]
            )

            routing_d0 /= float(
                routing[
                    layer,
                    0,
                ].sum()
            )

            routing_d1 /= float(
                routing[
                    layer,
                    1,
                ].sum()
            )

            shared_routing = min(
                routing_d0,
                routing_d1,
            )

            results.append({
                "layer":
                    layer,

                "expert":
                    expert_index,

                "delta_loss_d0":
                    delta_loss_d0,

                "delta_loss_d1":
                    delta_loss_d1,

                "acc_drop_d0":
                    acc_drop_d0,

                "acc_drop_d1":
                    acc_drop_d1,

                "shared_loss_score":
                    shared_loss_score,

                "shared_acc_score":
                    shared_acc_score,

                "shared_routing":
                    shared_routing,
            })

            print(
                f"L{layer} E{expert_index}: "
                f"D0_loss={delta_loss_d0:+.5f} "
                f"D1_loss={delta_loss_d1:+.5f} "
                f"D0_acc={acc_drop_d0 * 100:+.2f}pp "
                f"D1_acc={acc_drop_d1 * 100:+.2f}pp"
            )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    candidates = sorted(
        results,
        key=lambda item: (
            item[
                "shared_loss_score"
            ],
            item[
                "shared_acc_score"
            ],
            item[
                "shared_routing"
            ],
        ),
        reverse=True,
    )

    print()
    print("=" * 80)
    print(
        "BEST SHARED EXPERT CANDIDATES"
    )
    print("=" * 80)

    for rank, item in enumerate(
        candidates[:10],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"L{item['layer']}-E{item['expert']} "
            f"| shared_loss="
            f"{item['shared_loss_score']:+.6f} "
            f"| shared_acc="
            f"{item['shared_acc_score'] * 100:+.2f}pp "
            f"| shared_route="
            f"{item['shared_routing'] * 100:5.1f}% "
            f"| D0_loss="
            f"{item['delta_loss_d0']:+.6f} "
            f"D1_loss="
            f"{item['delta_loss_d1']:+.6f}"
        )

    print()
    print(
        "TEST SPLIT REMAINS SEALED."
    )


if __name__ == "__main__":
    main()
