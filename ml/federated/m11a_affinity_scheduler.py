import json
from pathlib import Path

import torch

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

M10_REPORT = Path(
    "results/m10c_clean_validation_fl.json"
)

REPORT_PATH = Path(
    "results/m11a_affinity_scheduler.json"
)

SEQ_LEN = 16

NUM_LAYERS = 3
NUM_EXPERTS = 8

CLIENTS = {
    "client_A": {
        "domain": 0,
        "device": "Mac MPS",
    },

    "client_B": {
        "domain": 1,
        "device": "MSI CUDA",
    },
}


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


def collect_client_routing(
    model,
    domain,
    device,
):
    """
    Yalnız TRAIN split routing telemetry.
    Validation ve test kullanılmaz.
    """

    routing_counts = torch.zeros(
        NUM_LAYERS,
        NUM_EXPERTS,
        dtype=torch.long,
    )

    captured = [
        None
        for _ in range(NUM_LAYERS)
    ]

    hooks = []

    for layer_index, block in enumerate(
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
                make_hook(layer_index)
            )
        )

    sequences = (
        build_split_tensor(
            domain,
            "train",
            SEQ_LEN,
        )
        .to(device)
    )

    # Küçük prototip olduğu için bütün train split
    # tek forward pass'te işlenebilir.
    x = sequences[:, :-1]

    model.eval()

    with torch.no_grad():
        _ = model(
            input_ids=x,
        )

    for layer in range(
        NUM_LAYERS
    ):
        # Recurrence prediction'da kullanılan
        # positions ile aynı kısmı ölç.
        selected = (
            captured[layer][:, 1:, :]
            .reshape(-1)
        )

        routing_counts[layer] = (
            torch.bincount(
                selected,
                minlength=NUM_EXPERTS,
            )
        )

    for hook in hooks:
        hook.remove()

    routing_share = (
        routing_counts.float()
        / routing_counts.sum(
            dim=1,
            keepdim=True,
        )
    )

    return {
        "counts":
            routing_counts,

        "shares":
            routing_share,

        "num_sequences":
            int(
                sequences.shape[0]
            ),
    }


def expert_parameter_count(
    model,
    layer,
    expert,
):
    module = (
        model
        .blocks[layer]
        .moe
        .experts[expert]
    )

    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
    )


def main():
    device = get_device()

    print()
    print("FedEdgeMoE - M11A")
    print(
        "Train-Routing Affinity Scheduler"
    )

    print()
    print("Model:", SNAPSHOT)
    print("Analysis device:", device)

    print()
    print(
        "SCHEDULER INPUT: TRAIN ROUTING ONLY"
    )

    print(
        "VALIDATION: NOT USED"
    )

    print(
        "TEST: NOT USED"
    )

    model = build_model()

    load_global_snapshot(
        model,
        SNAPSHOT,
    )

    model = model.to(device)

    client_results = {}

    # --------------------------------------------------------
    # CLIENT TELEMETRY
    # --------------------------------------------------------

    for client_id, info in (
        CLIENTS.items()
    ):
        domain = info["domain"]

        routing = (
            collect_client_routing(
                model,
                domain,
                device,
            )
        )

        client_results[
            client_id
        ] = routing

        print()
        print("=" * 82)

        print(
            f"{client_id} "
            f"/ D{domain} "
            f"/ {info['device']}"
        )

        print("=" * 82)

        for layer in range(
            NUM_LAYERS
        ):
            shares = (
                routing[
                    "shares"
                ][layer]
                * 100
            )

            text = " ".join(
                f"E{expert}:"
                f"{shares[expert].item():5.1f}%"

                for expert in range(
                    NUM_EXPERTS
                )
            )

            print(
                f"Layer {layer}: "
                f"{text}"
            )

    # --------------------------------------------------------
    # BUILD SHARED-EXPERT CANDIDATES
    # --------------------------------------------------------

    candidates = []

    share_a = (
        client_results[
            "client_A"
        ]["shares"]
    )

    share_b = (
        client_results[
            "client_B"
        ]["shares"]
    )

    for layer in range(
        NUM_LAYERS
    ):
        for expert in range(
            NUM_EXPERTS
        ):
            affinity_a = float(
                share_a[
                    layer,
                    expert,
                ]
            )

            affinity_b = float(
                share_b[
                    layer,
                    expert,
                ]
            )

            shared_min = min(
                affinity_a,
                affinity_b,
            )

            shared_mean = (
                affinity_a
                + affinity_b
            ) / 2

            imbalance = abs(
                affinity_a
                - affinity_b
            )

            params = (
                expert_parameter_count(
                    model,
                    layer,
                    expert,
                )
            )

            candidates.append({
                "layer":
                    layer,

                "expert":
                    expert,

                "client_A_affinity":
                    affinity_a,

                "client_B_affinity":
                    affinity_b,

                "shared_min_affinity":
                    shared_min,

                "shared_mean_affinity":
                    shared_mean,

                "affinity_imbalance":
                    imbalance,

                "parameter_count":
                    params,
            })

    # Primary:
    # yüksek minimum ortak affinity.
    #
    # Tie-break:
    # yüksek mean affinity.
    #
    # Sonraki tie-break:
    # daha düşük imbalance.
    candidates.sort(
        key=lambda item: (
            item[
                "shared_min_affinity"
            ],

            item[
                "shared_mean_affinity"
            ],

            -item[
                "affinity_imbalance"
            ],
        ),
        reverse=True,
    )

    print()
    print("=" * 82)
    print(
        "DYNAMIC SHARED-EXPERT RANKING"
    )
    print("=" * 82)

    for rank, item in enumerate(
        candidates[:12],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"L{item['layer']}-E{item['expert']} "
            f"| A="
            f"{item['client_A_affinity'] * 100:5.1f}% "
            f"B="
            f"{item['client_B_affinity'] * 100:5.1f}% "
            f"| shared_min="
            f"{item['shared_min_affinity'] * 100:5.1f}% "
            f"| mean="
            f"{item['shared_mean_affinity'] * 100:5.1f}% "
            f"| imbalance="
            f"{item['affinity_imbalance'] * 100:5.1f}%"
        )

    selected = candidates[0]

    print()
    print("=" * 82)
    print("SCHEDULER DECISION")
    print("=" * 82)

    print(
        "Selected expert:",
        f"L{selected['layer']}-E"
        f"{selected['expert']}",
    )

    print(
        "Client A affinity:",
        f"{selected['client_A_affinity'] * 100:.2f}%",
    )

    print(
        "Client B affinity:",
        f"{selected['client_B_affinity'] * 100:.2f}%",
    )

    print(
        "Shared minimum affinity:",
        f"{selected['shared_min_affinity'] * 100:.2f}%",
    )

    print(
        "Expert parameters:",
        f"{selected['parameter_count']:,}",
    )

    # --------------------------------------------------------
    # COMPARE WITH M10 MANUAL/VALIDATION SELECTION
    # --------------------------------------------------------

    manual_layer = None
    manual_expert = None

    if M10_REPORT.exists():
        with open(
            M10_REPORT,
            "r",
            encoding="utf-8",
        ) as f:
            m10 = json.load(f)

        manual_layer = (
            m10.get(
                "target_layer"
            )
        )

        manual_expert = (
            m10.get(
                "target_expert"
            )
        )

        print()
        print(
            "M10 selected expert:",
            f"L{manual_layer}-E"
            f"{manual_expert}",
        )

        same = (
            manual_layer
            == selected["layer"]
            and
            manual_expert
            == selected["expert"]
        )

        print(
            "Scheduler matches M10:",
            same,
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_clients = {}

    for client_id, routing in (
        client_results.items()
    ):
        serializable_clients[
            client_id
        ] = {
            "domain":
                CLIENTS[
                    client_id
                ]["domain"],

            "device":
                CLIENTS[
                    client_id
                ]["device"],

            "num_sequences":
                routing[
                    "num_sequences"
                ],

            "routing_shares":
                routing[
                    "shares"
                ].tolist(),
        }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model":
                    SNAPSHOT,

                "data_split":
                    "train",

                "validation_used":
                    False,

                "test_used":
                    False,

                "scheduler":
                    (
                        "max shared minimum "
                        "routing affinity"
                    ),

                "clients":
                    serializable_clients,

                "ranking":
                    candidates,

                "selected":
                    selected,

                "m10_reference":
                    {
                        "layer":
                            manual_layer,

                        "expert":
                            manual_expert,
                    },
            },
            f,
            indent=2,
        )

    print()
    print(
        "Report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()
