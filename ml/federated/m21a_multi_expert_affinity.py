import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen2MoeForCausalLM

from ml.training.synthetic_v2 import build_split_tensor


BASE_MODEL = Path(
    "checkpoints/m19a/tiny_qwen2_global_v000"
)

GLOBAL_E0 = Path(
    "checkpoints/m20/global_v006/L0_E0.safetensors"
)

OUT = Path(
    "results/m21a_multi_expert_affinity.json"
)


def apply_e0_shard(model, shard):
    experts = model.model.layers[0].mlp.experts

    with torch.no_grad():
        experts.gate_up_proj[0].copy_(
            torch.cat(
                [
                    shard["gate_proj"],
                    shard["up_proj"],
                ],
                dim=0,
            )
        )

        experts.down_proj[0].copy_(
            shard["down_proj"]
        )


def main():
    torch.manual_seed(20260928)

    model = Qwen2MoeForCausalLM.from_pretrained(
        BASE_MODEL
    )

    shard = load_file(str(GLOBAL_E0))
    apply_e0_shard(model, shard)

    model.eval()

    num_layers = len(model.model.layers)

    num_experts = (
        model.config.num_experts
    )

    top_k = (
        model.config.num_experts_per_tok
    )

    current_domain = {"value": None}

    counts = defaultdict(
        lambda: defaultdict(
            lambda: [0] * num_experts
        )
    )

    totals = defaultdict(
        lambda: defaultdict(int)
    )

    hooks = []

    for layer_idx, layer in enumerate(
        model.model.layers
    ):
        mlp = layer.mlp

        def make_hook(layer_index):
            def hook(module, args):
                domain = current_domain["value"]

                if domain is None:
                    return

                hidden = args[0]

                with torch.no_grad():
                    gate_output = module.gate(hidden)

                    if isinstance(gate_output, tuple):
                        selected = gate_output[-1]
                    else:
                        selected = torch.topk(
                            gate_output,
                            k=top_k,
                            dim=-1,
                        ).indices

                    flat = selected.reshape(-1)

                    bincount = torch.bincount(
                        flat.cpu(),
                        minlength=num_experts,
                    )

                    for expert_idx in range(
                        num_experts
                    ):
                        counts[domain][
                            layer_index
                        ][expert_idx] += int(
                            bincount[
                                expert_idx
                            ]
                        )

                    totals[domain][
                        layer_index
                    ] += flat.numel()

            return hook

        hooks.append(
            mlp.register_forward_pre_hook(
                make_hook(layer_idx)
            )
        )

    with torch.no_grad():
        for domain in range(4):
            current_domain["value"] = domain

            x = build_split_tensor(
                domain,
                "validation",
                16,
            )

            # Aynı kontrollü subset.
            x = x[:64]

            model(
                input_ids=x,
            )

    current_domain["value"] = None

    for h in hooks:
        h.remove()

    print()
    print(
        "FedEdgeMoE - M21A"
    )
    print(
        "Multi-Expert Routing Affinity"
    )
    print()

    report = {
        "global_version": "V006",
        "top_k": top_k,
        "num_experts": num_experts,
        "domains": {},
    }

    for domain in range(4):
        print(f"=== D{domain} ===")

        report["domains"][
            f"D{domain}"
        ] = {}

        for layer_idx in range(num_layers):
            total = totals[
                domain
            ][layer_idx]

            row = []

            for expert_idx in range(
                num_experts
            ):
                c = counts[
                    domain
                ][layer_idx][expert_idx]

                pct = (
                    100.0 * c / total
                    if total
                    else 0.0
                )

                row.append(
                    {
                        "expert":
                            expert_idx,
                        "count":
                            c,
                        "share_pct":
                            pct,
                    }
                )

            row.sort(
                key=lambda x:
                    x["share_pct"],
                reverse=True,
            )

            report["domains"][
                f"D{domain}"
            ][
                f"layer_{layer_idx}"
            ] = row

            print(
                f"Layer {layer_idx}:"
            )

            for item in row:
                print(
                    f"  E{item['expert']}: "
                    f"{item['count']:4d} "
                    f"({item['share_pct']:.2f}%)"
                )

        print()

    # Aggregate each expert across layers.
    aggregate = {}

    for domain in range(4):
        expert_totals = [
            0
        ] * num_experts

        total_slots = 0

        for layer_idx in range(
            num_layers
        ):
            total_slots += totals[
                domain
            ][layer_idx]

            for e in range(
                num_experts
            ):
                expert_totals[e] += (
                    counts[
                        domain
                    ][layer_idx][e]
                )

        shares = [
            (
                100.0
                * expert_totals[e]
                / total_slots
                if total_slots
                else 0.0
            )
            for e in range(num_experts)
        ]

        aggregate[f"D{domain}"] = {
            f"E{e}": shares[e]
            for e in range(num_experts)
        }

    report[
        "aggregate_affinity_pct"
    ] = aggregate

    print(
        "=== Aggregate affinity "
        "across layers ==="
    )

    for domain in range(4):
        ordered = sorted(
            aggregate[
                f"D{domain}"
            ].items(),
            key=lambda x: x[1],
            reverse=True,
        )

        print(
            f"D{domain}: "
            + " | ".join(
                f"{e}={p:.2f}%"
                for e, p in ordered
            )
        )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
