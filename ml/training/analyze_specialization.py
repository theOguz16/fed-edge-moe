import torch

from ml.model.transformer import MiniMoELM
from ml.training.synthetic import generate_batch


NUM_DOMAINS = 4
NUM_LAYERS = 3
NUM_EXPERTS = 8

BATCH_SIZE = 256
NUM_BATCHES = 20
SEQ_LEN = 16


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    device = get_device()

    print("Device:", device)

    model = MiniMoELM(
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

    checkpoint = torch.load(
        "checkpoints/m1/minimoe_m1.pt",
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # [layer, domain, expert]
    counts = torch.zeros(
        NUM_LAYERS,
        NUM_DOMAINS,
        NUM_EXPERTS,
        dtype=torch.long,
    )

    captured_routes = [None] * NUM_LAYERS

    hooks = []

    for layer_idx, block in enumerate(model.blocks):

        def make_hook(idx):
            def hook(module, inputs, output):
                captured_routes[idx] = (
                    output.topk_indices
                    .detach()
                    .cpu()
                )

            return hook

        handle = block.moe.router.register_forward_hook(
            make_hook(layer_idx)
        )

        hooks.append(handle)

    with torch.no_grad():

        for _ in range(NUM_BATCHES):

            input_ids, labels, domain_ids = generate_batch(
                batch_size=BATCH_SIZE,
                seq_len=SEQ_LEN,
                device=device,
            )

            _ = model(
                input_ids=input_ids,
                labels=labels,
            )

            domain_cpu = domain_ids.detach().cpu()

            for layer_idx in range(NUM_LAYERS):

                routes = captured_routes[layer_idx]

                for domain in range(NUM_DOMAINS):

                    mask = domain_cpu == domain

                    selected = routes[mask]

                    flat = selected.reshape(-1)

                    domain_counts = torch.bincount(
                        flat,
                        minlength=NUM_EXPERTS,
                    )

                    counts[
                        layer_idx,
                        domain,
                    ] += domain_counts

    for handle in hooks:
        handle.remove()

    domain_names = [
        "D0 (+1)",
        "D1 (+2)",
        "D2 (+3)",
        "D3 (+5)",
    ]

    print()
    print("=" * 76)
    print("DOMAIN -> EXPERT ROUTING MATRIX")
    print("=" * 76)

    for layer in range(NUM_LAYERS):

        print()
        print(f"LAYER {layer}")
        print(
            "Domain      "
            + " ".join(
                f"E{i:>5}"
                for i in range(NUM_EXPERTS)
            )
        )

        for domain in range(NUM_DOMAINS):

            row = counts[
                layer,
                domain,
            ].float()

            percentages = (
                row / row.sum()
            ) * 100

            values = " ".join(
                f"{p:5.1f}%"
                for p in percentages.tolist()
            )

            print(
                f"{domain_names[domain]:<10} {values}"
            )

        print()

        # Expert -> favorite domain
        print("Expert preferred domains:")

        for expert in range(NUM_EXPERTS):

            expert_domain_counts = counts[
                layer,
                :,
                expert,
            ].float()

            total = expert_domain_counts.sum()

            if total == 0:
                print(
                    f"E{expert}: unused"
                )
                continue

            distribution = (
                expert_domain_counts / total
            )

            favorite = int(
                distribution.argmax()
            )

            strength = float(
                distribution[favorite]
            )

            print(
                f"E{expert}: "
                f"{domain_names[favorite]} "
                f"({strength * 100:.1f}% "
                f"of its routes)"
            )


if __name__ == "__main__":
    main()
