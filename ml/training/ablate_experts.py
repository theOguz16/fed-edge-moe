import torch

from ml.model.transformer import MiniMoELM


NUM_DOMAINS = 4
NUM_LAYERS = 3
NUM_EXPERTS = 8

SEQ_LEN = 16
EVAL_SAMPLES = 2048
BATCH_SIZE = 256

DOMAIN_NAMES = [
    "D0 (+1)",
    "D1 (+2)",
    "D2 (+3)",
    "D3 (+5)",
]

DOMAIN_STEPS = [
    1,
    2,
    3,
    5,
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_domain_dataset(domain):
    """
    Her ablation deneyinde AYNI evaluation verisini
    kullanmak için deterministic test set oluşturur.
    """

    generator = torch.Generator()
    generator.manual_seed(1000 + domain)

    starts = torch.randint(
        0,
        16,
        (EVAL_SAMPLES,),
        generator=generator,
    )

    positions = torch.arange(
        SEQ_LEN + 1
    )

    base = domain * 16
    step = DOMAIN_STEPS[domain]

    sequences = (
        starts[:, None]
        + step * positions[None, :]
    ) % 16

    sequences = sequences + base

    input_ids = sequences[:, :-1].long()
    labels = sequences[:, 1:].long()

    return input_ids, labels


@torch.no_grad()
def evaluate_domain(
    model,
    device,
    input_ids,
    labels,
):
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0

    for start in range(
        0,
        input_ids.shape[0],
        BATCH_SIZE,
    ):
        x = input_ids[
            start:start + BATCH_SIZE
        ].to(device)

        y = labels[
            start:start + BATCH_SIZE
        ].to(device)

        output = model(
            input_ids=x,
            labels=y,
        )

        prediction = output.logits.argmax(
            dim=-1
        )

        total_correct += int(
            (prediction == y).sum().cpu()
        )

        total_tokens += y.numel()

        total_loss += float(
            output.lm_loss.cpu()
        )

        batches += 1

    return {
        "loss": total_loss / batches,
        "accuracy": (
            total_correct / total_tokens
        ),
    }


def evaluate_all_domains(
    model,
    device,
    datasets,
):
    results = []

    for domain in range(NUM_DOMAINS):
        x, y = datasets[domain]

        result = evaluate_domain(
            model,
            device,
            x,
            y,
        )

        results.append(result)

    return results


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

    datasets = [
        build_domain_dataset(domain)
        for domain in range(NUM_DOMAINS)
    ]

    print()
    print("=" * 80)
    print("BASELINE")
    print("=" * 80)

    baseline = evaluate_all_domains(
        model,
        device,
        datasets,
    )

    for domain, result in enumerate(
        baseline
    ):
        print(
            f"{DOMAIN_NAMES[domain]:<8} "
            f"loss={result['loss']:.6f} "
            f"acc={result['accuracy'] * 100:6.2f}%"
        )

    experiments = []

    print()
    print("=" * 80)
    print("SINGLE EXPERT ZERO-ABLATION")
    print("=" * 80)

    for layer in range(NUM_LAYERS):

        for expert_id in range(NUM_EXPERTS):

            expert = (
                model
                .blocks[layer]
                .moe
                .experts[expert_id]
            )

            def zero_output(
                module,
                inputs,
                output,
            ):
                return torch.zeros_like(
                    output
                )

            hook = expert.register_forward_hook(
                zero_output
            )

            result = evaluate_all_domains(
                model,
                device,
                datasets,
            )

            hook.remove()

            delta_losses = []

            accuracy_drops = []

            for domain in range(NUM_DOMAINS):

                delta_loss = (
                    result[domain]["loss"]
                    - baseline[domain]["loss"]
                )

                accuracy_drop = (
                    baseline[domain]["accuracy"]
                    - result[domain]["accuracy"]
                )

                delta_losses.append(
                    delta_loss
                )

                accuracy_drops.append(
                    accuracy_drop
                )

            experiments.append({
                "layer": layer,
                "expert": expert_id,
                "delta_losses": delta_losses,
                "accuracy_drops": accuracy_drops,
            })

            loss_text = " ".join([
                (
                    f"{DOMAIN_NAMES[d]} "
                    f"{delta_losses[d]:+.4f}"
                )
                for d in range(NUM_DOMAINS)
            ])

            print(
                f"L{layer} E{expert_id}: "
                + loss_text
            )

    print()
    print("=" * 80)
    print("STRONGEST FUNCTIONAL EXPERTS PER DOMAIN")
    print("=" * 80)

    for domain in range(NUM_DOMAINS):

        ranked = sorted(
            experiments,
            key=lambda x: (
                x["delta_losses"][domain]
            ),
            reverse=True,
        )

        print()
        print(DOMAIN_NAMES[domain])

        for rank, experiment in enumerate(
            ranked[:5],
            start=1,
        ):
            layer = experiment["layer"]
            expert = experiment["expert"]

            delta_loss = (
                experiment[
                    "delta_losses"
                ][domain]
            )

            accuracy_drop = (
                experiment[
                    "accuracy_drops"
                ][domain]
            )

            print(
                f"{rank}. "
                f"L{layer} E{expert} "
                f"delta_loss={delta_loss:+.6f} "
                f"acc_drop={accuracy_drop * 100:+.2f}pp"
            )


if __name__ == "__main__":
    main()
