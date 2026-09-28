EXPERT_PARAMS = 98_304

LAYERS = [
    ("gate_proj", 128, 256),
    ("up_proj",   128, 256),
    ("down_proj", 256, 128),
]

RANKS = [2, 4, 8, 16]


def lora_params(
    in_features,
    out_features,
    rank,
):
    return rank * (
        in_features
        + out_features
    )


def main():
    print()
    print("FedEdgeMoE - M16A")
    print("LoRA Parameter Budget")
    print()

    for rank in RANKS:
        total = 0

        for _, inp, out in LAYERS:
            total += lora_params(
                inp,
                out,
                rank,
            )

        ratio = (
            total
            / EXPERT_PARAMS
            * 100
        )

        fp32_kb = (
            total * 4 / 1000
        )

        print(
            f"rank={rank:<2} "
            f"| trainable={total:>6,} "
            f"| expert={ratio:5.2f}% "
            f"| fp32≈{fp32_kb:5.1f} KB"
        )

    print()
    print(
        "Proposed rank: 8"
    )


if __name__ == "__main__":
    main()
