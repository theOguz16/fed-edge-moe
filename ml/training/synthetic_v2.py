import random

import torch


VOCAB_SIZE = 64
TOKENS_PER_DOMAIN = 16
NUM_DOMAINS = 4


# x_t = (
#     a * x_(t-1)
#     + b * x_(t-2)
#     + c
# ) mod 16
#
# Her domain farklı bir recurrence kullanıyor.
DOMAIN_RULES = [
    (1, 1, 1),
    (2, 1, 3),
    (1, 2, 5),
    (3, 2, 7),
]


def pair_bucket(
    domain,
    first,
    second,
):
    # Deterministik split.
    #
    # Aynı initial pair her çalıştırmada
    # aynı split'te kalır.
    return (
        first * 17
        + second * 31
        + domain * 13
    ) % 20


def pair_split(
    domain,
    first,
    second,
):
    bucket = pair_bucket(
        domain,
        first,
        second,
    )

    if bucket < 14:
        return "train"

    if bucket < 17:
        return "validation"

    return "test"


def get_pairs(
    domain,
    split,
):
    pairs = []

    for first in range(
        TOKENS_PER_DOMAIN
    ):
        for second in range(
            TOKENS_PER_DOMAIN
        ):
            if (
                pair_split(
                    domain,
                    first,
                    second,
                )
                == split
            ):
                pairs.append(
                    (first, second)
                )

    return pairs


def build_sequence(
    domain,
    first,
    second,
    seq_len,
):
    a, b, c = (
        DOMAIN_RULES[domain]
    )

    values = [
        first,
        second,
    ]

    while len(values) < (
        seq_len + 1
    ):
        next_value = (
            a * values[-1]
            + b * values[-2]
            + c
        ) % TOKENS_PER_DOMAIN

        values.append(
            next_value
        )

    base = (
        domain
        * TOKENS_PER_DOMAIN
    )

    return [
        base + value
        for value in values
    ]


def build_split_tensor(
    domain,
    split,
    seq_len,
):
    pairs = get_pairs(
        domain,
        split,
    )

    sequences = [
        build_sequence(
            domain,
            first,
            second,
            seq_len,
        )
        for first, second in pairs
    ]

    return torch.tensor(
        sequences,
        dtype=torch.long,
    )


def generate_balanced_batch(
    split,
    batch_size,
    seq_len,
    device,
):
    if batch_size % NUM_DOMAINS != 0:
        raise ValueError(
            "batch_size must be "
            "divisible by NUM_DOMAINS"
        )

    per_domain = (
        batch_size
        // NUM_DOMAINS
    )

    batches = []

    for domain in range(
        NUM_DOMAINS
    ):
        pool = build_split_tensor(
            domain,
            split,
            seq_len,
        )

        indices = torch.randint(
            0,
            pool.shape[0],
            (per_domain,),
        )

        batches.append(
            pool[indices]
        )

    sequences = torch.cat(
        batches,
        dim=0,
    )

    permutation = torch.randperm(
        sequences.shape[0]
    )

    sequences = (
        sequences[permutation]
        .to(device)
    )

    return (
        sequences[:, :-1],
        sequences[:, 1:],
    )


def print_split_summary():
    print(
        "Synthetic V2 split sizes"
    )

    print()

    for domain in range(
        NUM_DOMAINS
    ):
        train = len(
            get_pairs(
                domain,
                "train",
            )
        )

        validation = len(
            get_pairs(
                domain,
                "validation",
            )
        )

        test = len(
            get_pairs(
                domain,
                "test",
            )
        )

        print(
            f"D{domain}: "
            f"train={train:3d} "
            f"validation={validation:3d} "
            f"test={test:3d} "
            f"total={train + validation + test}"
        )
