import torch


VOCAB_SIZE = 64
TOKENS_PER_DOMAIN = 16
NUM_DOMAINS = 4


# M11E başlamadan önce sabitlenen yeni kurallar.
#
# x_t = (
#     a * x_(t-1)
#     + b * x_(t-2)
#     + c
# ) mod 16
DOMAIN_RULES = [
    (1, 1, 3),
    (2, 1, 7),
    (1, 3, 5),
    (3, 1, 11),
]


SPLIT_SALT = 11


def pair_bucket(
    domain,
    first,
    second,
):
    return (
        first * 37
        + second * 61
        + domain * 17
        + SPLIT_SALT
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
        value = (
            a * values[-1]
            + b * values[-2]
            + c
        ) % TOKENS_PER_DOMAIN

        values.append(
            value
        )

    offset = (
        domain
        * TOKENS_PER_DOMAIN
    )

    return [
        offset + value
        for value in values
    ]


def build_split_tensor(
    domain,
    split,
    seq_len,
):
    sequences = [
        build_sequence(
            domain,
            first,
            second,
            seq_len,
        )
        for first, second
        in get_pairs(
            domain,
            split,
        )
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
            "batch_size must be divisible "
            "by number of domains"
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
        "Synthetic V3 split sizes"
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
            f"total="
            f"{train + validation + test}"
        )
